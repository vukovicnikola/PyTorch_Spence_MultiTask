import os
import time
import torch
import matplotlib
from collections import defaultdict
import wandb
from utils import get_predictions_batch

from .callbacks import CallbackGroup


class Loop:
    """
    Implements a training loop, containing a testing and a validation phase.
    Each phase takes a separate dataloader as input and tracks its own metrics.
    Arguments:
        model (nn.Module): PyTorch model to be trained.
        alpha: Weight used to make a smooth moving average between the
               past and current loss value, according  to formula:
               new_loss = old_loss*alpha + (1 - alpha)*new_loss
    """
    def __init__(self, model, loss_fn, optimizer, scheduler, config, alpha: float=0.98,
                 move_to_device=True, device=None, timestamp=None):

        # move model to GPU, if specified
        if move_to_device:
            device = torch.device(device or 'cpu')
            model = model.to(device)
            self.device = device

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.alpha = alpha
        self.move_to_device = move_to_device
        self.timestamp = timestamp
        self.stop = False
        self.callbacks = None
        self.stepper = None
        self.current_epoch = 0
        self.wandblogger = WandB_Logger(config, model, timestamp)

    def run(self, train_dataloader, valid_dataloader=None,
            epochs: int=100, callbacks=None, metrics=None):

        phases = [Phase(name='train', dataloader=train_dataloader)]
        if valid_dataloader is not None:
            phases.append(Phase(name='valid', dataloader=valid_dataloader))

        cb = CallbackGroup(callbacks)
        cb.set_loop(self)
        cb.training_start(optimizer=self.optimizer, phases=phases)
        self.callbacks = cb
        self.stepper = self.make_stepper(self.loss_fn, metrics)

        for epoch in range(self.current_epoch, epochs):
            if self.stop:
                break
            metrics = {}
            self.wandblogger.clear_images()
            log_pics = 0
            for phase in phases:
                cb.epoch_start(epoch=epoch, phase=phase)
                is_training = phase.name == 'train'
                for idx, batch in enumerate(phase.dataloader):
                    input_batch = self.prepare_batch(batch)
                    phase.batch_num += 1
                    cb.batch_start(epoch=epoch, phase=phase)
                    batch_metrics, model_output = self.stepper.step(input_batch, is_training)
                    cb.batch_end(epoch=epoch, phase=phase, metrics=batch_metrics, model_output=model_output)
                    self.update_metrics(phase, batch_metrics)
                    if phase.name == 'valid':
                        # log examples from first few batches
                        if log_pics < 18:
                            plot = get_predictions_batch(model_output)
                            self.wandblogger.log_batch(predictions=plot)
                            matplotlib.pyplot.close('all')
                            log_pics += 1
                metrics.update({
                    f'{phase.name}_{k}': v
                    for k, v in phase.metrics.items()})

            cb.epoch_end(epoch=epoch, phase=phase, metrics=metrics)
            self.wandblogger.log_epoch(epoch=epoch, metrics=metrics)
        cb.training_end()
        self.wandblogger.cleanup()

    def make_stepper(self, loss_fn, metrics=None, stepper=None):
        stepper_cls = stepper or Stepper
        inst = stepper_cls(
            self.model, self.optimizer, self.scheduler, loss_fn, metrics)
        return inst

    def save_checkpoint(self, epoch, path, metrics):
        self.stepper.save_checkpoint(epoch, path, metrics)

    def load_checkpoint(self, path):
        """
        Loads checkpoint.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['opt_state'])
        self.scheduler.load_state_dict(checkpoint['sched_state'])
        self.current_epoch = checkpoint['epoch']

    def update_metrics(self, phase, batch_metrics):
        a = self.alpha
        updated = {}
        for name, new_value in batch_metrics.items():
            old_value = phase.rolling_metrics[name]
            avg_value = a*old_value + (1 - a)*new_value
            debias_value = avg_value/(1 - a**phase.batch_num)
            updated[name] = debias_value
            phase.rolling_metrics[name] = avg_value
        phase.metrics = updated

    def prepare_batch(self, batch):
        """Perform any unpacking/processing of the batch
        and send to device, if needed.
        """
        if self.move_to_device:
            batch['image'] = batch['image'].to(self.device)
            batch['letter_class'] = batch['letter_class'].to(self.device)
            batch['file_index'] = batch['file_index'].to(self.device)
            batch['keypoints'] = batch['keypoints'].to(self.device)
            batch['num_keypoints'] = batch['num_keypoints'].to(self.device)
        else:
            # do some other unpacking, if needed
            batch = batch
        return batch

    def __getitem__(self, item):
        return self.callbacks[item]


class Phase:
    """
    A model training loop phase.
    Every training loop iteration can be separated into (at least) two
    phases: training and validation. Each Phase instance keeps track of
    a dataloader, and metrics/counters specific to that phase.
        Args:
        name (str): 'train' or 'valid'
    """
    def __init__(self, name: str, dataloader):
        self.name = name
        self.dataloader = dataloader
        self.batch_num = 0
        self.rolling_metrics = defaultdict(lambda: 0)
        self.metrics = None

    def __repr__(self):
        if self.metrics is None:
            return f'<Phase: {self.name}, metrics: none>'
        metrics = ', '.join([
            f'{key}={value:2.4f}'
            for key, value in self.metrics.items()])
        return f'<Phase: {self.name}, metrics: {metrics}>'


class Stepper:
    """
    A simple class containing the model, optimizer, and a
    loss function. Stepper instances are invoked during each training
    iteration and return the loss on a batch.
    """
    def __init__(self, model, optimizer, scheduler, loss, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.metrics = metrics

    def step(self, input_batch, train: bool=True):
        """
        Performs a single training step.
        Args:
            input_batch: Dictionary of features and targets.
            train: If True, computes gradients and updates model parameters.
        Returns:
            loss: The loss value on batch.
        """
        metrics = {}
        self.model.train(train)

        with torch.set_grad_enabled(train):
            model_output = self.model(input_batch)
            combined_loss, class_l, kp_l = self.loss(model_output)
            metrics['loss'] = combined_loss.item()
            metrics['class loss'] = class_l.item()
            metrics['keypoint loss'] = kp_l.item()

            if self.metrics is not None:
                for metric in self.metrics:
                    metrics[metric.__name__] = metric(output=model_output, optimizer=self.optimizer)

            if train:
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        return metrics, model_output

    def save_checkpoint(self, epoch, path: str, metrics):
        """
        Saves model/optimizer/scheduler state into file.
        """
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'model_state': self.model.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict()
        }
        try:
            torch.save(checkpoint, path)
        except Exception:
            print(f"error saving checkpoint to {path}")


class WandB_Logger:
    '''A class to handle metrics logging to Weights and Biases
    '''
    def __init__(self, config, model, timestamp):
        self.example_images = []
        # Initialize Wandb
        self.run = wandb.init(
                               config=config,
                               project=config['wandb_project'],
                               dir=config['model_dir'],
                               name=timestamp,
                               id=timestamp,
                               resume=True
                               )

        # WandB – log all layers, gradients, and model parameters
        wandb.watch(model, log='all')

    def clear_images(self):
        self.example_images = []

    def log_batch(self, predictions=None, **kwargs):
        '''Log batch sample predictions'''
        self.example_images.append(
                wandb.Image(predictions))

    def log_epoch(self, epoch=None, metrics=None, **kwargs):
        wandb.log({'Epoch': epoch, 'Examples': self.example_images, **metrics})

    def cleanup(self):
        os.system('wandb gc')
