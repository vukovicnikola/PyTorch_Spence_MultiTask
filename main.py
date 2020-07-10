import os
import argparse
import math
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import utils, transforms
import wandb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from metrics import MultiLoss
from models import SpenceNet
from collections import OrderedDict
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from datasets import LetterDataset, LetterBatchSampler
from datasets import LetterTransforms as lt
from datetime import datetime
from pytz import timezone, utc


def train(config, model, device, train_loader, optimizer, scheduler, criterion):
    model.train()
    correct_classifications = 0
    train_multi_loss = 0
    train_class_loss = 0
    train_point_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_batch = prepare_batch(batch, device)
        optimizer.zero_grad()
        output = model(input_batch)
        combined_loss, class_loss, point_loss = criterion(output)
        combined_loss.backward()
        optimizer.step()
        scheduler.step()

        train_multi_loss += combined_loss.item()  # sum combined loss
        train_class_loss += class_loss.item()  # sum classification loss
        train_point_loss += point_loss.item()  # sum regression loss

        # get indices for max log-probability predictions
        pred_class = output['predicted_class'].argmax(dim=1, keepdim=True)
        # real/ground truth classes
        real_class = output['real_class'].view_as(pred_class)
        # correctly classified outputs
        correct_classifications += pred_class.eq(real_class).sum().item()

    # we get total number of samples from sampler, because of oversampling
    avg_multi_loss = train_multi_loss / len(train_loader.sampler)
    avg_class_loss = train_class_loss / len(train_loader.sampler)
    percent_correct = 100. * correct_classifications / len(train_loader.sampler)
    avg_point_loss = train_point_loss / len(train_loader.sampler)

    # Record learning rate and momentum for WandB
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        current_mom = param_group['betas'][0]

    return {'train_multi_loss': avg_multi_loss,
            'train_class_loss': avg_class_loss,
            'train_keypoint_loss': avg_point_loss,
            'train_accuracy': percent_correct,
            'lr': current_lr,
            'momentum': current_mom
            }


def test(config, model, device, test_loader, criterion, letters):
    model.eval()
    correct_classifications = 0
    test_multi_loss = 0
    test_class_loss = 0
    test_point_loss = 0
    logged_pics = 0
    example_pics = []

    with torch.no_grad():
        for batch in test_loader:
            input_batch = prepare_batch(batch, device)
            output = model(input_batch)
            combined_loss, class_loss, point_loss = criterion(output)
            test_multi_loss += combined_loss.item()  # sum combined loss
            test_class_loss += class_loss.item()  # sum classification loss
            test_point_loss += point_loss.item()  # sum regression loss

            # get indices for max log-probability predictions
            pred_class = output['predicted_class'].argmax(dim=1, keepdim=True)
            # real/ground truth classes
            real_class = output['real_class'].view_as(pred_class)
            # correctly classified outputs
            correct_classifications += pred_class.eq(real_class).sum().item()

            if logged_pics < letters:
                plot = get_predictions_batch(output)
                example_pics.append(wandb.Image(plot))
                matplotlib.pyplot.close('all')
                logged_pics += 1

    # we get total number of samples from sampler, because of oversampling
    avg_multi_loss = test_multi_loss / len(test_loader.sampler)
    avg_class_loss = test_class_loss / len(test_loader.sampler)
    percent_correct = 100. * correct_classifications / len(test_loader.sampler)
    avg_point_loss = test_point_loss / len(test_loader.sampler)

    return {'test_multi_loss': avg_multi_loss,
            'test_class_loss': avg_class_loss,
            'test_keypoint_loss': avg_point_loss,
            'test_accuracy': percent_correct,
            'example_images': example_pics
            }


def prepare_batch(batch, device):
    """Perform any processing of the batch and send to device."""
    batch['image'] = batch['image'].to(device)
    batch['letter_class'] = batch['letter_class'].to(device)
    batch['file_index'] = batch['file_index'].to(device)
    batch['keypoints'] = batch['keypoints'].to(device)
    batch['num_keypoints'] = batch['num_keypoints'].to(device)
    return batch


def get_predictions_batch(sample_batched):
    """Show a sample of model predicted landmarks for a given batch."""
    images_batch = sample_batched['image'].cpu()
    keypoints_pred = sample_batched['predicted_keypoints'].cpu()
    keypoints_batch = keypoints_pred.view(-1,len(sample_batched['keypoints'][1]),2)  # reshape
    predicted_classes = sample_batched['predicted_class'].cpu().max(1, keepdim=True)[1]
    real_classes = sample_batched['real_class'].cpu()

    im_size = images_batch.size(2)
    grid_border_size = 2

    nrow = 8
    plt.figure(figsize=(20, 20))
    grid = utils.make_grid(images_batch[:nrow,:,:], nrow=nrow)
    grid = np.clip(grid.numpy(), 0, 1)  # also clip value ranges for matplotlib
    plt.imshow(grid.transpose((1, 2, 0)))
    title = "Real vs Predicted: "

    for i in range(nrow):
        plt.scatter(keypoints_batch[i, :, 1].numpy() + i * im_size + (i + 1) * grid_border_size,  # x
                    keypoints_batch[i, :, 0].numpy() + grid_border_size,  # y
                    s=30, marker='.', c='r')
        predicted_class = predicted_classes[i]
        title = title + f"**Image {i}: {real_classes[i]} vs {predicted_class.item()}**    "

    plt.title(title)
    plt.axis('off')

    return plt


def save_checkpoint(checkpoint,
                    is_best,
                    checkpoint_dir='saved/'):
    """
    Saves model/optimizer/scheduler state into compressed file.
    Args:
        checkpoint (dict): dict with epoch, model, optimizer, scheduler states
        e.g.:  {'epoch': epoch,
                'loss': current loss,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict()}
        is_best (bool): flag if best model so far
        checkpoint_path: path to save checkpoints
    """
    checkpoint_fname = 'checkpoint.pth'
    best_fname = 'best_model.pth'

    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir+checkpoint_fname)
        if is_best:
            print(f"Saving best model to {checkpoint_dir}")
            torch.save(checkpoint['model_state'], checkpoint_dir+best_fname)
    except Exception:
        print(f"Error saving checkpoint to {checkpoint_dir}...")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load saved checkpoint.
    """
    # load checkpoint dictionary
    checkpoint = torch.load(checkpoint_path)
    # load model state_dict
    model.load_state_dict(checkpoint['model_state'])
    # load optimizer state
    optimizer.load_state_dict(checkpoint['opt_state'])
    # load scheduler state
    scheduler.load_state_dict(checkpoint['sched_state'])
    # load associated loss value from checkpoint
    best_loss = checkpoint['loss']
    # load epoch from checkpoint
    curr_epoch = checkpoint['epoch']
    # return model, optimizer, scheduler, current_epoch, and loss
    return model, optimizer, scheduler, curr_epoch, best_loss


def log_metrics(timestamp, train_start, curr_epoch, total_epoch, train_metrics, test_metrics):
    # Log minutes since training started
    training_duration = (time.time() - train_start)/60
    # Log example images and metrics to WandB.com
    wandb.log({'training_duration': training_duration,
               'epoch': curr_epoch,
               **train_metrics,
               **test_metrics})

    stats = f"epoch: {curr_epoch}, "\
            f"multi loss: {train_metrics['train_multi_loss']:.4f} "\
            f"(test: {test_metrics['test_multi_loss']:.4f}), "\
            f"class accuracy: {train_metrics['train_accuracy']:.2f}% "\
            f"(test: {test_metrics['test_accuracy']:.2f}), "\
            f"class loss: {train_metrics['train_class_loss']:.4f} "\
            f"(test: {test_metrics['test_class_loss']:.4f}), "\
            f"keypoint loss: {train_metrics['train_keypoint_loss']:.4f} "\
            f"(test: {test_metrics['test_keypoint_loss']:.4f}), "\
            f"duration: {training_duration:.2f} min"

    os.makedirs(f'saved/{timestamp}/', exist_ok=True)

    history_fname = f'saved/{timestamp}/history.csv'

    if os.path.exists(history_fname):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    history = open(history_fname, append_write)

    print(stats)
    history.write(stats+'\n')
    history.flush()

    # On last epoch cleanup Wandb and upload best checkpoint
    if curr_epoch == total_epoch-1:
        history.close()
        wandb.save(f'saved/{timestamp}/best_model.pth')
        os.system('wandb gc')


def main():

    # Training settings and hyperparameters
    parser = argparse.ArgumentParser(description='SpenceNet Pytorch Training')
    parser.add_argument('--encoder', default='XResNet34', type=str,
                        choices=['XResNet18', 'XResNet34', 'XResNet50'],
                        help='encoder architecture (default: XResNet34)')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of total training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--use_grayscale', default=True,
                        help='turn input images to grayscale (default: True)')
    parser.add_argument('--img_size', type=int, default=300,
                        help='target image size for training (default: 300)')
    parser.add_argument('--max_lr', type=float, default=0.001,
                        help='maximum learning rate (default: 0.001)')
    parser.add_argument('--encoder_lr_mult', type=float, default=0.25,
                        help='encoder_lr = max_lr * this value (0.25 default)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')
    parser.add_argument('--sched_pct_start', type=float, default=0.3,
                        help='OneCycleLR pct_start parameter (default: 0.3)')
    parser.add_argument('--sched_div_factor', type=float, default=10.0,
                        help='OneCycleLR div factor (default: 10.0)')
    parser.add_argument('--wing_loss_e', type=float, default=2.0,
                        help='Wing Loss e parameter (default: 2.0)')
    parser.add_argument('--wing_loss_w', type=float, default=10.0,
                        help='Wing Loss w parameter (default: 10.0)')
    parser.add_argument('--use_cuda', default=True,
                        help='Enables CUDA training (default: True)')
    parser.add_argument('--seed', type=int, default=None,
                        help='fix random seed for training (default: None)')
    parser.add_argument('--wandb_project', default='multi-head-spencenet',
                        type=str, help='WandB project name')
    parser.add_argument('--save_dir', default='saved/', type=str,
                        help='directory to save outputs in (default: saved/)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to checkpoint to optionally resume from')

    config = parser.parse_args()
    wandb_config = vars(config)  # WandB expects dictionary

    # Get timestamp
    today = datetime.now(tz=utc)
    today = today.astimezone(timezone('US/Pacific'))
    timestamp = today.strftime("%b_%d_%Y_%H_%M")

    wandb.init(config=wandb_config,
               project=config.wandb_project,
               dir=config.save_dir,
               name=timestamp,
               id=timestamp)

    use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': config.num_workers,
              'pin_memory': True} if use_cuda else {}

    # Fix random seeds and deterministic pytorch for reproducibility
    if config.seed:
        torch.manual_seed(config['seed'])  # pytorch random seed
        np.random.seed(config['seed'])  # numpy random seed
        torch.backends.cudnn.deterministic = True

    # DATASET LOADING
    # Letters Dictionary >>> Class ID: [Letter Name, # of Coordinate Values]
    letter_dict = {0: ['alpha', 20], 1: ['beta', 28], 2: ['gamma', 16]}
    letter_ordered_dict = OrderedDict(sorted(letter_dict.items()))

    # Define the tranformations
    train_transforms = transforms.Compose([
                                lt.RandomCrop(10),
                                lt.RandomRotate(10),
                                lt.RandomLightJitter(0.2),
                                lt.RandomPerspective(0.5),
                                lt.Resize(config.img_size),
                                lt.ToNormalizedTensor()
                                ])
    test_transforms = transforms.Compose([
                                    lt.Resize(config.img_size),
                                    lt.ToNormalizedTensor()
                                    ])

    # Add grayscale transform
    if config.use_grayscale:
        train_transforms.transforms.insert(0, lt.ToGrayscale())
        test_transforms.transforms.insert(0, lt.ToGrayscale())

    # Define separate datasets for each annotated class
    letters = [key for key, val in letter_dict.items() if val[1] != 0]
    train_ds_list = []
    test_ds_list = []

    for letter in letters:
        train_ds_list.append(LetterDataset(f'./data/{letter_dict[letter][0]}_small_data.csv',
                                           num_coordinates=letter_dict[letter][1],
                                           transform=train_transforms))
        test_ds_list.append(LetterDataset(f'./data/{letter_dict[letter][0]}_small_data.csv',
                                          is_validation=True,
                                          num_coordinates=letter_dict[letter][1],
                                          transform=test_transforms))

    # Concatenated Datasets
    train_datasets = ConcatDataset(train_ds_list)
    test_datasets = ConcatDataset(test_ds_list)

    # Define Dataloaders with custom LetterBatchSampler
    train_loader = DataLoader(dataset=train_datasets,
                              sampler=LetterBatchSampler(
                                          dataset=train_datasets,
                                          batch_size=config.batch_size,
                                          drop_last=True),
                              batch_size=config.batch_size,
                              **kwargs)

    test_loader = DataLoader(dataset=test_datasets,
                             sampler=LetterBatchSampler(
                                        dataset=test_datasets,
                                        batch_size=config.batch_size,
                                        drop_last=True),
                             batch_size=config.batch_size,
                             **kwargs)

    # INITIALIZE MODEL
    model = SpenceNet(letter_ordered_dict,
                      backbone=config.encoder,
                      c_in=1 if config.use_grayscale else 3,
                      img_size=config.img_size).to(device)

    optimizer = optim.AdamW([
                            {'params': model.encoder.parameters(),
                             'lr': config.max_lr*config.encoder_lr_mult},
                            {'params': model.classification_head.parameters()},
                            {'params': model.keypoint_heads.parameters()}
                            ],
                            lr=config.max_lr, betas=(0.9, 0.99),
                            weight_decay=config.weight_decay)

    # Initialize Loss Function
    criterion = MultiLoss(e=config.wing_loss_e, w=config.wing_loss_w)

    # LR Scheduler
    scheduler = OneCycleLR(optimizer,
                           max_lr=config.max_lr,
                           pct_start=config.sched_pct_start,
                           div_factor=config.sched_div_factor,
                           steps_per_epoch=len(train_loader),
                           epochs=config.epochs)

    # Optionally resume from saved checkpoint
    if config.resume:
        model, optimizer, scheduler, curr_epoch, ckp_loss = load_checkpoint(config.resume, model, optimizer, scheduler)
        start_epoch = curr_epoch
        best_loss = ckp_loss
        print(f'Resuming from checkpoint... Epoch: {start_epoch} Loss: {best_loss:.4f}')
    else:
        start_epoch = 0
        best_loss = math.inf

    # Track all gradients/parameters with WandB
    wandb.watch(model, log='all')

    # Training start time
    training_start = time.time()

    for epoch in range(start_epoch, config.epochs):
        train_metrics = train(config, model, device, train_loader, optimizer, scheduler, criterion)
        test_metrics = test(config, model, device, test_loader, criterion, len(letters))

        # Log training data and metrics
        # TODO: in test, randomly return 4 img per class based on len(letters)
        log_metrics(timestamp,
                    training_start,
                    epoch,
                    config.epochs,
                    train_metrics,
                    test_metrics)

        # Checkpoint saving
        is_best = test_metrics['test_multi_loss'] < best_loss
        best_loss = min(test_metrics['test_multi_loss'], best_loss)
        save_checkpoint({'epoch': epoch,
                         'loss': test_metrics['test_multi_loss'],
                         'model_state': model.state_dict(),
                         'opt_state': optimizer.state_dict(),
                         'sched_state': scheduler.state_dict()},
                        is_best,
                        checkpoint_dir=f'saved/{timestamp}/')


if __name__ == '__main__':
    main()
