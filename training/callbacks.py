import os
import sys
from os.path import join
from tqdm import tqdm
from collections import defaultdict, OrderedDict


class Callback:
    """
    Base class for all training loop callbacks.
    The callback class has a set of methods invoked during training/val loops.
    Callbacks can adjust the model's properties, save state, log output, and
    other processes at prespecified portions of the training loop.
    """
    def training_start(self, **kwargs):
        pass

    def training_end(self, **kwargs):
        pass

    def epoch_start(self, **kwargs):
        pass

    def epoch_end(self, **kwargs):
        pass

    def batch_start(self, **kwargs):
        pass

    def batch_end(self, **kwargs):
        pass


class CallbackGroup(Callback):
    """
    Wrapper class for a collection of callbacks which delegates
    specified methods calls to each contained callback.
    """
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = callbacks
        self._callbacks = {type(cb).__name__: cb for cb in self.callbacks}

    def training_start(self, **kwargs):
        for cb in self.callbacks: cb.training_start(**kwargs)

    def training_end(self, **kwargs):
        for cb in self.callbacks: cb.training_end(**kwargs)

    def epoch_start(self, **kwargs):
        for cb in self.callbacks: cb.epoch_start(**kwargs)

    def epoch_end(self, **kwargs):
        for cb in self.callbacks: cb.epoch_end(**kwargs)

    def batch_start(self, **kwargs):
        for cb in self.callbacks: cb.batch_start(**kwargs)

    def batch_end(self, **kwargs):
        for cb in self.callbacks: cb.batch_start(**kwargs)

    def set_loop(self, loop):
        for cb in self.callbacks: cb.loop = loop

    def __getitem__(self, item):
        if item not in self._callbacks:
            raise KeyError(f'unknown callback: {item}')
        return self._callbacks[item]


class Logger(Callback):
    """
    Writes performance metrics during the training process into a list
    of streams.
    Parameters:
        streams: A list of file-like objects with 'write()' method.
    """
    def __init__(self, streams=None, log_every=1):
        self.streams = streams or [sys.stdout]
        self.log_every = log_every
        self.epoch_history = {}
        self.curr_epoch = 0

    def epoch_end(self, epoch=None, metrics=None, **kwargs):
        stats = [f'{name}: {value:2.4f}' for name, value in metrics.items()]
        metrics = ' - '.join(stats)
        string = f'Epoch {epoch:4d} | {metrics}\n'
        for stream in self.streams:
            stream.write(string)
            stream.flush()


class CSVLogger(Logger):
    """
    A wrapper build on top of bbase Logger callback which opens a CSV file
    to write metrics.
    Parameters:
        filename: A name of CSV file to store metrics history.
    """
    def __init__(self, folder=None, filename='history.csv'):
        super().__init__()
        self.folder = folder
        self.filename = filename
        self.file = None

    def training_start(self, optimizer=None, phases=None, **kwargs):
        self.file = open(join(self.folder, self.filename), 'w')
        self.streams = [self.file]

    def training_end(self, **kwargs):
        if self.file:
            self.file.close()


class History(Callback):

    def __init__(self):
        self.history = []

    def epoch_end(self, epoch=None, metrics=None, **kwargs):
        self.history.append(metrics)

    def training_end(self, **kwargs):
        history = []
        for i, record in enumerate(self.history):
            item = record.copy()
            item['epoch'] = i
            history.append(item)
        self.history = history


class ImprovementTracker(Callback):
    """
    Tracks a specified metric during training and reports when the
    metric does not improve after the predefined number of iterations.

    Arguments:
        patience (int): how many iterations to wait for improvement
        metric (str): name of tracked metric
        better (min/max): if metric decrease/increase is considered improvement
    """

    def __init__(self, patience=1, metric='valid_loss', better=min):
        self.patience = patience
        self.metric = metric
        self.better = better
        self.no_improvement = None
        self.best_value = None
        self.stagnation = None
        self.loop = None

    def training_start(self, optimizer=None, phases=None, **kwargs):
        self.no_improvement = 0
        self.stagnation = False

    def epoch_end(self, epoch=None, metrics=None, **kwargs):
        value = metrics[self.metric]
        best_value = self.best_value or value
        improved = self.better(best_value, value) == value
        if not improved:
            self.no_improvement += 1
        else:
            self.best_value = value
            self.no_improvement = 0
        if self.no_improvement >= self.patience:
            self.stagnation = True

    @property
    def improved(self):
        return self.no_improvement == 0


class EarlyStopping(ImprovementTracker):
    """
    Stops observed training loop if the tracked performance metric does not
    improve during predefined number of iterations.
    """

    def epoch_end(self, epoch=None, metrics=None, **kwargs):
        super().epoch_end(epoch, metrics)
        if self.stagnation:
            self.loop.stop = True


class Checkpoint(ImprovementTracker):
    """
    Saves model attached to the loop each time the tracked performance metric
    is improved, or on each iteration if required.

    Args:
        timestamp (str): prefix for the file names
        folder (str): main folder to save checkpoint to
        backup_path (str): backup checkpoint to another folder
        e.g. useful to copy from local Google Colab folder to external drive
        filename (str): filename to append to timestamp
        save_best_only (bool): only saves best model by default
    """
    def __init__(self, timestamp=None, folder=None, save_best_only=True,
                 filename='_ckp_{metric}_{value:2.0f}.pth',
                 **kwargs):

        super().__init__(**kwargs)
        self.folder = folder or os.getcwd()
        self.save_best_only = save_best_only
        self.filename = filename
        self.best_checkpoint = None
        self.timestamp = timestamp

    @property
    def need_to_save(self):
        if not self.save_best_only:
            return True
        return self.improved

    def get_name(self):
        return self.timestamp + self.filename.format(metric=self.metric, value=self.best_value)

    def epoch_end(self, epoch=None, metrics=None, **kwargs):
        super().epoch_end(epoch, metrics)
        if self.need_to_save:
            try:
                os.remove(self.best_checkpoint)  # delete old best checkpoint
            except Exception:
                pass
            best_path = join(self.folder, self.get_name())
            self.loop.save_checkpoint(epoch, best_path, metrics)
            self.best_checkpoint = best_path


class ParameterUpdater:
    """Apply a specific schedule to the optimizer's parameters.
    The method save_start_values() saves the starting values of optimizer parameters to
    multiply them by scheduling coefficient returned by schedule's update() method.
    The method step() performs an actual update of optimizer properties.
    """
    def __init__(self, schedule, params, opt=None):
        self.schedule = schedule
        self.params = params
        self.opt = opt
        self.start_parameters = None

    def set_optimizer(self, opt):
        self.opt = opt

    def save_start_values(self):
        start = []
        for group in self.opt.param_groups:
            params = {}
            for item in self.params:
                name = item['name']
                if name in group:
                    params[name] = group[name]
            start.append(params)
        self.start_parameters = start

    def current_values(self):
        return [
            {conf['name']: group[conf['name']]
             for conf in self.params}
            for group in self.opt.param_groups]

    def step(self):
        mult = self.schedule.update()
        momentum = self.schedule.momentum
        for i, group in enumerate(self.opt.param_groups):
            for item in self.params:
                name = item['name']
                if name in group:
                    params = self.start_parameters[i]
                    inverse = item.get('inverse', False)
                    start_value = params.get(name)
                    if name == 'betas':  # fix for AdamW momentum/first beta
                        self.opt.param_groups[i]['betas'] = (momentum, 0.99)
                    else:
                        self.opt.param_groups[i][name] = start_value * ((1 - mult) if inverse else mult)


class Scheduler(Callback):
    """Takes in a scheduling function to get a multiplier (0-1) and applies it to the
    oprimizer's parameters using the ParameterUpdater class (except during validation phase).
    Arguments:
        schedule: scheduling function
        mode: 'epoch'/'batch' update learning rate/parameters every epoch/batch
    """
    default = [{'name': 'lr'}]

    def __init__(self, schedule, mode='epoch', params_conf=None):
        self.schedule = schedule
        self.params_conf = params_conf or self.default
        self.mode = mode
        self.history = []

    def training_start(self, optimizer=None, phases=None, **kwargs):
        self.updater = ParameterUpdater(self.schedule, self.params_conf, optimizer)
        self.updater.save_start_values()

    def batch_end(self, phase=None, **kwargs):
        if self.mode == 'batch' and phase.name == 'train':
            self.update_parameters()

    def epoch_start(self, epoch=None, phase=None, **kwargs):
        if self.mode == 'epoch' and phase.name == 'train':
            self.update_parameters()

    def update_parameters(self):
        self.updater.step()
        self.history.append(self.updater.current_values())

    def parameter_history(self, name, *names, group_index=0):
        if not self.history:
            return {}
        curve = defaultdict(list)
        names = [name] + list(names)
        for record in self.history:
            group = record[group_index]
            for name in names:
                if name not in group:
                    raise ValueError(f'no history for parameter \'{name}\'')
                curve[name].append(group[name])
        return dict(curve)


class ProgressBar(Callback):

    def training_start(self, optimizer=None, phases=None, **kwargs):
        bars = OrderedDict()
        for phase in phases:
            bars[phase.name] = tqdm(total=len(phase.dataloader), desc=phase.name)
        self.bars = bars

    def batch_end(self, phase=None, **kwargs):
        bar = self.bars[phase.name]
        bar.set_postfix_str(f"loss: {phase.rolling_metrics['loss']:.4f}")
        bar.update()
        bar.refresh()

    def epoch_end(self, **kwargs):
        for bar in self.bars.values():
            bar.n = 0
            bar.refresh()

    def training_end(self, **kwargs):
        for bar in self.bars.values():
            bar.n = bar.total
            bar.refresh()
            bar.close()
