import numbers
import os
import sys
import time
import shutil
import warnings
from typing import Dict, Union, List
from pathlib import Path
from tensorboardX import SummaryWriter

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from ignite.contrib.handlers import ProgressBar
from ignite.engine import _prepare_batch
from ignite.handlers import ModelCheckpoint
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from ignite.engine import Events, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy, Metric
from torchvideo.datasets import GulpVideoDataset, ImageFolderVideoDataset
from torchvideo.samplers import TemporalSegmentSampler
from torchvideo.transforms import ResizeVideo, CenterCropVideo, CollectFrames, \
    PILVideoToTensor, np, NormalizeVideo, TimeToChannel, TimeApply
from torchvision.transforms import Compose, ToPILImage

from label_sets import FileListLabelSet, FileList
from models import TSN
from transforms import *
from opts import parser, log_levels
import logging

cudnn.benchmark = True


class GlobalStep:
    def __init__(self):
        self._step = 0

    def increment(self, *args, **kwargs):
        self._step += 1

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step


def filename_format_value(value, precision=2):
    if isinstance(value, numbers.Number):
        return ("{:." + str(precision) + "e}").format(value)
    return "{}".format(value)


def get_model_filename(model: Union[TSN, DataParallel],
                       optimizer: torch.optim.Optimizer,
                       dataset: str):
    if isinstance(model, DataParallel):
        model = model.module
    optimizer_str = "optim=" + optimizer.__class__.__name__.lower()
    optimizer_str += "_".join([f"{name}={filename_format_value(value)}"
                               for name, value in optimizer.defaults.items()])
    return (f"{model.arch}" +
            f"_dataset={dataset}" +
            f"_segments={model.num_segments}" +
            f"_consensus={model.consensus_type}" +
            "_" + optimizer_str)


def get_summary_writer(model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       log_dir=Path('.')):
    writer = SummaryWriter(log_dir=str(log_dir))
    x, _ = next(iter(data_loader))
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def add_progress_bar(engine, metrics: List[str] = None):
    progress_bar = ProgressBar()
    progress_bar.attach(engine, metric_names=metrics)


def get_log_dir(experiment_dir: Path, model_name: str):
    return experiment_dir / model_name


def main():
    global args
    args = parser.parse_args()
    print("Argument settings")
    print("=================")
    parser.print_values()
    print("=================")
    logging.basicConfig(level=log_levels[args.verbosity])

    categories = read_categories(args.categories)
    num_class = len(categories)

    modality = args.modality
    segment_count = args.segment_count
    arch = args.arch

    model = TSN(num_class, segment_count, modality,
                base_model=arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn)

    train_loader, val_loader = get_dataloaders(model, args.root_path, args.train_list, args.val_list,
                                               filename_prefix=args.image_prefix, batch_size=args.batch_size,
                                               worker_count=args.workers)

    criterion = get_criterion(args.loss_type)
    optimizer = get_optimizer(model, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model_filename = get_model_filename(model, optimizer, args.dataset)
    log_dir = get_log_dir(args.experiment_dir, model_filename)
    summary_writer = get_summary_writer(model, train_loader, log_dir=log_dir)

    device = get_device()
    model = torch.nn.DataParallel(model)
    model.to(device)

    if args.resume is not None:
        epoch = resume_model(args, model, optimizer)
    else:
        epoch = 0

    print("Train dataset size: {}".format(len(train_loader) * train_loader.batch_size))
    print("Val dataset size: {}".format(len(val_loader) * val_loader.batch_size))

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        return y_pred, y

    trainer = Engine(train_step)
    # It's important that the EngineTimer be attached first, otherwise
    # invalid timings will be obtained due to other handlers excuting prior
    # to this one.
    EngineTimer(trainer)

    evaluator = create_supervised_evaluator(model, device=device)
    EngineTimer(evaluator)

    def get_metrics():
        return {
            'top-1 accuracy': Accuracy(),
            'top-5 accuracy': TopKCategoricalAccuracy(k=5),
            'loss': Loss(criterion)
        }

    def attach_metrics(metrics: Dict[str, Metric], engine: Engine, val=False):
        for name, metric in metrics.items():
            metric.attach(engine, name)

            # By default metrics only attach their started/completed method to the
            # EPOCH_STARTED, EPOCH_COMPLETED event. The completed method is what assigns
            # the resulting metric to engine.state.metrics so we also want to add it
            # after an iteration completion so that we can log running-metrics We also
            # have to reset the metric which is what the metric.started method does, so
            # we fire that on ITERATION_START.
            # TODO: Look at using RunningAverage to replace this:
            #   https://pytorch.org/ignite/metrics.html#ignite.metrics.RunningAverage
            if not val:
                engine.add_event_handler(Events.ITERATION_STARTED, metric.started)
                engine.add_event_handler(Events.ITERATION_COMPLETED, metric.completed, name)
        return metrics

    # Metrics must be attached after adjust_lr is attached so lr is available in
    # state.metrics
    attach_metrics(get_metrics(), trainer)
    attach_metrics(get_metrics(), evaluator, val=True)

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_lr(engine):
        adjust_learning_rate(optimizer, engine.state.epoch, args.lr_steps)
        engine.state.metrics['lr'] = optimizer.param_groups[-1]['lr']

    global_step = GlobalStep()

    trainer.add_event_handler(Events.ITERATION_COMPLETED, global_step.increment)

    # Attach after logging metrics
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_stats(engine):
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % args.print_freq == 0:
            log_metrics(engine, len(train_loader))
        if iteration % args.tensboard_log_freq == 0:
            log_tensorboard_metrics(summary_writer, engine, global_step)
        sys.stdout.flush()

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_model(engine):
        if (engine.state.epoch + 1) % args.eval_freq == 0 or \
                engine.state.epoch + 1 == args.epochs:
            evaluator.run(val_loader)

    if args.progress_bar:
        metric_names = ['top-1 accuracy', 'top-5 accuracy']
        add_progress_bar(evaluator, metrics=metric_names)
        add_progress_bar(trainer, metrics=metric_names)
    print(f"=> Model will be saved with prefix '{args.experiment_dir / model_filename}'")
    checkpointer = ModelCheckpoint(args.experiment_dir, model_filename,
                                         save_interval=1, n_saved=1,
                                         require_empty=False, create_dir=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def checkpoint(engine):
        checkpointer(engine, {
            # First dictionary specifies filename_suffix: object
            # we want a single file containing a dictionary of
            # objects
            'state': {
                'epoch': engine.state.epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
        })

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_val_stats(engine):
        log_metrics(engine, len(val_loader), val=True)
        log_tensorboard_metrics(summary_writer, engine, global_step, val=True)
        sys.stdout.flush()

    if args.evaluate:
        evaluator.run(val_loader)
    # TODO: Figure out how to set initial epoch count for resumed models
    trainer.run(train_loader, start_epoch=epoch, max_epochs=args.epochs)


def resume_model(args, model, optimizer):
    checkpoint_path = args.resume
    if checkpoint_path.exists():
        print(("=> loading checkpoint '{}'".format(checkpoint_path)))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print(("=> loaded checkpoint '{}' (epoch {})"
               .format(checkpoint_path, epoch)))
        return epoch
    else:
        print(("=> no checkpoint found at '{}'".format(checkpoint_path)))
        return 0


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_optimizer(model, lr, momentum=0.9, weight_decay=5e-4):
    policies = model.get_optim_policies()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies,
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    return optimizer


def get_criterion(loss_type):
    if loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    return criterion


def get_dataloaders(model, root_path, train_list, val_list,
                    filename_prefix='{:05d}.jpg', batch_size=64, worker_count=0):
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation()

    is_gulp_dataset = (root_path / 'train' / 'data_0.gulp').exists()

    local_dir = args.local_dir
    if args.copy_local:
        if not local_dir.exists() :
            try:
                local_dir.mkdir(parents=True)
            except OSError:
                warnings.warn("Cannot copy dataset to {} as it doesn't exist".format(local_dir))
        else:
            dest_root = local_dir / root_path.name
            if is_gulp_dataset:
                for folder in ('train', 'validation'):
                    src = root_path / folder
                    dest = dest_root / folder
                    if not dest.exists():
                        print("Copying {} to {}".format(src, dest))
                        shutil.copytree(src, dest)
                root_path = dest_root
            else:
                if not dest_root.exists():
                    print("Copying {} to {}".format(root_path, dest_root))
                    shutil.copytree(root_path, dest_root)
                root_path = dest_root

    if model.modality == 'RGB':
        normalize = NormalizeVideo(input_mean * model.num_segments,
                                   input_std * model.num_segments)
        data_length = 1
    elif model.modality == 'Flow':
        normalize = NormalizeVideo(np.mean(input_mean), np.mean(input_std))
        data_length = 5
    elif model.modality == 'RGBDiff':
        normalize = lambda d: d
        data_length = 5
    else:
        raise ValueError("Unknown modality {}".format(model.modality))

    is_inception_model = (model.arch in ['BNInception', 'InceptionV3'])

    if is_inception_model:
        # Inception models are converted from caffe and expect BGR images, not RGB.
        channel_transform = FlipChannels()
    else:
        channel_transform = lambda f: f
    train_transform = Compose([
        train_augmentation,
        CollectFrames(),
        PILVideoToTensor(rescale=not is_inception_model),
        channel_transform,
        TimeToChannel(),
        normalize
    ])
    val_transform = Compose([
        ResizeVideo(int(scale_size)),
        CenterCropVideo(crop_size),
        CollectFrames(),
        PILVideoToTensor(rescale=not is_inception_model),
        channel_transform,
        TimeToChannel(),
        normalize
    ])
    train_filelist = FileList(train_list)
    val_filelist = FileList(val_list)

    def train_filter(vid_path: Union[Path, str]):
        try:
            name = vid_path.name
        except AttributeError:
            name = vid_path
        keep = name in train_filelist
        return keep

    def val_filter(vid_path: Union[Path, str]):
        try:
            name = vid_path.name
        except AttributeError:
            name = vid_path
        keep = name in val_filelist
        return keep

    sampler = TemporalSegmentSampler(model.num_segments, data_length)
    if is_gulp_dataset:
        gulp_label_field = "idx"
        pil_transform = TimeApply(ToPILImage())
        train_dataset = GulpVideoDataset(root_path / 'train',
                                         filter=train_filter,
                                         label_field=gulp_label_field,
                                         sampler=sampler,
                                         transform=Compose([pil_transform,
                                                            train_transform]))
        val_dataset = GulpVideoDataset(root_path / 'validation',
                                       filter=val_filter,
                                       label_field=gulp_label_field,
                                       sampler=sampler,
                                       transform=Compose([pil_transform,
                                                          val_transform]))
    else:
        def train_frame_counter(path):
            return train_filelist[path.name].num_frames

        def val_frame_counter(path):
            return val_filelist[path.name].num_frames

        train_dataset = ImageFolderVideoDataset(root_path,
                                                filename_prefix,
                                                filter=train_filter,
                                                label_set=FileListLabelSet(train_filelist),
                                                sampler=sampler,
                                                transform=train_transform,
                                                frame_counter=train_frame_counter
                                                )
        val_dataset = ImageFolderVideoDataset(root_path,
                                              filename_prefix,
                                              filter=val_filter,
                                              label_set=FileListLabelSet(val_filelist),
                                              sampler=sampler,
                                              transform=val_transform,
                                              frame_counter=val_frame_counter
                                              )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=worker_count, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=worker_count, pin_memory=True)
    return train_loader, val_loader


class EngineTimer:
    def __init__(self, engine):
        self._engine = engine
        self._iteration_start_time = None
        self._iteration_stop_time = None
        self._engine.add_event_handler(Events.ITERATION_STARTED, self.on_iteration_start)
        self._engine.add_event_handler(Events.ITERATION_COMPLETED, self.on_iteration_completion)

    def on_iteration_completion(self, engine):
        self._iteration_stop_time = time.time()
        engine.state.metrics['batch processing time'] = self._iteration_stop_time - self._iteration_start_time

    def on_iteration_start(self, engine):
        self._iteration_start_time = time.time()
        if self._iteration_stop_time is None:
            loading_time = 0
        else:
            loading_time = self._iteration_start_time - self._iteration_stop_time
        engine.state.metrics['data loading time'] = loading_time


def log_tensorboard_metrics(writer: SummaryWriter, engine: Engine,
                            global_step: GlobalStep, val=False):
    metrics = engine.state.metrics

    def add_scalar(name, value):
        writer.add_scalars(name, {
            'val' if val else 'train': value
        }, global_step=global_step.step)

    add_scalar('accuracy/top-1', metrics['top-1 accuracy'] * 100)
    add_scalar('accuracy/top-5', metrics['top-5 accuracy'] * 100)
    add_scalar('loss', metrics['loss'] * 100)
    if not val:
        add_scalar('time/data', metrics['data loading time'])
        add_scalar('time/main', metrics['batch processing time'])
        add_scalar('lr', metrics['lr'])


def log_metrics(engine, num_batches, val=False):
    iteration = (engine.state.iteration - 1) % num_batches + 1
    metrics = engine.state.metrics
    if not val:
        output = ('Epoch: [{epoch}][{iteration}/{num_batches}], lr: {lr:.5f}\t'
                  'Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@5 {top5:.3f}').format(
                epoch=engine.state.epoch,
                iteration=iteration,
                num_batches=num_batches,
                batch_time=metrics['batch processing time'],
                data_time=metrics['data loading time'],
                loss=metrics['loss'],
                top1=metrics['top-1 accuracy'] * 100,
                top5=metrics['top-5 accuracy'] * 100,
                lr=metrics['lr']
        )
    else:
        print("Validation stats:")
        output = ('Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@5 {top5:.3f}').format(
                batch_time=metrics['batch processing time'],
                data_time=metrics['data loading time'],
                loss=metrics['loss'],
                top1=metrics['top-1 accuracy'] * 100,
                top5=metrics['top-5 accuracy'] * 100
        )
    print(output)


def read_categories(categories_path: Path):
    with open(categories_path) as f:
        lines = f.readlines()

    categories = [item.rstrip() for item in lines]
    return categories


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * lr_decay
    weight_decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
