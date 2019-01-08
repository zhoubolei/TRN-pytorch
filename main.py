import os
import time
import shutil
from typing import Dict, Union
from pathlib import Path

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from ignite.contrib.handlers import ProgressBar
from ignite.engine import _prepare_batch
from ignite.handlers import ModelCheckpoint
from torch.nn.utils import clip_grad_norm_
from ignite.engine import Events, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy, Metric
from torchvideo.datasets import GulpVideoDataset, ImageFolderVideoDataset
from torchvideo.samplers import TemporalSegmentSampler
from torchvideo.transforms import ResizeVideo, CenterCropVideo, CollectFrames, \
    PILVideoToTensor, np, NormalizeVideo, TimeToChannel, NDArrayToPILVideo
from torchvision.transforms import Compose

from label_sets import FileListLabelSet, FileList
from models import TSN
from transforms import *
from opts import parser, log_levels
import logging


def main():
    global args, best_prec1
    args = parser.parse_args()
    logging.basicConfig(level=log_levels[args.verbosity])
    check_rootfolders()

    categories = read_categories(args.categories)
    num_class = len(categories)

    modality = args.modality
    num_segments = args.num_segments
    arch = args.arch
    if args.store_name is None:
        args.store_name = '_'.join(['TRN', args.dataset, modality, arch, args.consensus_type, 'segment%d' % num_segments])
        print('Set store_name: ' + args.store_name)

    model = TSN(num_class, num_segments, modality,
                base_model=arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    train_loader, val_loader = get_dataloaders(model, args.root_path, args.train_list, args.val_list,
                                               filename_prefix=args.image_prefix, batch_size=args.batch_size,
                                               worker_count=args.workers)
    criterion = get_criterion(args.loss_type)
    optimizer = get_optimizer(model, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    device = get_device()
    model = torch.nn.DataParallel(model)
    model.to(device)
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

    def attach_metrics(metrics: Dict[str, Metric], engine: Engine):
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
            engine.add_event_handler(Events.ITERATION_STARTED, metric.started)
            engine.add_event_handler(Events.ITERATION_COMPLETED, metric.completed, name)
        return metrics

    # Metrics must be attached after adjust_lr is attached so lr is available in
    # state.metrics
    train_metrics = attach_metrics(get_metrics(), trainer)
    val_metrics = attach_metrics(get_metrics(), evaluator)

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_lr(engine):
        adjust_learning_rate(optimizer, engine.state.epoch, args.lr_steps)
        engine.state.metrics['lr'] = optimizer.param_groups[-1]['lr']

    # Attach after logging metrics
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_stats(engine):
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % args.print_freq == 0:
            log_metrics(engine, len(train_loader))

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_model(engine):
        if (engine.state.epoch + 1) % args.eval_freq == 0 or \
                engine.state.epoch + 1 == args.epochs:
            evaluator.run(val_loader)


    pbar = ProgressBar()
    pbar.attach(evaluator)
    pbar.attach(trainer)

    checkpoint_handler = ModelCheckpoint(args.root_model, args.store_name, save_interval=1, n_saved=1,
                                         require_empty=False, create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model})

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_val_stats(engine):
        iteration = (engine.state.iteration - 1) % len(val_loader) + 1
        if iteration % args.print_freq == 0:
            log_metrics(engine, len(val_loader))

    # TODO: Figure out how to set initial epoch count for resumed models
    trainer.run(train_loader, max_epochs=args.epochs)


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
    train_filelist = FileList(args.train_list)
    val_filelist = FileList(args.val_list)

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

    is_gulp_dataset = (root_path / 'data_0.gulp').exists()
    sampler = TemporalSegmentSampler(model.num_segments, data_length)
    if is_gulp_dataset:
        train_dataset = GulpVideoDataset(root_path,
                                         filter=train_filter,
                                         label_field="label_numeric",
                                         sampler=sampler,
                                         transform=Compose([NDArrayToPILVideo(),
                                                            train_transform]))
        val_dataset = GulpVideoDataset(root_path,
                                       filter=val_filter,
                                       label_field="label_numeric",
                                       sampler=sampler,
                                       transform=Compose([NDArrayToPILVideo(),
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


def log_metrics(engine, num_batches):
    iteration = (engine.state.iteration - 1) % num_batches + 1
    metrics = engine.state.metrics
    output = ('Epoch: [{epoch}][{iteration}/{num_batches}], lr: {lr:.5f}\t'
              'Time {batch_time:.3f}\t'
              'Data {data_time:.3f}\t'
              'Loss {loss:.4f}\t'
              'Prec@1 {top1:.3f}\t'
              'Prec@5 {top5:.3f}'.format(
            epoch=engine.state.epoch,
            iteration=iteration,
            num_batches=num_batches,
            batch_time=metrics['batch processing time'],
            data_time=metrics['data loading time'],
            loss=metrics['loss'],
            top1=metrics['top-1 accuracy'] * 100,
            top5=metrics['top-5 accuracy'] * 100,
            lr=metrics['lr'])
    )
    # output = ('Epoch: [{epoch}][{iteration}/{num_batches}]\t'
    #           'Time {batch_time:.3f}\t'
    #           'Data {data_time:.3f}\t'
    #           'Loss {loss:.4f}\t'
    #           'Prec@1 {top1:.3f}\t'
    #           'Prec@5 {top5:.3f}'.format(
    #         epoch=engine.state.epoch,
    #         iteration=iteration,
    #         num_batches=num_batches,
    #         batch_time=metrics['batch processing time'],
    #         data_time=metrics['data loading time'],
    #         loss=metrics['loss'],
    #         top1=metrics['top-1 accuracy'],
    #         top5=metrics['top-5 accuracy']
    # ))
    print(output)


def read_categories(categories_path: Path):
    with open(categories_path) as f:
        lines = f.readlines()

    categories = [item.rstrip() for item in lines]
    return categories


def save_checkpoint(state, is_best):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * lr_decay
    weight_decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
