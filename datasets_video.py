import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET = 'datasets'


class ModalityError(Exception):
    def __init__(self, modality):
        super(ModalityError, self).__init__("Unknown modality '{}'".format(modality))


def return_something(modality):
    filename_categories = 'something/category.txt'
    prefix = '{:05d}.jpg'
    filename_imglist_train = 'something/train_videofolder.txt'
    filename_imglist_val = 'something/val_videofolder.txt'
    if modality == 'RGB':
        root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/20bn-something-something-v1'
    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/flow'
    else:
        raise ModalityError(modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    prefix = '{:06d}.jpg'
    filename_imglist_train = 'something/v2/train_videofolder.txt'
    filename_imglist_val = 'something/v2/val_videofolder.txt'
    if modality == 'RGB':
        root_data = '/mnt/localssd2/aandonia/something/v2/20bn-something-something-v2-frames'
    elif modality == 'Flow':
        root_data = '/mnt/localssd2/aandonia/something/v2/flow'
    else:
        raise ModalityError(modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    dataset_root = 'jester/'
    prefix = '{:05d}.jpg'
    filename_categories = os.path.join(dataset_root, 'jester-v1-numeric-labels.txt')
    filename_imglist_train = os.path.join(dataset_root, 'jester-v1-train-filelist.txt')
    filename_imglist_val = os.path.join(dataset_root, 'jester-v1-validation-filelist.txt')

    if modality == 'RGB':
        root_data = os.path.join(dataset_root, '20bn-jester-v1')
    elif modality == 'Flow':
        root_data = os.path.join(dataset_root, '20bn-jester-v1-flow')
    else:
        raise ModalityError(modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_charades(modality):
    filename_categories = 'charades/category.txt'
    filename_imglist_train = 'charades/train_segments.txt'
    filename_imglist_val = 'charades/test_segments.txt'
    prefix = '{:06d}.jpg'
    root_data = '/data/vision/oliva/scratch/bzhou/charades/Charades_v1_rgb'
    if modality != 'RGB':
        raise ModalityError(modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_moments(modality):
    filename_categories = '/data/vision/oliva/scratch/moments/split/categoryList_nov17.csv'
    if modality == 'RGB':
        prefix = '{:06d}.jpg'
        root_data = '/data/vision/oliva/scratch/moments/moments_nov17_frames'
        filename_imglist_train = '/data/vision/oliva/scratch/moments/split/rgb_trainingSet_nov17.csv'
        filename_imglist_val = '/data/vision/oliva/scratch/moments/split/rgb_validationSet_nov17.csv'

    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/moments/moments_nov17_flow'
        prefix = 'flow_xyz_{:05d}.jpg'
        filename_imglist_train = '/data/vision/oliva/scratch/moments/split/flow_trainingSet_nov17.csv'
        filename_imglist_val = '/data/vision/oliva/scratch/moments/split/flow_validationSet_nov17.csv'
    else:
        raise ModalityError(modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(args):
    dataset = args.dataset
    modality = args.modality
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'charades': return_charades, 'moments': return_moments}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()

    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix
