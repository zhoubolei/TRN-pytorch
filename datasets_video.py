import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET= 'video_datasets'

def return_something(modality):
    filename_categories = 'something/category.txt'
    if modality == 'RGB':
        root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/20bn-something-something-v1'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        #root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/flow'
        root_data = '/mnt/localssd1/bzhou/something/flow'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/data/vision/oliva/scratch/bzhou/video/jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_charades(modality):
    filename_categories = 'charades/category.txt'
    filename_imglist_train = 'charades/train_segments.txt'
    filename_imglist_val = 'charades/test_segments.txt'
    if modality == 'RGB':
        prefix = '{:06d}.jpg'
        root_data = '/data/vision/oliva/scratch/bzhou/charades/Charades_v1_rgb'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_moments(modality):
    filename_categories = '/home/shared/Moments_in_Time_Mini_Stream/moments_categories.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = '/home/shared/Moments_in_Time_Mini_Stream/' #/data/vision/oliva/scratch/moments/moments_nov17_frames'
        filename_imglist_train = '/home/shared/Moments_in_Time_Mini_Stream/training_frames_nano.txt' 
#'/home/shared/Moments_in_Time_Mini_Stream/training_frames.txt' #/data/vision/oliva/scratch/moments/split/rgb_trainingSet_nov17.csv'
        filename_imglist_val = '/home/shared/Moments_in_Time_Mini_Stream/validation_frames_nano.txt'
# '/home/shared/Moments_in_Time_Mini_Stream/validation_frames.txt' #/data/vision/oliva/scratch/moments/split/rgb_validationSet_nov17.csv'

    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/moments/moments_nov17_flow'
        prefix = 'flow_xyz_{:05d}.jpg'
        filename_imglist_train = '/data/vision/oliva/scratch/moments/split/flow_trainingSet_nov17.csv'
        filename_imglist_val = '/data/vision/oliva/scratch/moments/split/flow_validationSet_nov17.csv'
    else:
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    dict_single = {'jester':return_jester, 'something':return_something, 'charades': return_charades, 'moments': return_moments}
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

