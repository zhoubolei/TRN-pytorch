import random
import os
import numpy as np
import torch

from dataset import TSNDataSet
from vidaug import augmentors as va  # pip3 install git+https://github.com/okankop/vidaug --user


class RGB2Gray(object):
    def __call__(self, clip):
        return [x.convert('L').convert('RGB') for x in clip]


def augmentation(prob=0.5, N=2, random_order=True):
    sometimes = lambda aug: va.Sometimes(prob, aug) # Used to apply augmentor with 50% probability
    return va.Sequential([
        va.SomeOf(
        [
            sometimes(va.GaussianBlur(sigma=3.0)),
            sometimes(va.ElasticTransformation(alpha=3.5, sigma=0.25)),
            sometimes(va.PiecewiseAffineTransform(displacement=5, displacement_kernel=1, displacement_magnification=1)),
            sometimes(va.RandomRotate(degrees=10)),
            sometimes(va.RandomResize(0.5)),
            sometimes(va.RandomTranslate(x=20, y=20)),
            sometimes(va.RandomShear(x=0.2, y=0.2)),
            sometimes(va.InvertColor()),
            sometimes(va.Add(100)),
            sometimes(va.Multiply(1.2)),
            sometimes(va.Pepper()),
            sometimes(va.Salt()),
            sometimes(va.HorizontalFlip()),
            sometimes(va.TemporalElasticTransformation()),
            sometimes(RGB2Gray())
        ],
        N=N,
        random_order=random_order
    )]) 


aug = augmentation()


class SiameseDataset(TSNDataSet):
    def __getitem__(self, _):
        path, data, label, _ = self.get()
        should_get_same_class = random.randint(0, 1)
        if bool(should_get_same_class): 
            other_index = random.choice(self.label2videos[label])
            other_path, other_data, other_label, _ = self.get(other_index)
        else:
            # TODO: fix this dirty hack
            other_labels = random.sample(self.label2videos.keys(), 2)
            other_label = next(x for x in other_labels if x != label)
            other_index = random.choice(self.label2videos[other_label])
            other_path, other_data, other_label, _ = self.get(other_index)
        return data, other_data, torch.Tensor([float(label == other_label)])
    
    def get(self, index=None, apply_aug=True):
        if index is None:
            index = random.choice(range(len(self.video_list)))
        record = self.video_list[index]

        if not self.test_mode:
            indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            indices = self._get_test_indices(record)

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        
        if apply_aug:
            images = aug(images)
        process_data = self.transform(images)
        return record.path, process_data, record.label, index
    
    def __len__(self):
        return len(self.video_list) * 4