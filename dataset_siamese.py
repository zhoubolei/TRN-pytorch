import random
import os
import numpy as np
import torch

from dataset import TSNDataSet


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
#       label == 1 - match, 0 - no match
        return data, other_data, torch.Tensor([float(label == other_label)])
    
    def get(self, index=None):
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
        
        process_data = self.transform(images)
        return record.path, process_data, record.label, index
    
    def __len__(self):
        return len(self.video_list) * 4