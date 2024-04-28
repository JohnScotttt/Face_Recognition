import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NegativeDataset(Dataset):
    def __init__(self, root_path, description, transforms=None):
        self.transforms = transforms
        self.images_path = []
        self.labels = []
        self.max_label = 0
        with open(description) as f:
            lines = f.readlines()
        for l in lines:
            words = l.rstrip('\n').split('\t')
            try:
                self.images_path.append(os.path.join(root_path, words[1]))
                self.labels.append(int(words[0]))
                if int(words[0]) > self.max_label:
                    self.max_label = int(words[0])
            except:
                pass

    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx])
        if self.transforms:
            img = self.transforms(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.labels)

class PositiveDataset(Dataset):
    def __init__(self, root_path, description, transforms=None):
        self.transforms = transforms
        self.images_path = {}
        with open(description) as f:
            lines = f.readlines()
        for l in lines:
            words = l.rstrip('\n').split('\t')
            try:
                if self.images_path.get(int(words[0])) is None:
                    self.images_path[int(words[0])] = []
                self.images_path[int(words[0])].append(os.path.join(root_path, words[1]))
            except:
                pass

    def __getitem__(self, idx):
        size = len(self.images_path[idx])
        idx1 = np.random.randint(size)
        idx2 = np.random.randint(size)
        while idx2 == idx1:
            idx2 = np.random.randint(size)
        img1 = cv2.imread(self.images_path[idx][idx1])
        img2 = cv2.imread(self.images_path[idx][idx2])
        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        return img1, img2, torch.tensor(idx)

    def __len__(self):
        return len(self.images_path)
    
class FRDataset:
    def __init__(self, root_path, description, transforms=None, **kwargs):
        self.pos_dataset = PositiveDataset(root_path, description, transforms)
        self.neg_dataset = NegativeDataset(root_path, description, transforms)

class FRDataloader(DataLoader):
    def __init__(self, dataset, neg_batch_size, shuffle, **kwargs):
        super(FRDataloader, self).__init__(dataset.pos_dataset, batch_size=1, shuffle=shuffle)
        self.neg_dataset = DataLoader(dataset.neg_dataset, batch_size=neg_batch_size, shuffle=shuffle)

    def __iter__(self):
        base_batch = super(FRDataloader, self).__iter__()
        for query, pos, pos_label in base_batch:
            neg_batch, neg_labels = next(iter(self.neg_dataset))
            batch = torch.cat((query, pos, neg_batch), dim=0)
            labels = torch.cat((pos_label, pos_label, neg_labels), dim=0)
            yield batch, labels