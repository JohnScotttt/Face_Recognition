import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class NegativeDataset(Dataset):
    def __init__(self, root_path, description, transform=None):
        self.transform = transform
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
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.labels)

class PositiveDataset(Dataset):
    def __init__(self, root_path, description, transform=None):
        self.transform = transform
        self.images_path = []
        with open(description) as f:
            lines = f.readlines()
        tag = None
        for l in lines:
            words = l.rstrip('\n').split('\t')
            try:
                if int(words[0]) != tag:
                    tag = int(words[0])
                    self.images_path.append([])
                self.images_path[tag].append(os.path.join(root_path, words[1]))
            except:
                pass
        self.max_label = len(self.images_path) - 1

    def __getitem__(self, idx):
        size = len(self.images_path[idx])
        idx1 = np.random.randint(size)
        idx2 = np.random.randint(size)
        while idx2 == idx1:
            idx2 = np.random.randint(size)
        img1 = cv2.imread(self.images_path[idx][idx1])
        img2 = cv2.imread(self.images_path[idx][idx2])
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(idx)

    def __len__(self):
        return len(self.images_path)

class FaceRecognitionDataloader(DataLoader):
    def __init__(self, pos_dataset, neg_dataset, neg_batch_size, shuffle):
        super(FaceRecognitionDataloader, self).__init__(
            pos_dataset, batch_size=1, shuffle=shuffle)
        self.neg_batch_size = neg_batch_size
        self.neg_dataset = DataLoader(neg_dataset, batch_size=neg_batch_size, shuffle=shuffle)

    def __iter__(self):
        base_batch = super(FaceRecognitionDataloader, self).__iter__()
        for query, pos, pos_label in base_batch:
            neg_batch, neg_labels = next(iter(self.neg_dataset))
            batch = torch.cat((query, pos, neg_batch), dim=0)
            labels = torch.cat((pos_label, pos_label, neg_labels), dim=0)
            yield batch, labels
        

if __name__ == "__main__":
    cfg = yaml.safe_load(open("cfg.yml", "r"))
    args = argparse.Namespace()
    for key in cfg:
        setattr(args, key, cfg[key])
    
    augmentation = []
    for ways in args.augmentation:
        augmentation.append(eval(ways))
    PD = PositiveDataset(args.root_path, args.description, transforms.Compose(augmentation))
    ND = NegativeDataset(args.root_path, args.description, transforms.Compose(augmentation))
    FRD = FaceRecognitionDataloader(PD, ND, 64, True)
    for batch, labels in FRD:
        print(batch.shape, labels.shape)
        break