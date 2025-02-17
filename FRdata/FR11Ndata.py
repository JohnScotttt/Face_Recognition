import os
import itertools
import cv2
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.xpu import device
from tqdm import  tqdm
class FRdataset(Dataset):
    def __init__(self, description, neg_batch_size, transforms=None, **kwargs):
        self.transforms = transforms
        self.neg_batch_size = neg_batch_size
        images_path_dict = {}
        with open(description) as f:
            lines = f.readlines()
        for l in lines:
            words = l.rstrip('\n').split('\t')
            if images_path_dict.get(int(words[0])) is None:
                images_path_dict[int(words[0])] = []
            images_path_dict[int(words[0])].append(words[1])
        self.images_path_labels_list = list(images_path_dict.values())

    def __getitem__(self, idx):
        whole_pos_list = self.images_path_labels_list[idx]
        pos_pairs = random.sample(whole_pos_list, 2)
        whole_neg_list = list(itertools.chain.from_iterable(
            self.images_path_labels_list[:idx] + self.images_path_labels_list[idx+1:]))
        neg_path_list = random.sample(whole_neg_list, self.neg_batch_size)
        group = pos_pairs + neg_path_list
        imgs = self.read_list_img(group)
        if self.transforms:
            imgs = self.transforms_img_list(imgs)
        return imgs

    def __len__(self):
        return len(self.images_path_labels_list)
    
    def read_list_img(self, img_path_list):
        return [cv2.imread(img_path) for img_path in img_path_list]
    
    def transforms_img_list(self, imgs):
        return [self.transforms(img) for img in imgs]