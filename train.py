import argparse
import datetime
import logging
import os

import cv2
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

import FRbackbone
import FRloss
from FRdataset import (FaceRecognitionDataloader, NegativeDataset,
                       PositiveDataset)

cfg = yaml.safe_load(open("cfg.yml", "r"))
args = argparse.Namespace()
for key in cfg:
    setattr(args, key, cfg[key])

augmentation = []
for ways in args.augmentation:
    augmentation.append(eval(ways))

if args.device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

PD = PositiveDataset(args.root_path, args.description, transforms.Compose(augmentation))
ND = NegativeDataset(args.root_path, args.description, transforms.Compose(augmentation))
FRD = FaceRecognitionDataloader(PD, ND, args.neg_batch_size, True)

model = FRbackbone.iresnet50().to(device)
if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))
loss_fn = FRloss.FaceRecognitionLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.log:
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H%M%S")
    log_file_name = os.path.join(args.log_path, f"train{formatted_time}.log")
    logging.basicConfig(filename=log_file_name, filemode="w",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)

logging.info("Start training\n"
             f"Dataset: {args.dataset}\n"
             f"Device: {device}\n"
             f"Load Model: {args.load_model}\n"
             f"Epochs: {args.epochs}\n"
             f"Negative Batch Size: {args.neg_batch_size}\n"
             f"Learning Rate: {args.lr}\n"
             f"Weight Decay: {args.weight_decay}\n"
             f"Log Interval: {args.log_interval}\n"
             f"Save Path: {args.save_path}\n"
             f"Augmentation: {args.augmentation}")

for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    bar = tqdm(FRD)
    for counter, (batch, labels) in enumerate(bar):
        flag = False
        batch = batch.to(device)
        labels = labels.to(device)
        for i in range(2, len(labels)):
            if labels[i] == labels[0]:
                flag = True
                break
        if flag:
            continue
        out_batch = model(batch)
        label = torch.zeros(1, dtype=torch.long).cuda()
        loss = loss_fn(out_batch, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.log and counter % args.log_interval == 0:
            logging.info(f"Epoch {epoch}/{args.epochs} Batch {counter} Loss {loss.item()}")
            bar.set_description(f"Epoch {epoch}/{args.epochs} Batch {counter} Loss {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch_{epoch}.pth"))
