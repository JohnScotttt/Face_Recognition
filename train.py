import argparse
import datetime
import logging
import math
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
from FRdata import get_dataloader, get_dataset

config_path = "config/cwb_10e_11N_nbs63_ir50_f512_ArcCE_s64_m5e-1_gpu.yml"

cfg = yaml.safe_load(open(config_path, "r"))
args = argparse.Namespace()
for key in cfg:
    setattr(args, key, cfg[key])

if args.device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

augmentation = []
for ways in args.augmentation:
    augmentation.append(eval(ways))
augmentation = transforms.Compose(augmentation)

FRDS = get_dataset(**args.dataloader, transforms=augmentation)
FRDL = get_dataloader(**args.dataloader, dataset=FRDS, shuffle=True)

model = FRbackbone.get_model(**args.model).to(device)
if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))
loss_fn = FRloss.get_loss_fn(**args.loss_fn).to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.scheduler)

if args.log:
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H%M%S")
    log_file_name = os.path.join(args.log_path, f"train{formatted_time}.log")
    logging.basicConfig(filename=log_file_name, filemode="w",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

info_list = ["Start training\n",]
with open(config_path, "r") as f:
    info_list.append(f.read())
logging.info("".join(info_list))
logging.info(f"Real device: {device}")

for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    bar = tqdm(FRDL)
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
        out_batch_tem = out_batch
        label = torch.zeros(1, dtype=torch.long).cuda()
        loss = loss_fn(out_batch_tem, label)
        if math.isnan(loss.item()):
            logging.error(f"Epoch {epoch}/{args.epochs} Batch {counter} Loss is nan")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.log and counter % args.log_interval == 0:
            logging.info(f"Epoch {epoch}/{args.epochs} Batch {counter} Loss {loss.item()}")
        bar.set_description(f"Epoch {epoch}/{args.epochs} Loss {loss.item():.4f}")
    scheduler.step()
    torch.save(model.state_dict(), os.path.join(args.save_path, f"{formatted_time}epoch_{epoch}.pth"))
