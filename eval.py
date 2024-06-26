import torch
import torch.nn as nn
import cv2
import FRbackbone
import yaml
import argparse
from torchvision import transforms

cfg = yaml.safe_load(open("cfg.yml", "r"))
args = argparse.Namespace()
for key in cfg:
    setattr(args, key, cfg[key])

model = FRbackbone.get_model(**args.model).to(args.device)
model.load_state_dict(torch.load(args.load_model))
model.eval()

img1 = cv2.imread(r"D:\repos\Face_Recognition\images\img1.jpg")
img2 = cv2.imread(r"D:\repos\Face_Recognition\images\img_0.jpg")
img1 = transforms.ToTensor()(cv2.resize(img1, (112, 112)))
img2 = transforms.ToTensor()(cv2.resize(img2, (112, 112)))
out1 = model(torch.tensor(img1).unsqueeze(0).to(args.device))
out2 = model(torch.tensor(img2).unsqueeze(0).to(args.device))
result = nn.CosineSimilarity()(out1, out2)
print(result.item())