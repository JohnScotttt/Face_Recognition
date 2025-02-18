import sys
import os
import torch
import torch.nn as nn
import cv2
import FRbackbone
import yaml
import argparse
from torchvision import transforms

def predict(cfg_path, img1_path, img2_path):
    cfg_raw = yaml.safe_load(open(cfg_path, "r"))
    cfg = argparse.Namespace()
    for key in cfg_raw:
        setattr(cfg, key, cfg_raw[key])

    if cfg.load_model == "baseline.pth" and not os.path.exists("baseline.pth"):
        import requests
        from tqdm import tqdm

        url = "https://github.com/JohnScotttt/Face_Recognition/releases/download/v2.0/baseline.pth"
        response = requests.get(url, stream=True)
        with open('baseline.pth', 'wb') as file:
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    bar.update(len(data))

    model = FRbackbone.get_model(**cfg.model).to(cfg.device)
    model.load_state_dict(torch.load(cfg.load_model))
    model.eval()

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = transforms.ToTensor()(cv2.resize(img1, (112, 112)))
    img2 = transforms.ToTensor()(cv2.resize(img2, (112, 112)))
    out1 = model(torch.tensor(img1).unsqueeze(0).to(cfg.device))
    out2 = model(torch.tensor(img2).unsqueeze(0).to(cfg.device))
    result = nn.CosineSimilarity()(out1, out2)
    print(result.item())

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Required parameter missing")
    else:
        predict(sys.argv[1], sys.argv[2], sys.argv[3])