import os
import shutil
from tqdm import tqdm

data_root_path = "D:/data/face/archive/casia-webface"
f = open("casia-webface.txt", "w")
labels = os.listdir(data_root_path)
for label in tqdm(labels):
    img_path = os.path.join(data_root_path, label)
    imgs = os.listdir(img_path)
    for img in tqdm(imgs, leave=False):
        f.write(f"{int(label)}\t{os.path.join('casia-webface', label, img)}\n")
f.close()