import os
import sys
import shutil
from tqdm import tqdm

def discription(data_root_path, save_path):
    f = open(os.path.join(save_path, "description.txt"), "w")
    labels = os.listdir(data_root_path)
    for label in tqdm(labels):
        img_path = os.path.join(data_root_path, label)
        imgs = os.listdir(img_path)
        for img in tqdm(imgs, leave=False):
            f.write(f"{int(label)}\t{os.path.join(data_root_path, label, img)}\n")
    f.close()

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        print("Required parameter missing")
    else:
        data_root_path = args[1]
        save_path = args[2]
        discription(data_root_path, save_path)