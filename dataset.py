import numpy as np
import cv2, glob 
from typing import List 
import matplotlib.pyplot as plt 
import torch.nn as nn 
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import albumentations as A
import torch 
color_map = [
    np.array([0, 0, 0]),
    np.array([0, 0, 128]),
    np.array([0, 128, 0]),
    np.array([0, 128, 128]),
    np.array([128, 0, 0]),
    np.array([128, 0, 128]),
    np.array([128, 128, 0])    
]
#ans = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
#print(ans)

def segmentationMap2Label(mask_img: np.array, color_map: List) -> np.array:
    mask_img = mask_img.astype("uint8")
    label = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.uint8)
    for i, color in enumerate(color_map):
        label[(mask_img==color).all(axis=2)] = i  
    return label 

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None) -> None:
        self.img_dir = img_dir 
        self.mask_dir = mask_dir
        self.filenames = [p.split("\\")[1][:-4] for p in glob.glob(self.mask_dir + "*.png")]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img = Image.open(self.img_dir + self.filenames[index]+".jpg")
        mask = torch.tensor(np.array(Image.open(self.mask_dir + self.filenames[index]+".png")))
        t = T.Compose([T.ToTensor()])
        img = t(img)
        return img, mask # 3x512x512, 512x512

if __name__ == "__main__":
    d = SegmentationDataset("./data/train/", "./data/train_label/", None)
