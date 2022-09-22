import numpy as np
import cv2, glob 
from typing import List 
import matplotlib.pyplot as plt 
import torch.nn as nn 
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import albumentations as A
import torch, random 

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

def showImageWithMask(image: np.array, mask: np.array) -> None: 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.imshow(mask)
    plt.show()

def save_augmented(image: np.array, mask: np.array, filename: str, s: int) -> None: 
    aug_image = Image.fromarray(image.astype("uint8"), "RGB")
    aug_image.save("./aug_image/"+filename+"_seed_"+str(s)+".jpg")
    aug_mask = Image.fromarray(mask.astype("uint8"), "RGB")
    aug_mask.save("./aug_mask/"+filename+"_seed_"+str(s)+".png")
    print("[INFO] save augmented seed: ", s)

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, s, transform=None) -> None:
        self.img_dir = img_dir 
        self.mask_dir = mask_dir
        self.filenames = [p.split("\\")[1][:-4] for p in glob.glob(self.mask_dir + "*.png")]
        self.transforms = transform 
        self.class_map =  [
            np.array([0, 0, 0]), # 0 background
            np.array([0, 0, 128]), # 1 face
            np.array([0, 128, 0]), # 2 hair
            np.array([0, 128, 128]), # 3 cloth
            np.array([128, 0, 0]), # 4 eye 
            np.array([128, 0, 128]), # 5mouth
            np.array([128, 128, 0])    # 6 skin
        ]
        self.s = s
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img = np.array(Image.open(self.img_dir + self.filenames[index]+".jpg").convert("RGB"))
        mask = np.array(Image.open(self.mask_dir + self.filenames[index]+".png").convert("RGB"))
        if self.transforms is not None:  
            random.seed(self.s)
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"] # mask 3 channel

        # save_augmented(img, mask, self.filenames[index], self.s)
        # showImageWithMask(img, mask)
        mask = torch.from_numpy(segmentationMap2Label(mask, self.class_map)).long()
        t = T.Compose([T.ToTensor()])
        img = t(img)
        return img, mask # 3x512x512, 512x512

if __name__ == "__main__":
    seed = [(i+1)*5 for i in range(15)]
    augmented_transform = A.Compose([
        #A.Resize(512, 512, interpolation=cv2.INTER_NEAREST), 
        A.HorizontalFlip(), 
        A.VerticalFlip(), 
        A.GridDistortion(p=0.5),
        A.RandomBrightnessContrast((0,0.2),(0,0.2)),
        A.GaussNoise(),
        A.Rotate(limit=(-30, 30), p=0.5)
        ])
    train_transform = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST), 
        ]) 
    for s in seed: 
        print("[INFO] seed:", s)
        d = SegmentationDataset("./data/train/", "./data/train_label/", s, augmented_transform)
        for i in range(len(d)):
            img, mask = d[i]
            print(img.shape, mask.shape)
            plt.imshow(mask)
            plt.show()
            break 
        break 


