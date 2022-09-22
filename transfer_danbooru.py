import glob, os, torch
from turtle import color 
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
from sklearn.model_selection import train_test_split 
import numpy as np 
import matplotlib.pyplot as plt 
from typing import List 
from random import shuffle
import shutil 

color_mapping = {
    0:np.array([0, 0, 0]),
    1:np.array([0, 0, 128]),
    2:np.array([0, 128, 0]),
    3:np.array([0, 128, 128]),
    4:np.array([128, 0, 0]),
    5:np.array([128, 0, 128]),
    6:np.array([128, 128, 0])       
}

def label2SegmentationMap(label, color_mapping) -> np.array:
    smap = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(len(color_mapping)):
        smap[(label==i)] = color_mapping[i]
    return smap 

def save_image_grey(image: np.array, filename: str) -> None: 
    image = Image.fromarray(image.astype("uint8"), "L")
    image.save(filename)
    print("[INFO] save ", filename)
    
def save_image_rgb(image: np.array, filename: str) -> None: 
    image = Image.fromarray(image.astype("uint8"), "RGB")
    image.save(filename)
    print("[INFO] save ", filename)


if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    res = glob.glob("./datasets/anime/*/*.jpg")
    shuffle(res)
    train_res = res[5000:]
    valid_res = res[0:5000]
    if not os.path.exists("./datasets/train"):
        os.mkdir("./datasets/train")
    if not os.path.exists("./datasets/train_label"):
        os.mkdir("./datasets/train_label")
    if not os.path.exists("./datasets/valid"):
        os.mkdir("./datasets/valid")
    if not os.path.exists("./datasets/valid_label"):
        os.mkdir("./datasets/valid_label") 
    """
    # LOAD UNET MODEL
    """
    class_num = 7
    model = smp.Unet('timm-mobilenetv3_large_100', encoder_weights='imagenet', classes=class_num, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to("cuda")
    model.load_state_dict(torch.load("./weights/MobileNetV2-Anime-Unet.pt"))
    for index in range(len(train_res)):
        fname = res[index].split("\\")[2][:-4]
        img = Image.open(res[index]).convert("RGB")
        img2 = img 
        img = transform1(img).to("cuda")
        img = img.reshape((1, 3, 512, 512))
        out = model(img)[0]
        out = torch.argmax(out, dim=0)
        smap = label2SegmentationMap(out.cpu().detach().numpy(), color_mapping)
        out = out.cpu().detach().numpy().astype("uint8")
        save_image_grey(out, "./datasets/train_label/"+fname+".png")
        save_image_rgb(np.array(img2), "./datasets/train/"+fname+".jpg")


    
    for index in range(len(valid_res)):
        fname = res[index].split("\\")[2][:-4]
        img = Image.open(res[index]).convert("RGB")
        img2 = img 
        img = transform1(img).to("cuda")
        img = img.reshape((1, 3, 512, 512))
        out = model(img)[0]
        out = torch.argmax(out, dim=0)
        smap = label2SegmentationMap(out.cpu().detach().numpy(), color_mapping)
        out = out.cpu().detach().numpy().astype("uint8")
        save_image_grey(out, "./datasets/valid_label/"+fname+".png")
        save_image_rgb(np.array(img2), "./datasets/valid/"+fname+".jpg")
