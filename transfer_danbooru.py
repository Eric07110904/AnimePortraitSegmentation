import glob, os, torch
from turtle import color 
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
from sklearn.model_selection import train_test_split 
import numpy as np 
import matplotlib.pyplot as plt 
from typing import List 

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

def save_image(image: np.array, filename: str) -> None: 
    image = Image.fromarray(image.astype("uint8"), "RGB")
    image.save(filename)
    print("[INFO] save ", filename)



if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    res = glob.glob("./dataset/portraits/*.jpg")
    
    if not os.path.exists("./dataset/labels"):
        os.mkdir("./dataset/labels")

    """
    # LOAD UNET MODEL
    """
    class_num = 7
    model = smp.Unet('timm-mobilenetv3_large_100', encoder_weights='imagenet', classes=class_num, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to("cuda")
    model.load_state_dict(torch.load("./weights/MobileNetV2-Anime-Unet.pt"))

    for index in range(len(res)):
        fname = res[index].split("\\")[1][:-4]
        img = Image.open(res[index]).convert("RGB")
        img = transform1(img).to("cuda")
        img = img.reshape((1, 3, 512, 512))
        out = model(img)[0]
        out = torch.argmax(out, dim=0)
        smap = label2SegmentationMap(out.cpu().detach().numpy(), color_mapping)
        save_image(smap, "./dataset/labels/"+fname+".png")
        #save_image(smap, "./data/train_labels/")
        # print(smap.shape)
        # plt.imshow(smap)
        # plt.show()
        # exit()
    
