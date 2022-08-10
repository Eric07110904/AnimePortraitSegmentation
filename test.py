from operator import mod
from dataset import SegmentationDataset
from torchsummary import summary 
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
from tqdm.auto import tqdm
import numpy as np
from utils.draw import draw_loss
from utils.tools import EarlyStopping 
import cv2 
import albumentations as A
if __name__ == "__main__":
    # CREATE MobileNetV2 model
    class_num = 7
    model = smp.Unet('timm-mobilenetv3_large_100', encoder_weights='imagenet', classes=class_num, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to("cuda")
    # summary(model, (3, 512, 512))
    """
    # hyper parmas 
    """
    valid_transform = A.Compose([A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)])
    batch_size = 1
    test_dataset = SegmentationDataset("./data/valid/", "./data/valid_label/", 0, valid_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model.load_state_dict(torch.load("./weights/MobileNetV2-Anime-Unet.pt"))
    model.eval()
    for index, (img, mask) in enumerate(test_loader):
        img = img.to("cuda")
        mask = mask.to("cuda")
        out = model(img)[0]
        out = torch.argmax(out, dim=0)
        print(out.shape)
        out = out.cpu().detach().numpy()
        mask = mask[0].cpu().detach().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(mask)
        ax2.imshow(out)
        plt.show()
        