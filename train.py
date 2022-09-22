
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
import albumentations as A 
import cv2 

if __name__ == "__main__":
    # CREATE MobileNetV2 model
    class_num = 7 
    model = smp.Unet('mit_b0', encoder_weights='imagenet', classes=class_num, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to("cuda")
    #model = smp.Unet('timm-mobilenetv3_large_100', encoder_weights='imagenet', classes=class_num, activation=None, encoder_depth=5, decoder_channels=[128, 128, 64, 32, 16]).to("cuda")
    # summary(model, (3, 512, 512))
    # exit()
    """
    # hyper parmas 
    """
    batch_size = 8
    train_transform = A.Compose([A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)])

    valid_transform = A.Compose([A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)])
    train_dataset = SegmentationDataset("./data/train/", "./data/train_label/", 0, train_transform)
    valid_dataset = SegmentationDataset("./data/valid/", "./data/valid_label/", 0, valid_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    lr = 1e-4
    epoch = 150
    class_w = torch.FloatTensor([  1,   1,   1,   1,  1, 5,   1.5]).cuda() 
    criterion = nn.CrossEntropyLoss(weight=class_w)
    early_stopping = EarlyStopping(patience=5, verbose=True, path="./weights/mit_b0_test.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epoch, steps_per_epoch=len(train_loader))
    min_loss = np.inf
    train_loss_record = []
    valid_loss_record = []
    count = 0
    for e in range(epoch):
        model.train()
        running_loss = 0
        for index, (img, mask) in enumerate(tqdm(train_loader)):
            img = img.to("cuda")
            mask = mask.to("cuda")
            # forward 
            out = model(img)
            # compute loss 
            loss = criterion(out, mask)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # upaate lr 
            sched.step()
            running_loss += loss.item() * img.shape[0]
        
        # start validation 
        model.eval()
        running_valid_loss = 0
        with torch.no_grad():
            for index, (img, mask) in enumerate(tqdm(valid_loader)):
                img = img.to("cuda")
                mask = mask.to("cuda")
                out = model(img)
                loss = criterion(out, mask)
                running_valid_loss += loss.item() * img.shape[0]
                
        train_loss_record.append(running_loss/len(train_dataset))
        valid_loss_record.append(running_valid_loss/len(valid_dataset))
        print("Epoch {}: Loss {} LossV {} ".format(e, running_loss/len(train_dataset), running_valid_loss/len(valid_dataset)))
        early_stopping(running_valid_loss/len(valid_dataset), model, e)
        if early_stopping.early_stop:
            print("[INFO] Early Stopping")
            break
    draw_loss(train_loss_record, valid_loss_record, "./loss.png")