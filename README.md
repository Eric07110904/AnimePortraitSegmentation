# AnimePortraitSegmentation 

In this repository, we create an AI model to perform semantic segmentation for Anime Portrait. 

## Dataset 
Fist, We need training data about original image and it's corresponding mask image, so we annotate **200** anime portraits (512x512) from "Danbooru2019-Portraits Dataset".<br>
Note: Totally there are **7 tags (background、skin、face、cloth、eye、mouth、hair)** we need to annotate! 

![Image](/image/image&mask.jpg "image and mask")

Because 200 images are not enough for predicting semantic mask, therefore we use following data augmentation to create fake data!

1. horizontal、vertical filp
2. GridDistortion 
3. RandomBrightnessContrast 
4. GaussNoise
5. Rotation

![Image](/image/aug_image.jpg "aug_image")
![Image](/image/aug_mask.jpg "aug_mask")

## Method 
After doing data augementation, we get roughly 3000 paired datas to train our semantic segmentation model.<br>
In AI model, We use **MobileNetV3 as encoder** and **Unet as decoder** to complete this task, it can easliy done by using [this repo!](https://github.com/qubvel/segmentation_models.pytorch) 
```python
model = smp.Unet('timm-mobilenetv3_large_100', encoder_weights='imagenet', classes=class_num, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to("cuda")
```

## Result 
![Image](/image/res1.jpg "res1")
![Image](/image/res2.jpg "res2")
![Image](/image/res3.jpg "res3")

## Enviroment
||details|
|---|---|
|OS|Windows10|
|CPU|AMD|
|GPU|NVIDIA RTX 2060 6GB|
|language|Python|
|framework|pytorch|

## Reference 
[1] [Deep Learning Project — Drawing Anime Face with Simple Segmentation Mask](https://medium.com/@steinsfu/drawing-anime-face-with-simple-segmentation-mask-ca955c62ce09)

[2] [Anime-Semantic-Segmentation-GAN](https://github.com/pit-ray/Anime-Semantic-Segmentation-GAN)