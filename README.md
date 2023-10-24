# DeepLabV3Plus_Pytorch

## Precaution
This warehouse mainly uses two model technologies, ___DeepLabV3___ and ___DeepLabV3Plus___, and mixes ___MobileNetV2___, ___ResNet101___, and ___HRNetV2___ to implement segmentation tasks. Slightly modified based on the source code. You are free to use ___VOC___, ___CityScapes___ datasets, and your own datasets to train any segmentation models. Download links for the VOC and CityScapes datasets are below.

Download VOC and CityScapes：
链接：https://pan.baidu.com/s/1fy0R6NJo0YanfD4n6kgCiw?pwd=0grl 
提取码：0grl 
--来自百度网盘超级会员V3的分享

Note: Before training the model, you need to enter the ___train_gpu.py___ file to modify the path and batchsize of your own data set. If you enable automatic mixed precision and then load the weight file under the checkpoints file, you may encounter a ComplexFloat error during the forward process. (If you know how to solve this problem, you are welcome to suggest a solution, thank you very much!)

## TRAIN & EVALUATE MODELS
1. cd DeepLabV3Plus
2. python train_gpu.py
