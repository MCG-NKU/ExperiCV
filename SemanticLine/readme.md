# 计算机视觉实验课-语义线检测

## 实验环境搭建
进入实验目录，执行如下指令：
```
conda create −−name SDL python =3.8 −y
conda activate SDL
pip install torch
python setup.py build
python setup.py install – user
pip install pyyaml
pip install tqdm
pip install torchvision
pip install matplotlib
pip install opencv−python
pip install scikit −image
pip install pot
```

## 任务一： 霍夫变换
作业文件：task1/
空间变换部分利用深度霍夫变换进行特征空间转换，将图像空间的特征转换到参数空间中。该部分输入的
图像空间特征，并给定变换的角度和距离量化尺度，根据量化尺度对输入的图像空间特征根据霍夫变换原理转
换到量化大小的参数空间中。原始图像空间的每条直线的特征，通过该变换转换到参数空间的每个点中。
```
def  convert_line_to_hough(line, size=(32, 32)):
        #请根据霍夫变换定义实现空间的转换过程，补全上述代码
        H, W = ？
        theta = ？
        alpha = ？
        r = ？
        return alpha, r
def line2hough(line, size=(32, 32)):
        H, W = size
        alpha, r = convert_line_to_hough(line, size) 
```

## 任务二：前向传播网络搭建

文件：task2.
参数空间中每个点的特征表示了原始图像空间中一条直线特征的汇聚，且根据霍夫变换的原则，相近的直
线会被表示到参数空间中的近邻点上。因此，在参数空间进行局部特征聚合操作即可完成原始图像空间中的非
局部特征聚合。使用多个卷积操作在参数空间中进行特征聚合，构成内容感知的特征聚合模块，对不同直线的
特征进行交互融合。得到融合和增强的特征最终通过 1x1 卷积进行点的检测，将表示为语义线的点标识为正样
本，其他为负样本进行判别。
```
import torch
import numpy as np 
import torch.nn as nn
from model.backbone.fpn import FPN101, FPN50, FPN18, ResNext50_FPN
from model.backbone.mobilenet import MobileNet_FPN
from model.backbone.vgg_fpn import VGG_FPN
from model.backbone.res2net import res2net50_FPN
from model.dht import DHT_Layer
class Net(nn.Module):
    def __init__(self, numAngle, numRho, backbone):
        super(Net, self).__init__()
        if backbone == 'resnet18':
            self.backbone = FPN18(pretrained=True, output_stride=32)
            output_stride = 32
        if backbone == 'resnet50':
            self.backbone = FPN50(pretrained=True, output_stride=16)
            output_stride = 16
        if backbone == 'resnet101':
            self.backbone = FPN101(output_stride=16)
            output_stride = 16
        if backbone == 'resnext50':
            self.backbone = ResNext50_FPN(output_stride=16)
            output_stride = 16
        if backbone == 'vgg16':
            self.backbone = VGG_FPN()
            output_stride = 16
        if backbone == 'mobilenetv2':
            self.backbone = MobileNet_FPN()
            output_stride = 32
        if backbone == 'res2net50':
            self.backbone = res2net50_FPN()
            output_stride = 32
        if backbone == 'mobilenetv2':
            self.dht_detector1 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho)
            self.dht_detector2 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho // 2)
            self.dht_detector3 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho // 4)
            self.dht_detector4 = DHT_Layer(32, 32, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            self.last_conv = nn.Sequential(
                nn.Conv2d(128, 1, 1)
            )
        else:
            self.dht_detector1 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho)
            self.dht_detector2 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // 2)
            self.dht_detector3 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // 4)
            self.dht_detector4 = DHT_Layer(256, 128, numAngle=numAngle, numRho=numRho // (output_stride // 4))
            self.last_conv = nn.Sequential(
                nn.Conv2d(512, 1, 1)
            )
        self.numAngle = numAngle
        self.numRho = numRho
    def upsample_cat(self, p4):
        p4 = nn.functional.interpolate(p4, size=(self.numAngle, self.numRho), mode='bilinear')
        return p4
    def forward(self, x):
        _, _, _, p4 = ？ #经过backbone网络提取特征
        p4 = ？  #对stage4特征进行霍夫变换
        #分别在每个尺度进行局部信息聚合，将处理后的Y4 特征图双线性插值到 Y1 的尺寸，以匹配不同层的特征图的分辨率然后将其在通道维度连接在一起，得到融合后的多尺度参数空间表征，
        cat = ？
        #在连接后的特征图上应用一个 1 × 1 卷积层生成像素级别的预测。
        logist = ？
        return logist
```

## 任务三：霍夫反变换
作业文件：task3.
空间反变换将参数空间中检测到的表示语义线的点进行霍夫反变换，通过点的位置坐标（距离和角度）计
算得到图像空间中直线的矢量表达式。
```
        def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
                H, W = size 
                irho =   ？
                itheta = ？
                b_points = []   
                for (thetai, ri) in point_list:   #利用霍夫反变换的公式对图像中的所有的集合进行霍夫反变换
                    theta = ？
                    r = ？
                    cosi = np.cos(theta) / irho
                    sini = np.sin(theta) / irho
                    x = ？
        return b_points  
```
## 探索任务

探索1
本次使用的是resnet50+FPN网络进行训练，学有余力的同学可以使用我们提供的其他的网络进行实验效果，观察不同网络对结果的影响。同时本次仅提供小型数据集，同学们可以使用更大的数据集（比如NKL数据集，SEL数据集）进行训练。NKL数据集和SEL数据集下载地址：[NKL和SEL](https://kaizhao.net/nkl)
探索2
另外同学们可以参考[Deep Hough Transform for Semantic Line Detection](https://arxiv.org/abs/2003.04676) 这篇文章，学习搭建更复杂的多尺度网络，观察实验效果。
