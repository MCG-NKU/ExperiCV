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
        _, _, _, p4 = 
        p4 =   #对stage4特征进行霍夫变换
        #分别在每个尺度进行局部信息聚合，将处理后的 Y4 特征图双线性插值到 Y1 的尺寸，以匹配不同层的特征图的分辨率然后将其在通道维度连接在一起，得到融合后的多尺度参数空间表征，
        cat = 
        #在连接后的特征图上应用一个 1 × 1 卷积层生成像素级别的预测。
        logist = 
        return logist
