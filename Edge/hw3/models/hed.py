import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv5 = nn.Sequential(
            # conv5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
         
        self.dsn1 = ???
        self.dsn2 = ???
        self.dsn3 = ???
        self.dsn4 = ???
        self.dsn5 = ???
        self.fuse = ???

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
        d4 = ???
        d5 = ???

        # dsn fusion output
        fuse = ???
        
        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        fuse = F.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse
