# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class L_exp(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        """
        TODO: 定义均值函数，用来获得图像块的像素均值
        self.pool = ???
        """
        self.mean_val = mean_val
    def forward(self, x ):
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        """
        TODO: 根据公式补全曝光控制损失
        exp = ??? 
        """
        pass

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x ):
        """
        TODO: 计算图像像素均值;将像素均值拆分为RGB三部分;计算两两通道之间的方差
        mean_rgb = ???
        mr,mg, mb = ???
        Drg = ???, Drb = ???, Dgb = ???
        """
        color = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return color