import torch
import numpy as np
import math
import cv2 
import os
import torchvision
from PIL import Image
from basic_ops import *


def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    #return type: [(y1, x1, y2, x2)]
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []
    for (thetai, ri) in point_list:  #利用霍夫反变换的公式对图像中的所有的集合进行霍夫反变换
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        x = np.round(r / cosi + W / 2)
        b_points.append((0, int(x), H-1, int(x)))
    return b_points  