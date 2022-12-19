import torch
import numpy as np
import math
import cv2 
import os
import torchvision
from PIL import Image
from basic_ops import *


def convert_line_to_hough(line, size=(32, 32)):
  #请根据霍夫变换定义实现空间的转换过程，补全下方代码
    H, W = size
    theta = line.angle()
    alpha = theta + np.pi / 2
    r = line.coord[1] - W/2
    return alpha, r
  
def line2hough(line, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)
    return alpha, r