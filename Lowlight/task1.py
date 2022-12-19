# -*- coding: UTF-8 -*-
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2

def Correction(img,gamma=2.2):
    """
    TODO: 实现图像Gamma校正代码
    """
    pass
    
def Zero_DCE(img,n,alpha): #img is input image,n is the iteration,alpha:[-1,1].
    """
    TODO: 实现图像Zero-DCE曲线校正代码
    """
    pass

if __name__ == "__main__":
    img = imageio.imread("33.jpg")
    imageio.imsave("correctionimg.jpg",Correction(img))
    imageio.imsave("zero_dceimg.jpg",Zero_DCE(img,3,-0.5))

    plt.figure(figsize=(15,10))
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Input Image')
    
    plt.subplot(132)
    plt.imshow(Correction(img))
    plt.title('Gamma Correction Image')
    
    plt.subplot(133)
    plt.imshow(Zero_DCE(img))
    plt.title('Zero-DCE curve Correction Image')
    plt.show()   