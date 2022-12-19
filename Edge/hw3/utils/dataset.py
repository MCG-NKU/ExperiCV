import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join, split
from utils.utils import is_image_file
import os
from PIL import Image
import random
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms
# def is_image_file(x):
#     return any(x.endswith(w)for w in ['jpg','png'])

class BSDS500(data.Dataset):
    
    def __init__(self,dataPath, gtPath, transform=None, target_transform=None):
        super(BSDS500, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.gtPath = gtPath
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        img_path = os.path.join(self.dataPath,self.image_list[index])
        label_path = os.path.join(self.gtPath,self.image_list[index])

        img = Image.open(img_path).convert('RGB') 
        label = Image.open(label_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)


        return img, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

class BSDS500_TEST(data.Dataset):

    def __init__(self,dataPath, transform=None):
        super(BSDS500_TEST, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.transform = transform

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        img_path = os.path.join(self.dataPath,self.image_list[index])

        img = Image.open(img_path).convert('RGB') 
        name = self.image_list[index]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, name

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
