import torchvision
from torchvision.models import VGG16_Weights
from torch.nn import Linear
from torch.nn import MaxPool2d
import cv2
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time
import os
from torchvision.datasets import MNIST
import torchvision.transforms as T
import numpy as np
from torch_geometric.transforms import ToSLIC
import torchvision.transforms as transforms

from model import DEL_VGG
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'
transform = T.Compose([T.ToTensor(), ToSLIC(n_segments=64)])

train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        all_three = []
        data_path_root = '/home/ubuntu/code/DEL/BSDS500/'
        for line in fh:
            line = line.rstrip()
            words = line.split()
            image_path = os.path.join(data_path_root+words[0])
            gt_path = os.path.join(data_path_root+words[1])
            sp_path = os.path.join(data_path_root+words[2])
            # import pdb; pdb.set_trace()
            all_three.append((image_path, gt_path, sp_path))
            self.imgs = all_three 
            self.transform = transform
            self.target_transform = target_transform
        self.mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
    def __getitem__(self, index):
        image, gt, sp = self.imgs[index]
        img = cv2.imread(image)
        img = np.array(img, dtype=np.float32)
        img = (img - self.mean).transpose((2, 0, 1))
        gt = cv2.imread(gt, flags=cv2.IMREAD_GRAYSCALE)

        sp = Image.open(sp)
        sp = np.array(sp)        
        return img, gt, sp
    def __len__(self):
        return len(self.imgs)


def sim_loss(superpixel_pooling_out, superpixel_seg_label, sp_label):
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]
    height = sp_label.shape[1]
    width = sp_label.shape[2]
    total_loss = 0
    ##
    ## write your sim loss here
    ##
   
    return total_loss

def test(conv_dsp, sp_label, bount, min_size):
    ##
    ## write your test code here
    ##




def superpixel_pooling(conv_dsp, seg_label, sp_label):
    ## 
    ## write your code here
    ##


path = './BSDS500/train_gpu.txt'
dataset = MyDataset(path)
net = DEL_VGG()
net = net.to(device)
total_epoch = 1
optm = torch.optim.SGD(net.parameters(), lr=1e-5, weight_decay=0.0002, momentum=0.9)
loader = DataLoader(dataset, batch_size=1)
for epoch in range(total_epoch):
    for input, gt, sp in tqdm(loader):
        input = input.to(device)
        conv_dsp = net(input)
        superpixel_pooling_out, superpixel_seg_label = superpixel_pooling(conv_dsp, gt, sp)
        loss = sim_loss(superpixel_pooling_out, superpixel_seg_label, sp)
        optm.zero_grad()
        loss.backward()
        optm.step()
        print("cur loss: ", loss)
