import os
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from task1 import SalObjData

# 包含卷积操作和ReLU的基本卷积块
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# DSS网络结构类
class DSS(nn.Module):
    def __init__(self):
        super(DSS, self).__init__()
        base = torchvision.models.vgg16(pretrained=True).features

        self.layer1 = nn.Sequential(*list(base.children())[0:4])
        self.layer2 = nn.Sequential(*list(base.children())[4:9])
        self.layer3 = nn.Sequential(*list(base.children())[9:16])
        self.layer4 = nn.Sequential(*list(base.children())[16:23])
        self.layer5 = nn.Sequential(*list(base.children())[23:30])
        self.pool5 = list(base.children())[30]

        # layer6 (pool5) 侧向输出
        self.dsn6_conv = nn.Sequential(
            BasicConv2d(512, 512, kernel_size=7, padding=3),
            BasicConv2d(512, 512, kernel_size=7, padding=3),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        )

        # layer5 侧向输出
        self.dsn5_conv = nn.Sequential(
            BasicConv2d(512, 512, kernel_size=5, padding=2),
            BasicConv2d(512, 512, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        )
        # layer4 侧向输出
        self.dsn4_conv1 = nn.Sequential(
            BasicConv2d(512, 256, kernel_size=5, padding=2),
            BasicConv2d(256, 256, kernel_size=5, padding=2),
            BasicConv2d(256, 1, kernel_size=1)
        )
        # 上采样
        self.upsample_64 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_54 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 输出
        self.dsn4_conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        # layer3 侧向输出
        self.dsn3_conv1 = nn.Sequential(
            BasicConv2d(256, 256, kernel_size=5, padding=2),
            BasicConv2d(256, 256, kernel_size=5, padding=2),
            BasicConv2d(256, 1, kernel_size=1)
        )
        self.upsample_63 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_53 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dsn3_conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        # layer2 侧向输出
        self.dsn2_conv1 = nn.Sequential(
            BasicConv2d(128, 128, kernel_size=3, padding=1),
            BasicConv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        )
        self.upsample_62 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_52 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_42 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dsn2_conv2 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)

        # layer1 侧向输出
        self.dsn1_conv1 = nn.Sequential(
            BasicConv2d(64, 128, kernel_size=3, padding=1),
            BasicConv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        )
        self.upsample_61 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample_51 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_41 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_31 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dsn1_conv2 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        # 融合输出
        self.upsample_60 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample_50 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_40 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_30 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_20 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dsn0_conv = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1)
    
    def forward(self, x):
        '''
        x: (bs, 3, 256, 256)
        '''
        x1 = self.layer1(x)         # (bs, 64, 256, 256)
        x2 = self.layer2(x1)        # (bs, 128, 128, 128)
        x3 = self.layer3(x2)        # (bs, 256, 64, 64)
        x4 = self.layer4(x3)        # (bs, 512, 32, 32)
        x5 = self.layer5(x4)        # (bs, 512, 16, 16)
        x6 = self.pool5(x5)         # (bs, 512, 8, 8)   

        # side 6
        x_dsn6 = self.dsn6_conv(x6)
        # print("x_dsn6: ", x_dsn6.shape)
        # side 5
        x_dsn5 = self.dsn5_conv(x5)
        # print("x_dsn5: ", x_dsn5.shape)
        # side 4
        x_dsn4_init = self.dsn4_conv1(x4)  
        x_dsn_up64 = self.upsample_64(x_dsn6)    
        x_dsn_up54 = self.upsample_54(x_dsn5) 
        x_dsn4 = self.dsn4_conv2(torch.cat([x_dsn4_init, x_dsn_up64, x_dsn_up54], 1))
        # print("x_dsn4: ", x_dsn4.shape) 
        # side 3
        x_dsn3_init = self.dsn3_conv1(x3)   
        x_dsn_up63 = self.upsample_63(x_dsn6)    
        x_dsn_up53 = self.upsample_53(x_dsn5) 
        x_dsn3 = self.dsn3_conv2(torch.cat([x_dsn3_init, x_dsn_up63, x_dsn_up53], 1))
        # print("x_dsn3: ", x_dsn3.shape) 
        # side 2
        x_dsn2_init = self.dsn2_conv1(x2)  
        x_dsn_up62 = self.upsample_62(x_dsn6)    
        x_dsn_up52 = self.upsample_52(x_dsn5) 
        x_dsn_up42 = self.upsample_42(x_dsn4)    
        x_dsn_up32 = self.upsample_32(x_dsn3) 
        x_dsn2 = self.dsn2_conv2(torch.cat([x_dsn2_init, x_dsn_up62, x_dsn_up52, x_dsn_up42, x_dsn_up32], 1))
        # print("x_dsn2: ", x_dsn2.shape) 
        # side 1
        x_dsn1_init = self.dsn1_conv1(x1)   
        x_dsn_up61 = self.upsample_61(x_dsn6)    
        x_dsn_up51 = self.upsample_51(x_dsn5) 
        x_dsn_up41 = self.upsample_41(x_dsn4)    
        x_dsn_up31 = self.upsample_31(x_dsn3) 
        x_dsn1 = self.dsn1_conv2(torch.cat([x_dsn1_init, x_dsn_up61, x_dsn_up51, x_dsn_up41, x_dsn_up31], 1))
        # print("x_dsn1: ", x_dsn1.shape) 
        # fusion
        x_dsn_up60 = self.upsample_60(x_dsn6)    
        x_dsn_up50 = self.upsample_50(x_dsn5) 
        x_dsn_up40 = self.upsample_40(x_dsn4)    
        x_dsn_up30 = self.upsample_30(x_dsn3) 
        x_dsn_up20 = self.upsample_30(x_dsn3) 
        x_dsn0 = self.dsn0_conv(torch.cat([x_dsn_up60, x_dsn_up50, x_dsn_up40, x_dsn_up30, x_dsn_up20, x_dsn1], 1))
        # print("x_dsn0: ", x_dsn0.shape) 
        return x_dsn_up60, x_dsn_up50, x_dsn_up40, x_dsn_up30, x_dsn_up20, x_dsn1, x_dsn0


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dss')
    parser.add_argument('--n_epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')

    parser.add_argument('--data_root', type=str, default='/home/zhangxuying/Datasets/SOD2COD/',
                        help='the training data root')
    parser.add_argument('--sod', type=str, default='ICON-P', help='sod method')
    parser.add_argument('--is_multi_sf', type=bool, default=False, help='wheather using multi-level sf or not')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')

    parser.add_argument('--save_root', type=str,
                        default='./snapshot/',
                        help='the path to save model and log')
    opt = parser.parse_args()
    print(opt)


    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = True
    
    # 模型
    model = DSS()
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load)['state_dict'])
        print('load model from ', opt.load)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.epoch, eta_min=opt.lr*0.01)

    # 训练集
    train_data = SalObjData(data_root=opt.data_root, mode='train', image_size=256)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=32, 
        shuffle=True, 
        num_workers=8
    )
    total_step = len(train_loader)

    save_path = opt.save_root +  '/' + opt.model_name + '/'
    os.makedirs(save_path, exist_ok=True)

    print("Start train...")
    for epoch in range(1, opt.n_epoch):
        # schedule
        cosine_schedule.step()
        model.train()
        loss_all = 0
        epoch_step = 0

        for i, (images, gts) in enumerate(train_loader, start=1):

            images = images.cuda()
            gts = gts.cuda()

            preds = model(x=images)
            loss = structure_loss(preds, gts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.n_epoch, i, total_step, loss.data))
        
        # 记录当前epoch的loss
        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}]: {}'.format(epoch, opt.n_epoch, loss_all))
              
        # 每间隔10个epoch保存一次模型
        if epoch % 10 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, save_path + 'Net_epoch_{}.pth'.format(epoch))
        

