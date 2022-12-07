import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tensorboardX import SummaryWriter

from task1 import SalObjData
from task2 import DSS

_EPS = np.spacing(1)
_TYPE = np.float64


class Smeasure(object):
    '''
    https://github.com/mczhuge/SOCToolbox/blob/main/codes/metrics.py
    '''
    def __init__(self, alpha: float = 0.5):
        self.sms = []   
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)
    
    def _prepare_data(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        '''
        pred, gt \in (0, 255)
        '''
        gt = gt > 128           # 将gt转化为 0和1 组成的矩阵         
        pred = pred / 255       # 将pred转化为区间(0, 1)的值组成的矩阵
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        return pred, gt

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1])
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        h, w = matrix.shape
        if matrix.sum() == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            area_object = np.sum(matrix)
            row_ids = np.arange(h)
            col_ids = np.arange(w)
            x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
            y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                    pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                    weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)


# 损失函数
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


# 训练模型
def train(data_root, model_root, log_root, n_epochs=100, lr=1e-4):

    # 优化器
    optimizer = Adam(model.parameters(), lr)
    # 学习率衰减策略，余弦模拟退火
    cosine_schedule = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs, eta_min=lr*0.1)


    # 加载数据
    train_data = SalObjData(data_root=data_root, mode='train', image_size=256)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=32, 
        shuffle=True, 
        num_workers=8
    )
    total_step = len(train_loader)

    os.makedirs(log_root, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_root, 'dss'))

    model.train()
    loss_all = 0
    epoch_step = 0
    for epoch in range(n_epochs):
        # 学习率更新
        cosine_schedule.step()
        # 分批次加载数据
        for i, (images, gts) in enumerate(train_loader, start=1):
            # 将数据放到gpu上
            images, gts = images.cuda(), gts.cuda()

            optimizer.zero_grad()       # 梯度清零
            preds = model(x=images)     # 前向传播
            loss = structure_loss(preds, gts)   # 计算损失
            loss.backward()             # 梯度回传
            optimizer.step()            # 计算梯度

            epoch_step += 1
            loss_all += loss.data

            # 每间隔20个时间步打印一次log
            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, n_epochs, i, total_step, loss.data))
        
        # 记录每个epoch的损失
        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        # 每间隔10个epoch保存一次模型
        os.makedirs(model_root, exist_ok=True)
        if epoch % 10 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, os.path.join(model_root), 'dss_epoch_{}.pth'.format(epoch))

    
# 测试模型
def test(data_root, model_root):
    # 加载模型参数
    epoch_list = sorted([int(filename.split('epoch_')[-1].split('.')[0]) for filename in os.listdir(model_root)])
    filename = epoch_list[-1]
    model.load_state_dict(torch.load(os.path.join(model_root, filename)))
    model.eval()

    # 加载数据
    test_data = SalObjData(data_root=data_root, mode='test', image_size=256)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1, 
        shuffle=False, 
        num_workers=8
    )
    
    # 评价指标
    SM = Smeasure()
    
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt) in test_loader:
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)                     # 标准化处理,把数值范围控制到(0,1)
                image = image.cuda()

                res = model(x=image)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)                        # 标准化处理,把数值范围控制到(0,1)

                SM.step(pred=res*255, gt=gt*255)

                pbar.update()
    sm = SM.get_results()['sm'].round(3)

    print('the S-measure of our customed dss model is {}'.format(sm))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/zhangxuying/Datasets/SOD-Dataset/DUTS')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_root', type=str, default='./snapshot/models/')
    parser.add_argument('--log_root', type=str, default='./snapshot/logs/')
    parser.add_argument('--n_epochs', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    args = parser.parse_args()

    # 加载模型，并放置到gpu上
    model = DSS().cuda()

    # 训练 or 测试模型
    if args.mode == 'train':
        train(data_root=args.data_root, model_root=args.model_root, log_root=args.log_root, n_epochs=args.n_epochs, lr=args.lr)
    else:
        test(data_root=args.data_root, model_root=args.model_root)
