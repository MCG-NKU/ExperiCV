import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

class DEL_VGG(nn.Module):
    def __init__(self):
        super().__init__()

        # block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.act = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2)


        self.conv1_2_down = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv1_2_refeat = nn.Conv2d(32, 32, kernel_size=1)
        self.conv1_2_norm = nn.BatchNorm2d(32)

        self.conv2_2_down = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2_2_refeat = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2_2_norm = nn.BatchNorm2d(64)

        self.conv3_3_down = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3_3_refeat = nn.Conv2d(128, 64, kernel_size=1)
        self.conv3_3_norm = nn.BatchNorm2d(64)

        self.conv4_3_down = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv4_3_refeat = nn.Conv2d(256, 128, kernel_size=1)
        self.conv4_3_norm = nn.BatchNorm2d(128)

        self.conv5_3_down = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv5_3_refeat = nn.Conv2d(256, 128, kernel_size=1)
        self.conv5_3_norm = nn.BatchNorm2d(128)
        
        self.weight_deconv2 = self._make_bilinear_weights( 4, 64).cuda()
        self.weight_deconv3 = self._make_bilinear_weights( 8, 64).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(16, 128).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(16, 128).cuda()

        self.conv_dim = nn.Conv2d(416, 256,kernel_size=3, padding=1) # possible mistake
        self.conv_dsp = nn.Conv2d(256, 64, kernel_size=1, padding=0)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                nn.init.constant_(m.weight, 0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w
    
    def _crop(self, data, img_h, img_w, crop_h, crop_w):
        # import pdb; pdb.set_trace()
        _, _, h, w = data.size()
        assert(img_h <= h and img_w <= w)
        data = data[:, :, crop_h:crop_h + img_h, crop_w:crop_w + img_w]
        return data

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]
        conv1_1 = self.act(self.conv1_1(x))
        conv1_2 = self.act(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.act(self.conv2_1(pool1))
        conv2_2 = self.act(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.act(self.conv3_1(pool2))
        conv3_2 = self.act(self.conv3_2(conv3_1))
        conv3_3 = self.act(self.conv3_3(conv3_2))
        pool3   = self.pool3(conv3_3)

        conv4_1 = self.act(self.conv4_1(pool3))
        conv4_2 = self.act(self.conv4_2(conv4_1))
        conv4_3 = self.act(self.conv4_3(conv4_2))
        pool4   = self.pool4(conv4_3)

        conv5_1 = self.act(self.conv5_1(pool4))
        conv5_2 = self.act(self.conv5_2(conv5_1))
        conv5_3 = self.act(self.conv5_3(conv5_2))

        # conv1 side output
        conv1_2_down = self.act(self.conv1_2_down(conv1_2))
        conv1_2_refeat = self.conv1_2_refeat(conv1_2_down)
        conv1_2_norm = self.conv1_2_norm(conv1_2_refeat)

        # conv2 side output
        conv2_2_down = self.act(self.conv2_2_down(conv2_2))
        conv2_2_refeat = self.conv2_2_refeat(conv2_2_down)
        conv2_2_norm = self.conv2_2_norm(conv2_2_refeat)
        upfeat2_2 = F.conv_transpose2d(conv2_2_norm, self.weight_deconv2, stride=2)
        feature_dsn2 = self._crop(upfeat2_2, img_h, img_w, 0, 0)

        # conv3 side output
        conv3_3_down = self.act(self.conv3_3_down(conv3_3))
        conv3_3_refeat = self.conv3_3_refeat(conv3_3_down)
        conv3_3_norm = self.conv3_3_norm(conv3_3_refeat)
        upfeat3_3 = F.conv_transpose2d(conv3_3_norm, self.weight_deconv3, stride=4)
        feature_dsn3 = self._crop(upfeat3_3, img_h, img_w, 0, 0)  # possible mistake

        # conv4 side output
        conv4_3_down = self.act(self.conv4_3_down(conv4_3))
        conv4_3_refeat = self.conv4_3_refeat(conv4_3_down)
        conv4_3_norm = self.conv4_3_norm(conv4_3_refeat)
        upfeat4_3 = F.conv_transpose2d(conv4_3_norm, self.weight_deconv4, stride=8)
        feature_dsn4 = self._crop(upfeat4_3, img_h, img_w, 0, 0)

        # conv5 side output
        conv5_3_down = self.act(self.conv5_3_down(conv5_3))
        conv5_3_refeat = self.conv5_3_refeat(conv5_3_down)
        conv5_3_norm = self.conv5_3_norm(conv5_3_refeat)
        upfeat5_3 = F.conv_transpose2d(conv5_3_norm, self.weight_deconv5, stride=8)
        feature_dsn5 = self._crop(upfeat5_3, img_h, img_w, 0, 0)   ## possible mistake
        # feature_dsn5 = upfeat5_3
        # import pdb; pdb.set_trace()
        try:
            feature = torch.cat((conv1_2_norm, feature_dsn2, feature_dsn3, feature_dsn4, feature_dsn5), dim=1)
        except Exception as e :
            print(e)
            import pdb; pdb.set_trace()

        conv_dim = self.act(self.conv_dim(feature))
        conv_dsp = self.conv_dsp(conv_dim)


        return conv_dsp
