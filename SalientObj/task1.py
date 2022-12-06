import os
import numpy as np
import random
from PIL import Image, ImageEnhance

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 读取图片
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# 几种数据增强策略
# 随机翻转
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label
# 随机裁剪
def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)
# 随机旋转
def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label
# 颜色增强
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
# 随机噪声
def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class SalObjData(Dataset):
    def __init__(self, data_root, mode, image_size):
        # 数据变换
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])
        
        self.data_root = data_root
        assert mode in ['train', 'test']
        self.mode = mode
        # get filenames
        image_dir = os.path.join(self.data_root, 'T' + self.mode[1:], 'Image')
        gt_dir = os.path.join(self.data_root, 'T' + self.mode[1:], 'GT')
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.gts = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')]

        self.images = sorted(self.images)    
        self.gts = sorted(self.gts)  
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, label_path = self.images[index], self.gts[index]
        # 数据读取
        image = rgb_loader(image_path)
        label = binary_loader(label_path)

        # 数据增强
        if self.mode == 'train':
            image, label = self.aug_data(image=image, label=label)

        # 数据处理
        image = self.img_transform(image)
        if self.mode == 'train':
            label = self.gt_transform(label)
        else:   # test， 保存原始的label,用于测试
            label = np.asarray(label, np.float32)    
        return image, label
                
    def aug_data(self, image, label):
        image, label = cv_random_flip(image, label)
        image, label = randomCrop(image, label)
        image, label = randomRotation(image, label)
        image = colorEnhance(image)
        label = randomPeper(label)
        return image, label

def test():
    # 验证是否加载成功
    data_root = '/home/zhangxuying/Datasets/SOD-Dataset/DUTS'

    # 训练集
    train_data = SalObjData(data_root=data_root, mode='train', image_size=256)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=32, 
        shuffle=True, 
        num_workers=8
    )
    train_images, train_labels = iter(train_loader).next()
    print('train_images: ', train_images.shape)
    print('train_labels: ', train_labels.shape)

    # 测试集 
    test_data = SalObjData(data_root=data_root, mode='test', image_size=256)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1, 
        shuffle=False, 
        num_workers=8
    )
    test_images, test_labels = iter(test_loader).next()
    print('test_images: ', test_images.shape)
    print('test_labels: ', test_labels.shape)

test()
