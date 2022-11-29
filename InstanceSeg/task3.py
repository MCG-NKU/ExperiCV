from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import time
 
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_time(image_path, use_gpu=False):
    """
    TODO: 测试Mask R-CNN模型在CPU和GPU平台上的时间
    """
    return -1

if __name__ == '__main__':
    cpu_time = sum([get_time('./demo.jpg', use_gpu=False) for _ in range(10)])
    gpu_time = sum([get_time('./demo.jpg', use_gpu=True) for _ in range(10)])
    print('cpu time:', cpu_time / 10.0)
    print('gpu time:', gpu_time / 10.0)
    print(f'Running time on GPU is {int(cpu_time / gpu_time)} times faster than on CPU')
