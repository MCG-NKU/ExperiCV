# 人工智能实验参考代码-显著性物体检测
本实验旨在定义一个基于卷积神经网络的显著性物体检测模型，并在该任务的公开数据集[DUTS](https://github.com/LUSSeg/ImageNet-S#imagenet-s-dataset-preparation)上进行训练与测试。

## 任务一： 数据集加载实验
数据是人工智能研究的三大核心要素（算法、算力、数据）之一，本任务要求读者基于Pytorch框架提供的Dataset类，自定义一个数据集类来加载显著性物体检测任务的公开数据集DUTS。该数据集的组织方式如下：

```mind
DUTS
    Train
        Image
        GT
        train.txt
    Test
        Image
        GT
        test.txt  
```
其中，Image和GT分别为存放原始图像和标注图像的文件夹。  
具体的代码实现请参考```task1.py```。

## 任务二：显著性物体检测模型的搭建和训练
模型/算法是人工智能研究的另一个关键要素，


Pytorch是常用的深度学习编程框架，其中已经提供了Mask R-CNN的接口，并且可以下载其预训练模型，通过加载模型，可以直接使用Mask R-CNN模型进行实例分割。其中，框架代码实现了使用Mask R-CNN模型对图像进行分割，并获得输出结果，以及获得不同颜色掩码的函数，任务二要求学习框架代码关键函数函数的实现思路，并根据框架代码中实现的函数封装一个使用Mask R-CNN模型进行实例分割的API，API的输入为原始图片，输出为标记好类别，检测框，以及分割掩码的图片，并将其以jpg形式保存。

```python
def instance_segmentation(img_path, threshold=0.5, rect_th=3, text_size=2, text_th=2):
    """
    TODO: 根据上述函数封装Mask R-CNN模型实例分割函数
    """
    pass
```

## 任务三：显著性物体检测模型的搭建和训练

GPU极大地加快了深度学习模型的训练和推理速度，本任务要求对比预训练好的Mask R-CNN模型对于同一张图片分别在CPU和GPU平台上的推理时间差异，体会GPU对于深度学习模型的加速作用。完善框架代码函数，记录对于相同的图片模型在不同平台上的不同时间。

```python
def get_time(image_path, use_gpu=False):
    """
    TODO: 测试Mask R-CNN模型在CPU和GPU平台上的时间
    """
    return -1
```

## 探索任务：显著性物体检测任务的类别无关性探索

在Mask R-CNN模型提出之后，后续有很多工作在实例分割任务上取得了更好的效果，其中包括适用于实例分割的检测头，以及适用于实例分割任务的骨干网络。本任务为开放性任务，要求调研下述模型或者自选的实例分割检测头以及骨干网络模型，加载论文提供的预训练模型进行推理，分析其性能提升的效果以及具体的原因。
- 实例分割算法：[RDPNet](https://arxiv.org/pdf/2008.12416.pdf)、[HTC](https://arxiv.org/pdf/1901.07518.pdf)、[CenterMask](https://arxiv.org/pdf/1911.06667.pdf)等；
- 骨干网络：[P2T](https://arxiv.org/pdf/2106.12011.pdf)、[Res2Net](https://arxiv.org/pdf/1904.01169.pdf)等。
