# 计算机视觉实验课-实例分割

## 任务一： 双线性插值实验
双线性插值算法是RoI Align中使用的插值算法，其可以没有误差地进行特征图大小的转化。请根据讲义理解框架代码，在`TODO`部分根据公式实现双线性插值。

```python
    # 根据公式进行双线性插值
    dst = np.zeros(shape=(dst_height, dst_width, src_channel), dtype=np.float32)
    """
    TODO: 根据公式进行双线性插值
    """
    return dst
```

## 任务二：使用Pytorch中Mask R-CNN进行实例分割

Pytorch是常用的深度学习编程框架，其中已经提供了Mask R-CNN的接口，并且可以下载其预训练模型，通过加载模型，可以直接使用Mask R-CNN模型进行实例分割。其中，框架代码实现了使用Mask R-CNN模型对图像进行分割，并获得输出结果，以及获得不同颜色掩码的函数，任务二要求学习框架代码关键函数函数的实现思路，并根据框架代码中实现的函数封装一个使用Mask R-CNN模型进行实例分割的API，API的输入为原始图片，输出为标记好类别，检测框，以及分割掩码的图片，并将其以jpg形式保存。

```python
def instance_segmentation(img_path, threshold=0.5, rect_th=3, text_size=2, text_th=2):
    """
    TODO: 根据上述函数封装Mask R-CNN模型实例分割函数
    """
    pass
```

## 任务三：对比Mask R-CNN模型在CPU和GPU平台的推理速度差异

GPU极大地加快了深度学习模型的训练和推理速度，本任务要求对比预训练好的Mask R-CNN模型对于同一张图片分别在CPU和GPU平台上的推理时间差异，体会GPU对于深度学习模型的加速作用。完善框架代码函数，记录对于相同的图片模型在不同平台上的不同时间。

```python
def get_time(image_path, use_gpu=False):
    """
    TODO: 测试Mask R-CNN模型在CPU和GPU平台上的时间
    """
    return -1
```

## 探索任务

在Mask R-CNN模型提出之后，后续有很多工作在实例分割任务上取得了更好的效果，其中包括适用于实例分割的检测头，以及适用于实例分割任务的骨干网络。本任务为开放性任务，要求调研下述模型或者自选的实例分割检测头以及骨干网络模型，加载论文提供的预训练模型进行推理，分析其性能提升的效果以及具体的原因。
- 实例分割算法：[RDPNet](https://arxiv.org/pdf/2008.12416.pdf)、[HTC](https://arxiv.org/pdf/1901.07518.pdf)、[CenterMask](https://arxiv.org/pdf/1911.06667.pdf)等；
- 骨干网络：[P2T](https://arxiv.org/pdf/2106.12011.pdf)、[Res2Net](https://arxiv.org/pdf/1904.01169.pdf)等。