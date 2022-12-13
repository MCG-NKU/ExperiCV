# 计算机视觉实验课-语义分割

## 实验环境搭建
实验需要使用`PyTorch`，并且在此基础上需要安装`mmcv`工具包。
在本次实验中，我们推荐使用`mmcv1.7.0`版本，可按照如下方式安装。
```shell
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cuxx/torchx.xx/index.html
```

其中cuxx和torchx.xx需要按照实际的CUDA版本和PyTorch版本进行调整，可参考
[mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

安装完毕`mmcv`后，进入实验目录，执行如下指令：
```shell
cd mmsegmentation
pip install -v -e .
```

## 数据集准备
本次实验使用数据集`ImageNet-S`进行半监督训练。
`ImageNet-S`半监督训练的训练集较小 (9190张训练集图片)，且图片的训练尺度较小 (长224，宽224)，
因此`ImageNet-S`半监督训练速度更快且消耗的显存较少。
数据集可在[ImageNet-S](https://github.com/LUSSeg/ImageNet-S#imagenet-s-dataset-preparation)下载，
需要的文件如下所示：

```
│   ├── ImageNetS
│   │   ├── ImageNetS919
│   │   │   ├── train-semi
│   │   │   ├── train-semi-segmentation
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
```

## 模型训练
基础任务的实验所需要的设置在文件`fcn_d32_r50_224x224_36k_imagenets919.py`中定义，学生可按照如下命令启动训练：
```shell
bash tools/dist_train.py \
configs/fcn/fcn_d32_r50_224x224_36k_imagenets919.py \
ngpus
```
ngpus表示显卡数量，学生可根据实际情况设置。
若显存不足，可降低配置文件中定义的`data.samples_per_gpu`。
此外，注意需要在`configs/_base_/datasets/imagenets.py`中设置数据集的路径。

## 任务一： 掌握语义分割评价指标
作业文件`mmseg/core/evaluation/metrics.py`的第26行开始是计算图片交并比的函数
`intersect_and_union`。
其中`pred_label`和`label`分别是一张图片的预测结果和真实标签。
而任务一则是补全该函数。
在讲义中我们已经说过，计算`mIoU`即平均交并比需要我们计算每一类的交并比，
而每一类的交并比需要通过`TP`、`FN`和`FP`计算。
这些值与需要补全的四个变量的关系已在如下的注释中给出，学生可以根据提示完成任务。

```python
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect = 0 # TODO
    area_pred_label = 0 # TODO
    area_label = 0 # TODO
    area_union = 0 # TODO
    # hint: use torch.histc
    # area_intersect = TP
    # area_pred_label = TP + FP
    # area_label = TP + FN
    # area_union = FN + FP + TP
```

## 任务二：FCN网络的实现与训练

文件`mmseg/models/decode_heads/fcn_head.py`定义了`FCN`网络的分割器。
学生应先阅读`FCNHead`类的父类，了解`cls_seg`的作用之后补全如下代码。

```python
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        # TODO use FCN head to process extracted features, and get segmentation results. 
        # hint: use cls_seg
        output = 0
        return output
```

## 任务三：FCN反卷积实现

实现反卷积网络可以基于任务二补全后的代码进行。
学生可以利用`PyTorch`实现的`convTranspose2d`进一步处理`output`，进行上采样。

## 探索任务

近年来，许多学者提出了不同的方法提升语义分割模型的准确性，并取得了巨大的进展。
在探索任务中，我们给出一些语义分割相关的方法，学生可自主选择单个或者数个方法进行理解与实现。
- 自适应感受野[RF-Next](https://github.com/ShangHua-Gao/RFNext)，该方法已集成到新版本的`mmcv`，学生可自主探索如何使用；
- 更强的骨干网路，如[Res2Net](https://github.com/Res2Net)、[ConvNext](https://arxiv.org/abs/2201.03545)、[Swin](https://arxiv.org/pdf/2103.14030.pdf)等；
- 更强的连接层模块，如[FPN](https://arxiv.org/abs/1612.03144)、[PAN](https://arxiv.org/abs/1803.01534)等；
- 更强的分割器，如[DeepLab](https://arxiv.org/abs/1606.00915)、[PSPNet](https://arxiv.org/abs/1612.01105)等。