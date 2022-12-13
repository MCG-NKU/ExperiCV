# 计算机视觉实验课-全景分割

## 实验环境搭建

实验需要使用`PyTorch`，并且在此基础上需要安装`RF-mmcv`工具包。`RF-mmcv`安装文件夹直接在当前项目页面下载。下载完毕后进入`RF-mmcv`文件夹，安装`ninja`与`psutil`用于加速编译：

```bash
pip install -r requirements/optional.txt
```

检查`nvcc`版本（需要大于9.2）以及`gcc`版本（需要大于5.4）后开始安装：

```
MMCV_WITH_OPS=1 pip install -e . -v
```

之后安装`RF-mmdetection`工具包，安装文件夹同样直接在当前项目页面下载。下载完毕后进入`RF-mmcv`文件夹，输入如下命令进行安装：

```bash
pip install -v -e .
```

最后还需要安装`panopticapi`：

```bash
pip install git+https://github.com/cocodataset/panopticapi.git  
```

## 数据集准备

本次实验使用的数据集来自`COCO`，从其中选取验证集的10张图片及其标注数据得到实验中用于测试的数据集`coco_tiny`。数据集可在当前项目页面下载，数据集结构如下所示：

```
│   ├── coco_tiny
│   │   ├── annotations
│   │   │   ├── panoptic_train2017_tiny
│   │   │   ├── panoptic_val2017_tiny
│   │   │   ├── panoptic_train2017_tiny.json
│   │   │   ├── panoptic_val2017_tiny.json
│   │   ├── train2017_tiny
│   │   ├── val2017_tiny
│   │   ├── test2017_tiny
```

## 任务一

进入RF-mmdetection下的mmdet/models/seg\_heads文件夹，查看panoptic\_fpn\_head.py。该文件中的PanopticFPNHead类共包含五个函数，请仔细阅读注释理解变量含义，并根据提示对thing类别置空函数\_set\_things\_to\_void以及前向传播函数forward进行代码补全。

```python
def _set_things_to_void(self, gt_semantic_seg):
    """Merge thing classes to one class.
    In PanopticFPN, the background labels will be reset from `0` to
    `self.num_stuff_classes-1`, the foreground labels will be merged to
    `self.num_stuff_classes`-th channel.
    """
    gt_semantic_seg = gt_semantic_seg.int()
    fg_mask = gt_semantic_seg < self.num_things_classes
    #参考计算前景mask的方式计算背景mask。
    #背景像素需要满足该像素对应的类别标号大于等于num_things_classes-1
    #且小于num_things_classes+num_stuff_classes。
    bg_mask = ???
    new_gt_seg = torch.clone(gt_semantic_seg)
    new_gt_seg = torch.where(bg_mask,
                             gt_semantic_seg - self.num_things_classes,
                             new_gt_seg)
    new_gt_seg = torch.where(fg_mask,
                             fg_mask.int() * self.num_stuff_classes,
                             new_gt_seg)
    return new_gt_seg
```

## 任务二

进入RF-mmdetection下的mmdet/models/seg\_heads/panoptic\_fusion\_heads文件夹，
打开heuristic\_fusion\_head.py。该文件中的HeuristicFusionHead类实现了推理时对于实例分割前景与语义分割背景的融合。请仔细阅读注释理解变量含义，并根据提示对\_lay\_masks函数进行代码补全。

```python
for idx in range(bboxes.shape[0]):
  _cls = labels[idx]
  _mask = segm_masks[idx]
  instance_id_map = torch.ones_like(
    _mask, dtype=torch.long) * instance_id
  area = _mask.sum()
  if area == 0:
    continue
    pasted = id_map > 0
    #计算当前区域(_mask)与已有id区域(pasted)的交集，并对交集像素数量求和
    intersect = ???
    if (intersect / (area + 1e-5)) > overlap_thr:
      continue
      #通过_mask和pasted计算属于当前实例的区域(即_mask为True，pasted为False的区域) 
      _part = ???
      id_map = torch.where(_part, instance_id_map, id_map)
      left_labels.append(_cls)
      instance_id += 1
```

## 任务三

本次实验使用三种不同的网络`ResNet`、`Res2Net`以及`RF-Next+Res2Net`分别对10张图片进行测试。在将代码进行补全后下载三个网络的模型参数，分别运行以生成相应的预测结果。查看结果，分析不同的骨干网络结构对于模型整体性能的影响。

## 探索任务

在完成以上内容后，本章节的开放性探索实验是实现连续学习的语义分割理解和尝试，实验内容借助[RC-IL](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Representation_Compensation_Networks_for_Continual_Semantic_Segmentation_CVPR_2022_paper.html)工作中的模型训练结果进行测试和对比，推荐在完成实验前阅读相应的论文。

实验使用的框架为CVPR2021_PLOP，直接在当前项目页面下载，环境配置请参考讲义相关内容，实验需要使用的模型参数文件有两个，分别为[compare.pth](https://drive.google.com/file/d/1D1nJWWx7jzSh1YAzj55So78mVojj2FH8/view?usp=share_link)和[test.pth](https://drive.google.com/file/d/1W_kOK1quJjePOH2IfI8rF7oU6GNoSuZa/view?usp=share_link)。将下载好的参数文件存放于CVPR2021\_PLOP文件夹下即可。

之后使用编辑器打开CVPR2021_PLOP下的`run.py`文件，参考`mergex()`函数，针对内部的`run.py`中的相应函数`merge()`填入相应的代码。

```python
def merge(conv, bias, bn, bn_bias, running_means, running_vars):
    TODO()
```

填补完成后，在代码目录下执行：

```shell
sh scripts/voc/run.sh
```

之后就可以在代码目录下的./results中看到预测图像的差异，ours和compare分别代表我们训练得到的最终模型所预测的结果与训练了一个阶段后进行直接预测的结果。
