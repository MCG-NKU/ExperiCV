# 计算机视觉实验参考代码-显著性物体检测
本实验旨在定义一个基于卷积神经网络的显著性物体检测模型，并在该任务的公开数据集[DUTS](https://github.com/LUSSeg/ImageNet-S#imagenet-s-dataset-preparation)上进行训练与测试。

## 任务一： 数据集加载实验
数据是人工智能研究的三大核心要素（算法、算力、数据）之一，本任务要求读者基于Pytorch框架提供的Dataset类，自定义一个数据集类SalObjData来加载显著性物体检测任务的公开数据集DUTS。该数据集的组织方式如下：

```
│   ├── DUTS
│   │   ├── Train
│   │   │   ├── Image
│   │   │   ├── GT
│   │   │   ├── train.txt
│   │   ├── Test
│   │   │   ├── Image
│   │   │   ├── GT
│   │   │   ├── test.txt
```
其中，Image和GT分别为存放原始图像和标注图像的文件夹。    

SalObjData类定义完成以后，使用如下代码测试该类能否完成数据集加载的功能。
```python
    # 训练集
    train_data = SalObjData(data_root=data_root, mode='train', image_size=256)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    train_images, train_labels = iter(train_loader).next()
    print('train_images: ', train_images.shape)
    print('train_labels: ', train_labels.shape)

    # 测试集 
    test_data = SalObjData(data_root=data_root, mode='test', image_size=256)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    test_images, test_labels = iter(test_loader).next()
    print('test_images: ', test_images.shape)
    print('test_labels: ', test_labels.shape)
```
具体的代码实现请参考```task1.py```。

## 任务二：显著性物体检测模型的搭建
模型/算法是人工智能研究的另一个关键要素，本任务要求读者基于Pytorch框架提供的Module类，搭建[DSS](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hou_Deeply_Supervised_Salient_CVPR_2017_paper.pdf)网络。

搭建DSS网络之前，读者可以参考dss.txt中保存的结构；搭建DSS网络之后，读者首先需要打印出网络结构与dss.txt中的结构对照，然后使用以下代码测试DSS网络的输入和输出是否符合期望。
```python
import torch
x = torch.randn(4, 3, 256, 256)
print('x:', x.shape)
model = DSS()
# print(model)

out_list = model(x)
print('out:')
for out in out_list:
    print(out.shape)
```
具体的代码实现请参考```task2.py```。

## 任务三：显著性物体检测模型的训练和测试
为了获得性能优异的参数，需要对任务二中搭建的DSS网络进行训练，请读者参考论文[F3Net](https://arxiv.org/pdf/1911.11445.pdf)中由加权的二元交叉熵损失和交并比损失组成的损失函数。DSS模型训练完成以后，需要对其性能进行测试，本任务选择的评价指标是[Sm](https://openaccess.thecvf.com/content_ICCV_2017/papers/Fan_Structure-Measure_A_New_ICCV_2017_paper.pdf)，读者需要根据讲义中的提示完成Smeasure类的定义。并测试任务二上训练的DSS网络的Sm得分。  

具体的代码实现请参考```task3.py```。

## 探索任务：显著性物体检测任务的类别无关性探索
类别无关性是显著性物体检测任务的重要特性之一，Cheng等人在[CSNet](https://mftp.mmcheng.net/Papers/21PAMI-Sal100K.pdf)对这一特征进行了详细的分析。学有余力的读者可以阅读该论文，并参考开源代码[SOD100K](https://github.com/ShangHua-Gao/SOD100K/)，测试在分类的 ECSSD 数据集有无“arthropod”类别样本的两种情况下训练的CSNet的性能，并分析原因。
