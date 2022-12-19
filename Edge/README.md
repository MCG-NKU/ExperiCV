# 计算机视觉实验课-边缘检测

## 实验环境搭建

实验需要使用`PyTorch`，以及一系列第三方库。具体库文件及其版本在requirements.txt中，可通过以下指令进行安装。

```bash
pip install -r requirements.txt
```

## 任务一： Laplacian 算法验证
建议使用 Python 环境下的 opencv 库进行实现。测试图片不同场景，不同复杂度的样例方便观察算法的优势
和不足。使用 cv2.Laplacian 提取出边缘图像后，可借助 cv2.imshow 或 cv2.imwrite 可视化其结果。注意理解算法
中的参数含义，并对比不同参数设置下的算法结果。补充hw1&2/main.py中相关代码部分。


## 任务二：Canny 算法验证

代码环境同上使用 cv2.Canny 实现算法过程，可以尝试不同超参数。建议用进度条控制滞回比较器的两个阈
值 threshold1, threshold2，以及控制尺度的 sigma 参数。尝试不同场景图片，比较不同参数情况下的边缘检测结
果。补充hw1&2/main.py中相关代码部分。
对于学有余力的同学，可以尝试自己去写整个 Canny 算法，包括：利用高斯函数偏导数对应的卷积核估计
梯度值及方向信息，沿着梯度方向进行非极大值抑制，在边缘点链接过程中利用滞回比较器增强检测结果的鲁
棒性和连续性等。

## 任务三：HED 算法关键步骤补全实验

在https://drive.google.com/file/d/1iccHFnX-tJyqiVUj3UroNYvaJehs2yeE/view?usp=share_link下载预训练好的backbone文件vgg16.pth，放到hw3文件夹目录下。
补充hw3项目中train.py，models/hed.py代码不完整的部分。
运行train.py训练模型，默认将模型输出保存在tmp文件夹下，模型保存在checkpoints文件夹下。
训练完成后运行test.py，测试结果会保存在test文件夹下。


## 探索任务

建议学有余力的同学探索如何进一步改进边缘检测算法的性能。可以尝试不同骨干网络，如 ResNet、
Res2Net、P2T 等，比较这些特征提取模型给算法性能带来的提升。

