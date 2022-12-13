# 计算机视觉实验课-图像分割
## 实验环境搭建
本次分割实验使用python3完成。推荐使用python3.9，实验过程中需要安装的库函数如下：

```bash
skimage
numpy
matplotlib
argparse
logging
time
graph
PIL
```

可直接通过`pip install -r requirements.txt`安装。

## 任务一：阈值法分割实验
本次实验较为简单，在`hw1/hw1.ipynb`文件中，给出了逐步完成实验的提示，读者可直接运行这个notebook文件，也可以把代码部分提取到py文件运行。

## 任务二：图论法分割实验
本次实验基于论文[Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)实现的，读者可先阅读论文，结合论文中的方法步骤阅读代码，进行补全。

### 根据输入图片搭建图（Graph）
在文件/hw2/graph.py中补全build_graph函数。
这个函数接受输入的图片`img `, 图片的宽和高`width height`， `diff`是一个在`/hw2/main.py`中定义的判别像素差的距离函数，`neighborhood_8`代表是使用八距离还是四距离，默认只使用四距离。

### 对满足的条件的区域块进行合并
在文件/hw2/graph.py中补全segment_graph函数。