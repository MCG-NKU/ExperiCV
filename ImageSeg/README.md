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
```python
def build_graph(img, width, height, diff, neighborhood_8=False):
    ## 
    ## write your code here
    ## 根据图片搭建图
    ##
    return graph_edges
```

### 对满足的条件的区域块进行合并
在文件/hw2/graph.py中补全segment_graph函数。此时函数已经以像素为最小单位建立了图结构，需要做的就是根据论文中提出的合并算法，不断地对图进行合并，得到最终分割结果。
```python
def segment_graph(graph_edges, num_nodes, const, min_size, threshold_func):
    # Step 1: initialization
    forest = Forest(num_nodes)
    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph_edges, key=weight)
    threshold = [ threshold_func(1, const) for _ in range(num_nodes) ]

    # Step 2: merging
    ## 
    ## write your code here
    ## 对满足条件的区域块进行合并
    ##

    return remove_small_components(forest, sorted_graph, min_size)


```

## 深度学习方法分割实验
请同学们阅读论文DEL: Deep Embedding Learning for Efficient Image Segmentation，根据提示实现基于深度学习的分割方法。

BSDS训练数据集：[here](https://github.com/yun-liu/del#:~:text=can%20be%20found-,here,-.%20In%20the%20provided)

### 补全超像素pooling
在`DEL.py`中`superpixel_pooling中接受深度学习的输出，groundtruth标签，超像素标签，返回论文中提到的，每一个区域块的特征。

### 补全loss代码
按照论文中对loss的描述，补全代码，为每一张照片计算损失函数
```python
def sim_loss(superpixel_pooling_out, superpixel_seg_label, sp_label):
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]
    height = sp_label.shape[1]
    width = sp_label.shape[2]
    total_loss = 0
    ##
    ## write your sim loss here
    ##
   
    return total_loss
```

### 补全测试代码
DEL中测试过程相较训练有些许区别，需要读者根据论文，补全测试部分，至此完整地复现DEL代码。