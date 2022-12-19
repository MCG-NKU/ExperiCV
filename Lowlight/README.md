# 计算机视觉实验课-低光增强

## 任务一： Gamma 校正和 ZeroDCE 曲线校正算法验证
Gamma 校正和 Zero-DCE 方法均是通过曲线校正的方式提升图像亮度。请根据讲义内容，在`TODO`部分根据提示实现Gamma 曲线和 Zero-DCE曲线 。

```python
def Correction(img,gamma=2.2):
    """
    TODO: 实现图像Gamma校正代码
    """
    pass
    
def Zero_DCE(img,n=3,alpha): #img is input image,n is the iteration,alpha:[-1,1].
    """
    TODO: 根据讲义内容实现图像Zero-DCE曲线校正代码
    """
    pass
```

## 任务二：Zero-DCE 网络搭建

考虑构建一个一个由二维卷积层、池最大池化层、Bilinear 上采样层组成的深度卷积神经网络，实现估计曲线的功能，进而完成低光图像增强的任务。请根据提示内容完成模型的搭建，实现低光图像增强的网络模型。

```python
class enhance_net_nopool(nn.Module):
	def __init__(self):
		super(enhance_net_nopool, self).__init__()
		"""
		TODO: 请根据讲义提示在此处定义网络参数
		"""
		pass

	def forward(self, x):
		"""
		TODO: 请根据讲义提示在此处完成前向传播函数
		"""
		pass
	
```

## 任务三：损失函数补全

通过多种损失函数来约束低光图像增强的结果，其中包含空间一致性损失、曝光控制损失、颜色恒常 损失，同时通过 Adam 优化器进行模型优化。请根据损失函数公式和提示内容实现曝光控制损失和颜色恒常损失。

```python
class L_exp(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        """
        TODO: 定义均值函数，用来获得图像块的像素均值
        self.pool = ???
        """
        self.mean_val = mean_val
    def forward(self, x ):
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        """
        TODO: 根据公式补全曝光控制损失
        exp = ??? 
        """
        pass

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x ):
        """
        TODO: 计算图像像素均值;将像素均值拆分为RGB三部分;计算两两通道之间的方差
        mean_rgb = ???
        mr,mg, mb = ???
        Drg = ???, Drb = ???, Dgb = ???
        """
        color = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return color
```

## 探索任务

请学有余力的读者阅读论文，探索实现利用更轻量级的深度学习网络较好地实现低光图像增强任务，可以参考一下文献进行阅读和尝试。可以从以下角度出发改进模型：从网络结构的角度如何减少计算量，从空间信息冗余的角度如何减少计算量，从定义增强曲线形式的角度如何减少计算量。可参考文献 [[1](https://arxiv.org/pdf/1704.04861.pdf), [2](https://arxiv.org/pdf/2103.00860.pdf), [3](https://arxiv.org/pdf/2207.14273.pdf), [4](https://arxiv.org/pdf/2012.05609.pdf)]。

