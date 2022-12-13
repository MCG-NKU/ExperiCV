## 环境搭建

pip install -r requirements.txt

## 加载并预览图片

```python
from skimage import io,img_as_ubyte
# 加载示例中草莓的灰度图片
img = ?
plt.imshow(img,cmap="gray")
plt.xlabel("origin photo")
```

## 画出图片灰度直方图

```python
plt.?
plt.xlabel("histogram")

```

## 手动选择阈值，进行分割
```python
## set your threshold here
threhold = ?
## 
segm = ?
plt.imshow(segm,cmap="gray")
plt.xlabel("segmentation by hand")
```

## 实现OTSU方法

```python
def OTSU(img_array):
	height = img_array.shape[0]
	width = img_array.shape[1]
	count_pixel = np.zeros(256)
 
	for i in range(height):
		for j in range(width):
			count_pixel[int(img_array[i][j])] += 1 
 
	max_variance = 0.0
	best_thresold = 0
	## 
    ## write your code here
    ##
	return best_thresold


otsu_the = OTSU(img)
print("the OTSU therohold is : ", otsu_the)

segm = ?
plt.imshow(segm,cmap="gray")
plt.xlabel("segmentation by OTSU")

```

## 实现HSV颜色分割
首先定义想要分割的颜色范围，可先通过HSV官方查找，然后手动测试调整

```python
from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv


class Mask(ABC):
    @classmethod
    @abstractmethod
    def mask(cls, hsv_image):

class Red(Mask):
    low_1 = np.array([0, 125, 20])
    high_1 = np.array([10, 255, 255])
    low_2 = np.array([170, 125, 20])
    high_2 = np.array([180, 255, 255])

    @classmethod
    def mask(cls, hsv_image):
        mask_1 = cv.inRange(hsv_image, cls.low_1, cls.high_1)
        mask_2 = cv.inRange(hsv_image, cls.low_2, cls.high_2)
        return mask_1 + mask_2


class Green(Mask):
    low = np.array([17, 79, 19])
    high = np.array([76, 217, 153])

    @classmethod
    def mask(cls, hsv_image):
        return cv.inRange(hsv_image, cls.low, cls.high)
```

定义分割的函数：

```python
def segment(image_path: str, mask: Mask):

    ##
    ## write your code here
    ##
    ## 函数接受RGB图片地址，返回用某个颜色分割后的结果
    return segmented_image
```

最后，调用函数，得到最终HSV颜色分割的结果：
```python
image_path = "strawbarry.png"
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(131)
ax.set_title("original")
ax.axis("off")
image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
_ = ax.imshow(image, interpolation="nearest")

ax = fig.add_subplot(132)
ax.set_title("red mask")
ax.axis("off")
image = segment(image_path, Red)
_ = ax.imshow(image, interpolation="nearest")

ax = fig.add_subplot(133)
ax.set_title("green mask")
ax.axis("off")
image = segment(image_path, Green)
_ = ax.imshow(image, interpolation="nearest")
```