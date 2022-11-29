import numpy as np
 
def bilinear_interpolate(src, dst_shape):
    src_height, src_width, src_channel = src.shape
    dst_height, dst_width = dst_shape

    # 坐标投影
    ls_w = [(i + 0.5) / dst_width * src_width - 0.5 for i in range(dst_width)]
    ls_h = [(i + 0.5) / dst_height * src_height - 0.5 for i in range(dst_height)]
    ls_w = np.clip(np.array(ls_w, dtype=np.float32), 0, src_width - 1)
    ls_h = np.clip(np.array(ls_h, dtype=np.float32), 0, src_height - 1)
    
    # 找到特定位置最近点
    ref_wl = np.clip(np.floor(ls_w), 0, src_width - 2).astype(int)
    ref_hl = np.clip(np.floor(ls_h), 0, src_height - 2).astype(int)
    
    # 根据公式进行双线性插值
    dst = np.zeros(shape=(dst_height, dst_width, src_channel), dtype=np.float32)
    """
    TODO: 根据公式进行双线性插值
    """
    return dst

if __name__ == '__main__':
    src = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 9, 27, 81], [4, 16, 64, 256]])
    src = np.expand_dims(src, axis=2)
    dst = bilinear_interpolate(src, dst_shape=(5, 5))
    print('src shape:', src.shape, '\ndst shape:', dst.shape)
    print(np.squeeze(dst))
