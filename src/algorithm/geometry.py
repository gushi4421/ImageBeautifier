"""
本模块实现图片几何变换相关的算法
采用数学方法实现,也提供OpenCV的实例代码

实现以下几种几何变换:
1. 平移 translate
2. 旋转 rotate
3. 缩放 zoom
4. 翻转 flip
5. 剪切 shear
6. 仿射变换 affine_transform
"""

import numpy as np
from pathlib import Path
import cv2 as cv
from typing import Callable


# 1. 平移
def translate(image: np.ndarray, tx, ty) -> np.ndarray:
    h, w, c = image.shape
    target_image = np.zeros_like(image)
    for u in range(w):
        for v in range(h):
            x = u - tx
            y = v - ty
            if 0 <= x < w and 0 <= y < h:
                target_image[v, u] = image[y, x]
    return target_image


# 2. 旋转
def rotate(image: np.ndarray, angle):
    pass


# 3. 缩放
def zoom(image: np.ndarray):
    pass


# 4. 翻转
def flip(image: np.ndarray):
    pass


# 5. 剪切
def shear(image: np.ndarray):
    pass


# 6. 仿射变换
def affine_transform(image: np.ndarray):
    pass
