"""
本模块实现图片几何变换相关的算法
采用数学方法实现,也提供OpenCV的实例代码

实现以下几种几何变换:
1. 平移
2. 旋转
3. 缩放
4. 翻转
5. 剪切
6. 仿射变换
"""

import numpy as np
from pathlib import Path
import cv2 as cv
from typing import Callable


# 1. 平移
def translate(image: np.ndarray, x, y):
    pass


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
