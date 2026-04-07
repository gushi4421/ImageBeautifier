"""
本模块实现图片几何变换相关的算法
采用数学方法实现,也提供OpenCV的实例代码

实现以下几种几何变换:
1. 平移 translate
2. 旋转 rotate
3. 缩放 zoom
4. 翻转 flip
5. 剪切 shear
6. 仿射变换 affine_transform(非opencv分支不提供)
"""

import numpy as np
from pathlib import Path
import cv2 as cv
from typing import Callable
import math


# 1. 平移
def translate(image: np.ndarray, tx, ty) -> np.ndarray:
    h, w, _ = image.shape
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
    h, w, _ = image.shape
    center_x = w / 2.0
    center_y = h / 2.0
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    target_image = np.zeros_like(image)

    for u in range(w):
        for v in range(h):
            # 计算该点伊中心点为原点的坐标
            u_centered = u - center_x
            v_centered = v - center_y
            # 计算旋转后的坐标
            x_centered = u_centered * cos_theta + v_centered * sin_theta
            y_centered = -u_centered * sin_theta + v_centered * cos_theta

            x = x_centered + center_x
            y = y_centered + center_y

            x = int(round(x))
            y = int(round(y))
            if 0 <= x < w and 0 <= y < h:
                target_image[v, u] = image[y, x]
    return target_image


# 3. 缩放
def zoom(image: np.ndarray, x_scale, y_scale):
    h, w, c = image.shape
    new_w = int(w * x_scale)
    new_h = int(h * y_scale)
    target_image = np.zeros((new_h, new_w, c), np.uint8)
    for u in range(new_w):
        for v in range(new_h):
            src_x = (u + 0.5) / x_scale - 0.5
            src_y = (v + 0.5) / y_scale - 0.5
            dx = src_x - math.floor(src_x)
            dy = src_y - math.floor(src_y)
            x1 = int(math.floor(src_x))
            y1 = int(math.floor(src_y))

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)

            p11 = image[y1, x1].astype(np.float32)
            p12 = image[y2, x1].astype(np.float32)
            p21 = image[y1, x2].astype(np.float32)
            p22 = image[y2, x2].astype(np.float32)

            r1 = p11 * (1 - dx) + p21 * dx
            r2 = p12 * (1 - dx) + p22 * dx

            p = (1 - dy) * r1 + dy * r2
            target_image[v, u] = np.clip(p, 0, 255).astype(np.uint8)

    return target_image


# 4. 翻转
def flip(image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    target_image = np.zeros_like(image)

    for u in range(w):
        for v in range(h):
            target_image[v, w - u - 1] = image[v, u]
    return target_image


# 5. 剪切
def shear(image: np.ndarray, start_x, end_x, start_y, end_y):
    h, w, c = image.shape
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w, end_x)
    end_y = min(h, end_y)
    target_image = np.zeros((end_y - start_y, end_x - start_x, c), np.uint8)
    for u in range(start_x, end_x):
        for v in range(start_y, end_y):
            target_image[v - start_y, u - start_x] = image[v, u]

    return target_image


# 6. 仿射变换
def affine_transform(image: np.ndarray):
    pass
