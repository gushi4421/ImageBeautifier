"""
该模块提供算法内部使用的工具函数.
包含高斯卷积核生成、伪彩色查找表生成以及图像格式统一转换.
"""

from __future__ import annotations

import math
from typing import Any

import cv2 as cv
import numpy as np
from PIL import Image


def generate_gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    生成高斯模糊的卷积核.

    Args:
        kernel_size: 卷积核的大小, 必须是大于 0 的奇数.
        sigma: 高斯分布的标准差, 控制模糊的平滑度.

    Returns:
        返回形状为 (kernel_size, kernel_size) 的二维浮点矩阵, 且和为 1.
    """
    if kernel_size % 2 == 0 or kernel_size < 0:
        raise ValueError()
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    radius = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - radius
            y = j - radius
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def generate_colormap_lut() -> np.ndarray:
    """
    生成伪彩色查找表(Look-Up Table).
    将 0-255 映射到蓝色到红色的渐变区间.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i <= 85:
            lut[i, 2] = i * 3
            lut[i, 1] = 0
            lut[i, 0] = 0
        elif i <= 170:
            lut[i, 2] = 255
            lut[i, 1] = (i - 85) * 3
            lut[i, 0] = 0
        elif i <= 255:
            lut[i, 2] = 255
            lut[i, 1] = 255
            lut[i, 0] = (i - 170) * 3
    return np.clip(lut, 0, 255).astype(np.uint8)


def to_bgr_numpy(image: Any) -> np.ndarray:
    """
    将任意输入的图片对象转换为标准的 3 通道 BGR Numpy 数组.
    包含对 RGBA 透明通道的白色背景融合处理.

    Args:
        image: PIL Image 对象或 Numpy 数组.

    Returns:
        标准的 BGR 格式 Numpy 数组.
    """
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            rgb_image = np.asarray(background, dtype=np.uint8)
        else:
            rgb_image = image.convert("RGB")
            rgb_image = np.asarray(rgb_image, dtype=np.uint8)

        return cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise TypeError("图片必须是 PIL Image 或 numpy.ndarray 类型.")

    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "图片数组必须是形状为 (H, W), (H, W, 1), (H, W, 3) 或 (H, W, 4) 的数组."
        )

    return image.copy()
