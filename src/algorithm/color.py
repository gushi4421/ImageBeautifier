"""
该模块实现图像色彩相关的操作.
包含灰度化、二值化(含自动阈值大津法)以及伪彩色映射映射功能.
"""

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    将 BGR 图像转为单通道灰度图.

    Args:
        image: 输入的图像数据, 类型为 np.ndarray, 形状通常为 (H, W, 3).

    Returns:
        转换后的单通道灰度图像, 形状为 (H, W), 数据类型为 uint8.
    """
    # 获取图像尺寸,虽然此处未直接使用 h, w, 但常用于预处理检查
    h, w, _ = image.shape

    # 定义加权权重,对应 OpenCV 的 BGR 顺序权重
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)

    # 通过矩阵点积快速计算全图灰度值
    gray_image = np.dot(image, weights)

    # 确保数值在 0-255 之间并转换数据类型
    return np.clip(gray_image, 0, 255).astype(np.uint8)


def binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    对图像进行简单固定阈值二值化处理.

    Args:
        image: 输入的灰度图像.
        threshold: 设定的分割阈值, 默认为 127.

    Returns:
        二值化后的图像, 像素值仅包含 0 和 255.
    """
    binary_image = np.zeros_like(image)
    # 大于阈值的像素点设置为白色(255)
    binary_image[image > threshold] = 255
    return binary_image


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """
    OTSU自动阈值二值化算法实现.

    Args:
        image: 输入的待处理灰度图像.

    Returns:
        使用自动阈值分割后的二值图像.
    """
    total_pixels = image.size
    histogram = np.bincount(image.ravel(), minlength=256)
    probabilities = histogram / total_pixels
    intensity_levels = np.arange(256)
    sum_mean = np.sum(probabilities * intensity_levels)

    background_weight = 0.0
    background_sum = 0.0
    best_threshold = 0
    max_variance = 0.0

    for t in range(256):
        background_weight += probabilities[t]
        if background_weight == 0.0:
            continue
        if background_weight >= 1.0:
            break

        background_sum += probabilities[t] * intensity_levels[t]
        background_mean = background_sum / background_weight
        foreground_weight = 1 - background_weight
        foreground_mean = (sum_mean - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    binary_image = np.zeros_like(image)
    binary_image[image > best_threshold] = 255
    return binary_image


def generate_colormap_lut():
    """
    生成伪彩色查找表(Look-Up Table)。
    将 0-255 映射到蓝色到红色的渐变区间。
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


def apply_colormap(image: np.ndarray):
    """
    应用查找表进行图像伪彩色映射。
    """
    lut = generate_colormap_lut()
    color_image = lut[image]
    return color_image
