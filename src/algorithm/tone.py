"""
该模块实现像素级点操作.
包含灰度化、二值化(含OTSU)、伪彩色映射、对比度调节、亮度调节以及直方图均衡化.
"""

import numpy as np

from src.utils import generate_colormap_lut


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


def apply_colormap(image: np.ndarray):
    """
    应用查找表进行图像伪彩色映射。
    """
    lut = generate_colormap_lut()
    color_image = lut[image]
    return color_image


def adjust_contrast(image: np.ndarray, alpha: float = 1.0):
    """
    调整图像对比度.

    Args:
        image: 输入图像.
        alpha: 对比度缩放因子, 大于 1.0 增强对比度, 小于 1.0 降低对比度.

    Returns:
        调整对比度后的图像.
    """
    # 转为浮点型防止计算溢出
    adjusted_image = image.astype(np.float32)
    # 线性变换: f(x) = alpha * x
    adjusted_image = adjusted_image * alpha
    # 截断处理并转回 uint8 类型
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def adjust_brightness(image: np.ndarray, beta: int = 0):
    """
    调整图像亮度.
    Args:
        image: 输入图像.
        beta: 亮度偏移量, 正值增加亮度, 负值降低亮度.

    Returns:
        调整亮度后的图像.
    """
    adjusted_image = image.astype(np.float32)
    # 线性变换: f(x) = x + beta
    adjusted_image = adjusted_image + beta
    # 确保数值合法性
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def histogram(image: np.ndarray) -> np.ndarray:
    """
    直方图均衡化实现.

    Args:
        image: 输入的低对比度灰度图像.

    Returns:
        亮度分布均匀化后的增强图像.
    """
    total_pixels = image.size
    # 统计每个灰度级(0-255)出现的频次
    hist_counts = np.bincount(image.ravel(), minlength=256)

    # 计算概率密度函数 (PDF: Probability Density Function)
    pdf = hist_counts / total_pixels

    # 计算累积分布函数 (CDF: Cumulative Distribution Function)
    cdf = np.cumsum(pdf)

    # 生成查找表 (LUT): 将 CDF 映射到 0-255 区间并取整
    lut = np.round(cdf * 255).astype(np.uint8)

    # 根据查找表重映射图像像素值
    color_image = lut[image]
    return color_image
