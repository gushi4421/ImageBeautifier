"""
该模块实现对图像的增强处理.
包含对比度调节、亮度调节以及基于直方图均衡化的图像增强功能.
"""

import numpy as np


def adjust_contrast(image: np.ndarray, alpha: float = 1.0):
    """
    调整图像对比度.

    通过乘法因子 alpha 缩放像素值, 使亮部更亮, 暗部更暗.
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

    通过加法因子 beta 整体提升或降低像素值.
    """
    adjusted_image = image.astype(np.float32)
    # 线性变换: f(x) = x + beta
    adjusted_image = adjusted_image + beta
    # 确保数值合法性
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def histogram(image: np.ndarray) -> np.ndarray:
    """
    直方图均衡化实现.

    利用累积分布函数(CDF)作为映射函数, 将原图的灰度分布转化为均匀分布.
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
