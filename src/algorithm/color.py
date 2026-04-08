"""
该模块实现图像色彩相关的操作
"""

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    将 BGR 图像转为单通道灰度图
    """
    h, w, _ = image.shape

    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    gray_image = np.dot(image, weights)
    return np.clip(gray_image, 0, 255).astype(np.uint8)


def binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    return binary_image


def big_jin_fa(image: np.ndarray) -> np.ndarray:
    total_pixels = image.size
    histogram = np.bincount(image.ravel(), minlength=256)
    probabilities = histogram / total_pixels
    intensity_levels = np.arange(256)
    sum_mean = np.sum(probabilities * intensity_levels)

    background_weight = 0.0  # 北京像素点所占权重
    background_sum = 0  # 总像素点像素灰度值乘对应概率之和
    best_threshold = 0
    max_variance = 0.0

    for t in range(256):
        background_weight += probabilities[t]
        if background_weight == 0.0:
            continue
        if background_weight == 1.0:
            break
        background_sum += probabilities[t] * intensity_levels[t]
        background_mean = background_sum / background_weight
        foreground_weight = 1 - background_weight
        foreground_mean = (
            sum_mean - background_mean * background_weight
        ) / foreground_weight
        vraiance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if vraiance > max_variance:
            max_variance = vraiance
            best_threshold = t
    binary_image = np.zeros_like(image)
    binary_image[image > best_threshold] = 255
    return binary_image


def generate_lut():
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


def colormap(image: np.ndarray):
    lut = generate_lut()
    color_image = lut[image]
    return color_image
