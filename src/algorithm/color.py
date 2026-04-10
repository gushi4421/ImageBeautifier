"""
该模块实现图像色彩相关的操作.
包含灰度化、二值化(含自动阈值大津法)以及伪彩色映射映射功能.
"""

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    将 BGR 图像转为单通道灰度图.

    使用心理学加权公式: Gray = R*0.299 + G*0.587 + B*0.114.
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
    """
    binary_image = np.zeros_like(image)
    # 大于阈值的像素点设置为白色(255)
    binary_image[image > threshold] = 255
    return binary_image


def big_jin_fa(image: np.ndarray) -> np.ndarray:
    """
    大津法(OTSU)自动阈值二值化算法实现.

    通过最大化类间方差来寻找最佳分割阈值.
    """
    total_pixels = image.size
    # 统计 0-255 灰度直方图
    histogram = np.bincount(image.ravel(), minlength=256)
    # 计算每个灰度级出现的概率
    probabilities = histogram / total_pixels
    intensity_levels = np.arange(256)
    # 计算图像的全局平均灰度值
    sum_mean = np.sum(probabilities * intensity_levels)

    background_weight = 0.0  # 背景像素点占整幅图的比例
    background_sum = 0.0  # 背景像素的累计灰度期望值
    best_threshold = 0  # 最终确定的最佳阈值
    max_variance = 0.0  # 最大类间方差

    for t in range(256):
        # 累加背景权重(w0)
        background_weight += probabilities[t]
        if background_weight == 0.0:
            continue
        if background_weight >= 1.0:
            break

        # 累加背景部分的灰度值期望
        background_sum += probabilities[t] * intensity_levels[t]

        # 计算背景平均灰度 (u0)
        background_mean = background_sum / background_weight

        # 计算前景权重 (w1)
        foreground_weight = 1 - background_weight
        # 计算前景平均灰度 (u1)
        foreground_mean = (sum_mean - background_sum) / foreground_weight

        # 计算类间方差公式: sigma^2 = w0 * w1 * (u0 - u1)^2
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        # 更新最大方差及其对应的阈值
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    # 根据最佳阈值生成二值图
    binary_image = np.zeros_like(image)
    binary_image[image > best_threshold] = 255
    return binary_image


def generate_lut():
    """
    生成伪彩色查找表(Look-Up Table).
    将 0-255 映射到蓝色到红色的渐变区间.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i <= 85:
            # 第一阶段: 增加红色通道
            lut[i, 2] = i * 3
            lut[i, 1] = 0
            lut[i, 0] = 0
        elif i <= 170:
            # 第二阶段: 红色保持, 增加绿色通道
            lut[i, 2] = 255
            lut[i, 1] = (i - 85) * 3
            lut[i, 0] = 0
        elif i <= 255:
            # 第三阶段: 红绿保持, 增加蓝色通道
            lut[i, 2] = 255
            lut[i, 1] = 255
            lut[i, 0] = (i - 170) * 3
    return np.clip(lut, 0, 255).astype(np.uint8)


def colormap(image: np.ndarray):
    """
    应用查找表进行图像伪彩色映射.
    """
    lut = generate_lut()
    # 利用 NumPy 的索引特性实现快速像素值映射
    color_image = lut[image]
    return color_image
