"""
该模块实现图像加噪处理.
支持高斯噪声与椒盐噪声的生成.
"""

import cv2 as cv
import numpy as np


def add_noise(image: np.ndarray, mode: str = "gaussian", **kwargs) -> np.ndarray:
    """
    为图像添加特定类型的噪声.

    Args:
        image: 输入的原始图像.
        mode: 噪声模式, 支持 "gaussian" 和 "salt_pepper".
        kwargs: 噪声参数 (mean, sigma 或 prob).
    Returns:
        去噪平滑后的图像.
    """
    support_types = ["gaussian", "salt_pepper"]
    if mode not in support_types:
        raise ValueError("不支持的噪声类型")

    if mode == "gaussian":
        # 获取高斯分布参数, 默认为均值0, 标准差20
        mean = kwargs.get("mean", 0.0)
        sigma = kwargs.get("sigma", 20.0)
        noise = np.empty(image.shape, dtype=np.float32)
        cv.randn(noise, mean, sigma)
        noisy_image = cv.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(image.dtype)

    elif mode == "salt_pepper":
        prob = kwargs.get("prob", 0.1)
        if prob > 1.0:
            raise ValueError("噪声比例不能大于1.0")

        noise_image = image.copy()
        h, w, _ = image.shape
        random_matrix = np.empty((h, w), dtype=np.float32)
        cv.randu(random_matrix, 0.0, 1.0)

        # 比例的一半设为椒噪声(黑色 0)
        noise_image[random_matrix < (prob / 2.0)] = [0, 0, 0]
        # 比例的另一半设为盐噪声(白色 255)
        # 修正原代码逻辑符号错误: 使用 = 赋值而非 - 运算
        noise_image[(random_matrix >= (prob / 2.0)) & (random_matrix < prob)] = [
            255,
            255,
            255,
        ]

        return noise_image


def add_salt_pepper_noise_optimized(image: np.ndarray, prob: float):
    """
    优化版的椒盐噪声生成。

    Args:
        image: 原始图像。
        prob: 噪声出现的概率, 范围在 [0.0, 1.0] 之间。

    Returns:
        添加了椒盐噪声的图像副本。
    """
    if prob > 1.0:
        raise ValueError("噪声比例不能大于1.0")
    noise_image = image.copy()
    h, w, _ = image.shape

    # 计算需要改变的像素总数
    total_pixels = int(h * w * prob)
    pepper_pixels = total_pixels // 2
    salt_pixels = total_pixels - pepper_pixels

    # 随机生成椒噪声坐标
    pepper_y = np.random.randint(0, h, pepper_pixels)
    pepper_x = np.random.randint(0, w, pepper_pixels)
    noise_image[pepper_y, pepper_x] = [0, 0, 0]

    # 随机生成盐噪声坐标
    salt_y = np.random.randint(0, h, salt_pixels)
    salt_x = np.random.randint(0, w, salt_pixels)
    noise_image[salt_y, salt_x] = [255, 255, 255]

    return noise_image
