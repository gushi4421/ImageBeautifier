"""
该模块用于实现对图像的加噪和去噪

"""

import numpy as np
from torch import Value


# 1. 加噪
def add_noise(image: np.ndarray, mode: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Args:
        image: 传入的图像
        noise_type: 加噪类型, 支持 gaussian 和 salt_pepper
    仅支持 "gaussian"(高斯噪声) 和 "salt_pepper"(椒盐噪声)

    高斯噪声支持的参数:
        - mean: 均值,默认为0.0
        - sigma: 标准差,默认为20.0
    椒盐噪声支持的参数:
        -prob: 噪声比例,默认为0.1
    """
    support_types = ["gaussian", "salt_pepper"]
    if mode not in support_types:
        raise ValueError("不支持的噪声类型")

    if mode == "gaussian":
        mean = kwargs.get("mean", 0.0)
        sigma = kwargs.get("sigma", 20.0)
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image.astype(np.float64) + noise
        return np.clip(noisy_image, 0, 255).astype(image.dtype)
    elif mode == "salt_pepper":
        prob = kwargs.get("prob", 0.1)
        if prob > 1.0:
            raise ValueError()
        noise_image = image.copy()
        h, w, _ = image.shape
        random_matrix = np.random.rand(h, w)

        noise_image[random_matrix < (prob / 2.0)] = [0, 0, 0]
        noise_image[(random_matrix > (prob / 2.0)) & (random_matrix < prob)] - [
            255,
            255,
            255,
        ]
    # elif mode == "salt_pepper":
    #     better_salt_pepper_noise(image, prob=prob)
        return noise_image


# 2. 去噪
def remove_noise(
    image: np.ndarray, mode: str = "mean", kernal_size: int = 3
) -> np.ndarray:
    """
    去噪函数,仅支持均值去噪和中值去噪
    """
    if mode not in ["mean", "median"]:
        raise ValueError()
    if kernal_size % 2 == 0:
        raise ValueError()
    h, w, c = image.shape
    pad_size = kernal_size // 2
    padded_image = np.pad(
        image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )
    filtered_image = np.zeros_like(image)
    for u in range(w):
        for v in range(h):
            window = padded_image[v : v + kernal_size, u : u + kernal_size, :]
            if mode == "mean":
                filtered_image[v, u] = window.mean(axis=(0, 1)).astype(np.uint8)
            elif mode == "median":
                filtered_image[v, u] = np.median(window, axis=(0, 1)).astype(np.uint8)
    return filtered_image


# 优化的椒盐噪声
def better_salt_pepper_noise(image: np.ndarray, prob: float):
    if prob > 1.0:
        raise ValueError("噪声比例不能大于1.0")
    noise_image = image.copy()
    h, w, _ = image.shape
    total_pixels = int(h * w * prob)
    pepper_pixels = total_pixels // 2
    salt_pixels = total_pixels - pepper_pixels

    pepper_y = np.random.randint(0, h, pepper_pixels)
    pepper_x = np.random.randint(0, w, pepper_pixels)
    noise_image[pepper_y, pepper_x] = [0, 0, 0]

    salt_y = np.random.randint(0, w, salt_pixels)
    salt_x = np.random.randint(0, w, salt_pixels)
    noise_image[salt_y, salt_x] = [255, 255, 255]
    return noise_image
