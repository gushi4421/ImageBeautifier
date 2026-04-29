"""
该模块实现空间域滤波算法.
包含均值滤波、中值滤波、高斯模糊以及双边滤波.
"""

import numpy as np

from src.utils import generate_gaussian_kernel


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    均值滤波, 通过对邻域像素取平均值来平滑图像.

    Args:
        image: 含有噪声的图像.
        kernel_size: 滤波核的大小, 必须为奇数(如 3, 5, 7).

    Returns:
        均值平滑后的图像.
    """
    if kernel_size % 2 == 0:
        raise ValueError("卷积核大小必须为奇数")

    h, w, c = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(
        image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )
    filtered_image = np.zeros_like(image)

    for u in range(w):
        for v in range(h):
            window = padded_image[v : v + kernel_size, u : u + kernel_size, :]
            filtered_image[v, u] = window.mean(axis=(0, 1)).astype(np.uint8)

    return filtered_image


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    中值滤波, 通过取邻域像素的中位数来去除椒盐噪声.

    Args:
        image: 含有噪声的图像.
        kernel_size: 滤波核的大小, 必须为奇数(如 3, 5, 7).

    Returns:
        中值平滑后的图像.
    """
    if kernel_size % 2 == 0:
        raise ValueError("卷积核大小必须为奇数")

    h, w, c = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(
        image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )
    filtered_image = np.zeros_like(image)

    for u in range(w):
        for v in range(h):
            window = padded_image[v : v + kernel_size, u : u + kernel_size, :]
            filtered_image[v, u] = np.median(window, axis=(0, 1)).astype(np.uint8)

    return filtered_image


def gaussian_blurring(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0):
    """
    实现高斯模糊

    Args:
        image: 输入的图像
        kernel_size: 卷积核大小,由于未使用 opencv 故不建议将其设置大于3
        sigma: 标准差

    Returns:
        返回经高斯模糊处理的图像
    """
    kernel = generate_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    h, w, _ = image.shape
    pad_size = kernel_size // 2

    padded_image = np.pad(
        image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )
    blurred_image = np.zeros_like(image, dtype=np.float32)

    for u in range(w):
        for v in range(h):
            # 提取与卷积核等大的局部图像窗口
            window = padded_image[v : v + kernel_size, u : u + kernel_size, :]

            # 将二维的卷积核通过 reshape(kernel_size, kernel_size, 1) 扩展为三维
            # 利用 Numpy 的广播机制, 将高斯权重同时乘以 R, G, B 三个通道的像素值
            weighted_window = window * kernel.reshape(kernel_size, kernel_size, 1)

            # 将加权后的像素值求和, 赋给中心点
            blurred_image[v, u] = np.sum(weighted_window, axis=(0, 1))

    return np.clip(blurred_image, 0, 255).astype(np.uint8)


def bilateral_filter_manual(
    image: np.ndarray,
    kernel_size: int = 3,
    sigma_s: float = 15.0,
    sigma_r: float = 30.0,
) -> np.ndarray:
    """
    实现双边滤波

    Args:
        image: 输入的图像
        kernel_size: 滤波窗口的大小, 必须为奇数
        sigma_s: 空间标准差, 控制物理模糊的范围
        sigma_r: 色彩标准差, 控制对边缘颜色的敏感度

    Returns:
        平滑后的图像,类型为np.uint8
    """
    h, w, _ = image.shape
    pad_size = kernel_size // 2
    image_float = image.copy().astype(np.float32)
    padded_image = np.pad(
        image_float,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode="reflect",
    )
    filtered_image = np.zeros_like(image_float)
    x_grid, y_grid = np.mgrid[-pad_size : pad_size + 1, -pad_size : pad_size + 1]
    gaussian_weight = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma_s**2))
    gaussian_weight = gaussian_weight.reshape(kernel_size, kernel_size, 1)
    for u in range(w):
        for v in range(h):
            window = padded_image[v : v + kernel_size, u : u + kernel_size, :]
            center_pixel = padded_image[v + pad_size, u + pad_size, :]

            # 色彩差异
            color_diff = np.sum((window - center_pixel) ** 2, axis=2, keepdims=True)
            # 颜色权重
            color_weight = np.exp(-color_diff / (2 * sigma_r**2))

            combined_weight = gaussian_weight * color_weight
            combined_weight /= np.sum(combined_weight, axis=(0, 1), keepdims=True)
            filtered_image[v, u] = np.sum(window * combined_weight, axis=(0, 1))

    return np.clip(filtered_image, 0, 255).astype(np.uint8)
