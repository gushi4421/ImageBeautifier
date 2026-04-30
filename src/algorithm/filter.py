"""
该模块实现空间域滤波算法.
包含均值滤波、中值滤波、高斯模糊以及双边滤波.
"""

import cv2 as cv
import numpy as np


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

    return cv.blur(
        image,
        ksize=(kernel_size, kernel_size),
        borderType=cv.BORDER_REFLECT_101,
    )


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

    pad_size = kernel_size // 2
    padded_image = cv.copyMakeBorder(
        image,
        pad_size,
        pad_size,
        pad_size,
        pad_size,
        borderType=cv.BORDER_REFLECT_101,
    )
    filtered_image = cv.medianBlur(padded_image, kernel_size)
    return filtered_image[
        pad_size : pad_size + image.shape[0],
        pad_size : pad_size + image.shape[1],
    ]


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
    return cv.GaussianBlur(
        image,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv.BORDER_REFLECT_101,
    )


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
    return cv.bilateralFilter(
        image,
        d=kernel_size,
        sigmaColor=sigma_r,
        sigmaSpace=sigma_s,
        borderType=cv.BORDER_REFLECT_101,
    )
