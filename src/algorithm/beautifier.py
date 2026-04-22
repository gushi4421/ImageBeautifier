"""
该模块实现了对图像的美化处理.
包含艺术滤镜特效: 浮雕(Emboss)与毛玻璃(Frosted).
"""

import numpy as np
import math


def emboss(image: np.ndarray) -> np.ndarray:
    """
    浮雕效果滤镜.

    Args:
        image: 输入图像.

    Returns:

    """
    # 转为浮点型以处理负数差值
    image_float = image.astype(np.float32)
    # 边缘填充, 确保输出尺寸一致
    padded = np.pad(image_float, ((1, 1), (1, 1), (0, 0)), mode="edge")

    # p1 为左上方邻域像素的加权和
    p1 = padded[:-2, :-2] + padded[:-2, 1:-1] + padded[1:-1, :-2]

    # p2 为右下方邻域像素的加权和
    p2 = padded[2:, 2:] + padded[1:-1, 2:] + padded[2:, 1:-1]

    # 核心公式: 邻域差值 + 灰度偏移量(128.0)
    # 这种差分运算能抵消平滑区域(差值为0), 突出变化剧烈的边缘
    embossed_image = p1 - p2 + 128.0

    return np.clip(embossed_image, 0, 255).astype(np.uint8)


def frosted(image: np.ndarray, offset: int = 1) -> np.ndarray:
    """
    毛玻璃(磨砂)效果滤镜.

    通过在局部邻域内进行随机像素采样, 破坏图像的连续性, 产生磨砂质感.
    """
    h, w, _ = image.shape
    # 生成标准的坐标网格
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # 生成与图像等大的随机偏移矩阵
    # offset 决定了磨砂的颗粒感粗细
    random_x = np.random.randint(-offset, offset + 1, size=(h, w))
    random_y = np.random.randint(-offset, offset + 1, size=(h, w))

    # 将随机偏移叠加到原始坐标上, 得到映射后的源坐标
    src_x = x_grid + random_x
    src_y = y_grid + random_y

    # 边界检查: 确保偏移后的坐标不会超出原图范围
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # 核心映射: 利用索引广播(Indexing)一次性提取所有随机点的像素值
    frosted_image = image[src_y, src_x]
    return frosted_image


def generate_gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    生成高斯模糊的卷积核

    Args:
        kernel_size: 卷积核的大小,必须是大于 0 的奇数
        sigma: 高斯分布的标准差, 控制模糊的平滑度

    Returns:
        返回形状为 (kernel_size,kernel_size) 的二维浮点矩阵, 且和为1
    """
    if kernel_size % 2 == 0 or kernel_size < 0:
        raise ValueError()
    # 初始化卷积核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    radius = kernel_size // 2
    # 遍历矩阵每一个点
    for i in range(kernel_size):
        for j in range(kernel_size):
            # 以卷积核中心为坐标中心的坐标
            x = i - radius
            y = j - radius
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    # 保证权重之和为1
    kernel = kernel / kernel.sum()
    return kernel


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

    return np.clip(filtered_image.astype(np.uint8), 0, 255)
