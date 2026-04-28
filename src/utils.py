"""
该模块提供算法内部使用的工具函数.
包含高斯卷积核生成、伪彩色查找表生成、图像格式转换以及傅里叶变换工具.
"""

from __future__ import annotations

import math
from typing import Any

import cv2 as cv
import numpy as np
from PIL import Image


def generate_gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    生成高斯模糊的卷积核.

    Args:
        kernel_size: 卷积核的大小, 必须是大于 0 的奇数.
        sigma: 高斯分布的标准差, 控制模糊的平滑度.

    Returns:
        返回形状为 (kernel_size, kernel_size) 的二维浮点矩阵, 且和为 1.
    """
    if kernel_size % 2 == 0 or kernel_size < 0:
        raise ValueError()
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    radius = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - radius
            y = j - radius
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def generate_colormap_lut() -> np.ndarray:
    """
    生成伪彩色查找表(Look-Up Table).
    将 0-255 映射到蓝色到红色的渐变区间.
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


def to_bgr_numpy(image: Any) -> np.ndarray:
    """
    将任意输入的图片对象转换为标准的 3 通道 BGR Numpy 数组.
    包含对 RGBA 透明通道的白色背景融合处理.

    Args:
        image: PIL Image 对象或 Numpy 数组.

    Returns:
        标准的 BGR 格式 Numpy 数组.
    """
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            rgb_image = np.asarray(background, dtype=np.uint8)
        else:
            rgb_image = image.convert("RGB")
            rgb_image = np.asarray(rgb_image, dtype=np.uint8)

        return cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise TypeError("图片必须是 PIL Image 或 numpy.ndarray 类型.")

    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "图片数组必须是形状为 (H, W), (H, W, 1), (H, W, 3) 或 (H, W, 4) 的数组."
        )

    return image.copy()


def fft(image: np.ndarray) -> np.ndarray:
    """
    对灰度图执行傅里叶变换, 并将低频分量搬移至频谱中心.

    Args:
        image: 灰度图像, 形状为 (H, W).

    Returns:
        中心化后的频谱, 复数类型, 形状为 (H, W).
    """
    if image.ndim != 2:
        raise ValueError("频域变换前必须将图像转换为单通道的灰度图")
    # 转化为float类型
    image_float = image.copy().astype(np.float32)

    # 利用numpy.fft.fft2执行快速傅里叶变换
    # 输出的 f_transform是 一个包含复数的矩阵
    f_trainsform = np.fft.fft2(image_float)

    # FFT 默认将最低频分量放置在矩阵的左上角
    # fftshift 通过对角象限交换，将最低频分量强行移动到矩阵的物理正中心
    f_shift = np.fft.fftshift(f_trainsform)

    return f_shift


def ifft(spectrum: np.ndarray) -> np.ndarray:
    """
    对频谱执行逆傅里叶变换, 并还原为 uint8 灰度图像.

    Args:
        spectrum: 中心化的频谱, 复数类型, 形状为 (H, W).

    Returns:
        重建后的灰度图像, uint8 类型, 形状为 (H, W).
    """
    if spectrum.ndim != 2:
        raise ValueError("输入的图像必须是单通道的灰度图")
    # 将最低频分量放回四个角
    f_ishift = np.fft.ifftshift(spectrum)

    # 执行逆傅里叶变换
    f_inverse = np.fft.ifft2(f_ishift)

    return np.clip(np.abs(f_inverse), 0, 255).astype(np.uint8)


def create_lowpass_mask(
    image_shape: tuple,
    d0: float,
    mode: str = "ideal",
    n: int = 2,
) -> np.ndarray:
    """
    生成低通滤波器频域掩膜.

    低通滤波是其他三种模式的基础:
    - 高通 = 1 - 低通
    - 带通 = 低通(外圈) - 低通(内圈)
    - 带阻 = 1 - 带通

    Args:
        image_shape: 图像形状 (H, W).
        d0: 截止频率半径 (像素单位).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        n: 巴特沃斯阶数 (仅 mode="butterworth" 时生效).

    Returns:
        形状为 (H, W) 的浮点掩膜矩阵, 值域为 [0, 1].
    """
    h, w = image_shape
    center_x, center_y = w // 2, h // 2

    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    distance_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    dist = np.sqrt(distance_sq)

    if mode == "ideal":
        mask = np.zeros(image_shape, dtype=np.float32)
        mask[distance_sq <= d0**2] = 1.0
        return mask

    elif mode == "gaussian":
        return np.exp(-distance_sq / (2 * d0**2)).astype(np.float32)

    elif mode == "butterworth":
        return (1.0 / (1.0 + (dist / d0) ** (2 * n))).astype(np.float32)

    else:
        raise ValueError(f"不支持的滤波器类型: {mode}")


def create_highpass_mask(
    image_shape: tuple,
    d0: float,
    mode: str = "ideal",
    n: int = 2,
) -> np.ndarray:
    """
    生成高通滤波器频域掩膜.

    基于互补性质: H_hp = 1 - H_lp.

    Args:
        image_shape: 图像形状 (H, W).
        d0: 截止频率半径 (像素单位).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        n: 巴特沃斯阶数.

    Returns:
        形状为 (H, W) 的浮点掩膜矩阵, 值域为 [0, 1].
    """
    lp_mask = create_lowpass_mask(image_shape=image_shape, d0=d0, mode=mode, n=n)
    return 1.0 - lp_mask


def create_bandpass_mask(
    image_shape: tuple,
    low_d0: float,
    high_d0: float,
    mode: str = "ideal",
    n: int = 2,
) -> np.ndarray:
    """
    生成带通滤波器频域掩膜.

    由两个不同截止频率的低通掩膜做差得到: H_bp = H_lp(outer) - H_lp(inner).

    Args:
        image_shape: 图像形状 (H, W).
        low_d0: 内圈截止半径 (像素单位), 较小的值.
        high_d0: 外圈截止半径 (像素单位), 较大的值.
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        n: 巴特沃斯阶数.

    Returns:
        形状为 (H, W) 的浮点掩膜矩阵, 值域为 [0, 1].
    """
    outer = create_lowpass_mask(image_shape=image_shape, d0=high_d0, mode=mode, n=n)
    inner = create_lowpass_mask(image_shape=image_shape, d0=low_d0, mode=mode, n=n)
    return np.clip(outer - inner, 0, 1)


def create_bandreject_mask(
    image_shape: tuple,
    low_d0: float,
    high_d0: float,
    mode: str = "ideal",
    n: int = 2,
) -> np.ndarray:
    """
    生成带阻滤波器频域掩膜.

    基于互补性质: H_br = 1 - H_bp.

    Args:
        image_shape: 图像形状 (H, W).
        low_d0: 内圈截止半径 (像素单位).
        high_d0: 外圈截止半径 (像素单位).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        n: 巴特沃斯阶数.

    Returns:
        形状为 (H, W) 的浮点掩膜矩阵, 值域为 [0, 1].
    """
    bp_mask = create_bandpass_mask(
        image_shape=image_shape,
        low_d0=low_d0,
        high_d0=high_d0,
        mode=mode,
        n=n,
    )
    return 1.0 - bp_mask


def compute_fft(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对图像执行二维快速傅里叶变换并计算可视化的幅度谱.

    Args:
        image: 输入的单通道灰度图像数组, 形状为 (H, W).

    Returns:
        包含两个元素的元组:
        1. 频域复数矩阵 f_shift (低频已移至中心).
        2. 用于可视化的幅度谱图像 magnitude_spectrum (uint8 类型).
    """
    # [层次一: 数据预处理与 FFT 变换]
    if image.ndim != 2:
        raise ValueError("频域变换前必须将图像转换为单通道灰度图.")

    image_float = image.astype(np.float32)

    # 利用 numpy.fft.fft2 执行二维快速傅里叶变换.
    # 输出的 f_transform 是一个包含复数的矩阵 (实部和虚部).
    f_transform = np.fft.fft2(image_float)

    # [层次二: 频谱中心化]
    # FFT 默认将零频分量(最低频)放置在矩阵的左上角 (0, 0) 处.
    # fftshift 通过对角象限交换, 将零频分量强行移动到矩阵的物理正中心.
    f_shift = np.fft.fftshift(f_transform)

    # [层次三: 振幅提取与对数压缩]
    # np.abs 计算复数的模长, 即特征波的振幅大小.
    magnitude = np.abs(f_shift)

    # 核心数学映射: 频域能量呈指数级衰减分布.
    # 使用 20 * log10(1 + x) 进行非线性压缩, 否则高频细节在视觉上完全不可见.
    magnitude_spectrum = 20 * np.log10(1 + magnitude)

    # [层次四: 阈值限幅与类型还原]
    # 将浮点型幅度谱限制在物理显示设备允许的范围, 并转为标准图像格式.
    magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255).astype(np.uint8)

    return f_shift, magnitude_spectrum


def extract_phase_spectrum(f_shift: np.ndarray) -> np.ndarray:
    """
    从中心化后的频域复数矩阵中提取相位谱并进行可视化映射.

    Args:
        f_shift: 中心化后的频域复数矩阵, 形状为 (H, W).

    Returns:
        可视化的相位谱图像, 数据类型为 uint8.
    """
    # 强制校验输入必须是复数类型
    if not np.iscomplexobj(f_shift):
        raise TypeError("输入必须是包含实部和虚部的复数矩阵.")

    # [层次一: 相位数学计算]
    # np.angle 底层调用 atan2 计算虚部与实部的比值, 得到相位角.
    # phase 的取值范围严格落在 [-pi, pi] 之间.
    phase = np.angle(f_shift)

    # [层次二: 可视化标度映射]
    # 相位角包含了负数, 无法直接显示为图像.
    # 需要将 [-pi, pi] 的区间线性映射到 [0, 255] 的灰度区间.

    # 1. 归一化: (phase + pi) 将范围平移到 [0, 2*pi].
    # 2. 除以 2*pi 将范围压缩到 [0, 1].
    phase_normalized = (phase + np.pi) / (2 * np.pi)

    # [层次三: 动态范围放大与截断]
    # 乘以 255 并转为无符号 8 位整数
    phase_spectrum = np.clip(phase_normalized * 255, 0, 255).astype(np.uint8)

    return phase_spectrum


def cutoff_to_d0(image_shape: tuple, cutoff: float) -> float:
    """
    将归一化截止频率转换为像素半径.

    Args:
        image_shape: 图像形状 (H, W).
        cutoff: 归一化截止频率 (0~1), 相对于对角线一半的距离.

    Returns:
        像素单位的截止半径 d0.
    """
    h, w = image_shape
    half_diag = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    return cutoff * half_diag


def spectral_filter(f_shift: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    将频域掩膜应用于频谱复数矩阵, 并执行逆傅里叶变换(IFFT)还原图像.

    Args:
        f_shift: 中心化后的频域复数矩阵
        mask: 构建好的频域滤波器掩膜 (0到1之间的浮点矩阵).

    Returns:
        逆变换还原后的空间域图像, 类型为 uint8.
    """
    # 直接将复数矩阵与掩膜相乘
    f_filtered = f_shift * mask

    return ifft(f_filtered)
