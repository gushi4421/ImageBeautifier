"""
该模块实现频域滤波算法.
包含理想滤波器、高斯滤波器以及巴特沃斯滤波器, 支持低通、高通、带通与带阻四种模式.
所有滤波操作均在灰度图上进行, 彩色图像会在函数内部自动转为灰度.
"""

import numpy as np

from src.algorithm.tone import to_grayscale
from src.utils import (
    fft,
    create_lowpass_mask,
    create_highpass_mask,
    create_bandpass_mask,
    create_bandreject_mask,
    cutoff_to_d0,
    spectral_filter,
)


def lowpass_filter(
    image: np.ndarray,
    cutoff: float = 0.2,
    mode: str = "gaussian",
    order: int = 2,
) -> np.ndarray:
    """
    频域低通滤波, 保留低频分量, 实现图像平滑/去噪.

    Args:
        image: 输入图像 (彩色或灰度均可, 内部自动转灰度).
        cutoff: 截止频率 (0~1).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        order: 巴特沃斯阶数.

    Returns:
        平滑后的灰度图像, uint8 类型.
    """
    # 彩色图转为灰度图, 便于后续傅里叶变换
    if image.ndim == 3:
        image = to_grayscale(image)

    # 第一步: 执行傅里叶变换, 将图像从空间域映射至频域
    f_shift = fft(image)

    # 第二步: 根据图像尺寸和截止频率生成低通掩膜
    h, w = image.shape
    d0 = cutoff_to_d0((h, w), cutoff)

    # 第三步: 根据指定模式生成对应的频域滤波器掩膜
    mask = create_lowpass_mask(image_shape=(h, w), d0=d0, mode=mode, n=order)

    # 第四步: 将掩膜作用于频谱并逆变换回空间域
    filtered_image = spectral_filter(f_shift=f_shift, mask=mask)
    return filtered_image


def highpass_filter(
    image: np.ndarray,
    cutoff: float = 0.1,
    mode: str = "gaussian",
    order: int = 2,
) -> np.ndarray:
    """
    频域高通滤波, 保留高频分量, 实现边缘提取/锐化.

    Args:
        image: 输入图像 (彩色或灰度均可, 内部自动转灰度).
        cutoff: 截止频率 (0~1).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        order: 巴特沃斯阶数.

    Returns:
        锐化后的灰度图像, uint8 类型.
    """
    # 彩色图转为灰度图
    if image.ndim == 3:
        image = to_grayscale(image)

    # 第一步: 执行傅里叶变换, 将图像从空间域映射至频域
    f_shift = fft(image)

    # 第二步: 将归一化的截止频率转化为像素单位的截止半径
    h, w = image.shape
    d0 = cutoff_to_d0(image_shape=(h, w), cutoff=cutoff)

    # 第三步: 生成高通掩膜 (内部调用低通掩膜取反, H_hp = 1 - H_lp)
    mask = create_highpass_mask(image_shape=(h, w), d0=d0, mode=mode, n=order)

    # 第四步: 掩膜作用于频谱并逆变换回空间域
    filtered_image = spectral_filter(f_shift=f_shift, mask=mask)
    return filtered_image


def bandpass_filter(
    image: np.ndarray,
    low_cut: float = 0.1,
    high_cut: float = 0.4,
    mode: str = "gaussian",
    order: int = 2,
) -> np.ndarray:
    """
    频域带通滤波, 保留中间频段分量.

    Args:
        image: 输入图像 (彩色或灰度均可, 内部自动转灰度).
        low_cut: 内圈截止频率 (0~1).
        high_cut: 外圈截止频率 (0~1).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        order: 巴特沃斯阶数.

    Returns:
        滤波后的灰度图像, uint8 类型.
    """
    # 彩色图转为灰度图
    if image.ndim == 3:
        image = to_grayscale(image)

    # 第一步: 执行傅里叶变换
    f_shift = fft(image)

    # 第二步: 将内外圈截止频率分别转换为像素半径单位
    h, w = image.shape
    low_d0 = cutoff_to_d0(image_shape=(h, w), cutoff=low_cut)
    high_d0 = cutoff_to_d0(image_shape=(h, w), cutoff=high_cut)

    # 第三步: 生成带通掩膜 (内部由外圈低通减去内圈低通得到环状通带)
    mask = create_bandpass_mask(
        image_shape=(h, w), low_d0=low_d0, high_d0=high_d0, mode=mode, n=order
    )

    # 第四步: 掩膜作用于频谱并逆变换回空间域
    filtered_image = spectral_filter(f_shift=f_shift, mask=mask)
    return filtered_image


def bandreject_filter(
    image: np.ndarray,
    low_cut: float = 0.1,
    high_cut: float = 0.4,
    mode: str = "gaussian",
    order: int = 2,
) -> np.ndarray:
    """
    频域带阻滤波, 抑制中间频段分量 (可用于去除周期性噪声).

    Args:
        image: 输入图像 (彩色或灰度均可, 内部自动转灰度).
        low_cut: 内圈截止频率 (0~1).
        high_cut: 外圈截止频率 (0~1).
        mode: 滤波器类型, "ideal" / "gaussian" / "butterworth".
        order: 巴特沃斯阶数.

    Returns:
        滤波后的灰度图像, uint8 类型.
    """
    # 彩色图转为灰度图
    if image.ndim == 3:
        image = to_grayscale(image)

    # 第一步: 执行傅里叶变换
    f_shift = fft(image)

    # 第二步: 将内外圈截止频率分别转换为像素半径单位
    h, w = image.shape
    low_d0 = cutoff_to_d0(image_shape=(h, w), cutoff=low_cut)
    high_d0 = cutoff_to_d0(image_shape=(h, w), cutoff=high_cut)

    # 第三步: 生成带阻掩膜 (内部由带通掩膜取反得到, H_br = 1 - H_bp)
    mask = create_bandreject_mask(
        image_shape=(h, w), low_d0=low_d0, high_d0=high_d0, mode=mode, n=order
    )

    # 第四步: 掩膜作用于频谱并逆变换回空间域
    filtered_image = spectral_filter(f_shift=f_shift, mask=mask)
    return filtered_image
