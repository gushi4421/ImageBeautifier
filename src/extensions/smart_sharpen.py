"""
智能锐化模块 — 边缘感知锐化 (Edge-Aware Sharpening).

普通 Unsharp Masking 锐化是"全图一刀切"——平滑区和噪点也一起锐化, 看起来生硬。
智能锐化只在检测到的边缘区域进行 USM, 平滑区和噪点保留不动。

流程:
  1. Sobel 算子提取边缘梯度图
  2. 梯度图二值化 / 软蒙版 → 生成加权掩膜
  3. Unsharp Masking 仅在掩膜区域生效

用法:
    from extensions.smart_sharpen import smart_sharpen
    result = smart_sharpen(image, amount=1.5, edge_threshold=30)
"""

from __future__ import annotations

import numpy as np
import cv2


def _sobel_edge_mask(
    gray: np.ndarray, threshold: float, soft_width: float = 10.0
) -> np.ndarray:
    """
    用 Sobel 算子提取边缘梯度强度并生成软蒙版.

    软蒙版 vs 硬二值: 在 edge_threshold ± soft_width/2 区间内做线性过渡,
    避免锐化边缘与非锐化区域之间出现明显分界线.

    Args:
        gray:         灰度图, (H, W), uint8.
        threshold:    边缘判定阈值, 梯度 > threshold 的区域被视为边缘.
        soft_width:   软过渡宽度 (像素).

    Returns:
        软蒙版, (H, W), float32, 值域 [0, 1].
    """
    # Sobel 水平 + 垂直梯度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # 软阈值: sigmoid-like 线性斜坡
    half = soft_width / 2.0
    mask = np.clip((mag - threshold + half) / soft_width, 0.0, 1.0)
    return mask.astype(np.float32)


def smart_sharpen(
    image: np.ndarray,
    amount: float = 1.5,
    radius: float = 1.0,
    edge_threshold: float = 30.0,
    soft_width: float = 10.0,
) -> np.ndarray:
    """
    智能锐化 — 仅锐化边缘区域.

    Args:
        image:          输入图像, BGR 格式, (H, W, 3), uint8.
        amount:         锐化强度. 0 无效果, 1~2 正常, >3 夸张.
        radius:         高斯模糊半径 (控制细节尺度).
        edge_threshold: 边缘检测灵敏度. 越低越多区域被锐化.
        soft_width:     边缘过渡带宽度. 0 = 硬切换, 10 = 平滑过渡.

    Returns:
        锐化后的图像, (H, W, 3), uint8.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_mask = _sobel_edge_mask(gray, edge_threshold, soft_width)

    # Unsharp Masking
    blur = cv2.GaussianBlur(image, (0, 0), radius)
    detail = image.astype(np.float32) - blur.astype(np.float32)
    sharpened = image.astype(np.float32) + amount * detail
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 仅在边缘区域应用锐化
    mask_3c = np.stack([edge_mask] * 3, axis=2)
    result = (sharpened * mask_3c + image * (1.0 - mask_3c)).astype(np.uint8)

    return result
