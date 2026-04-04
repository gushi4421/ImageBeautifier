"""
本模块实现了对图片的预处理、读取和保存功能
支持 BMP、PNG、JPG等常见图片格式
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2 as cv
import numpy as np
from PIL import Image


def load_image(file_path: str) -> np.ndarray:
    """
    从指定路径加载图片

    Args:
        file_path: 图片的文件路径

    Returns:
        BGR 格式的 numpy 数组
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片文件未找到: {file_path}")

    image = Image.open(path)
    image_bgr = _to_bgr_numpy(image)
    return image_bgr


def save_image(image: np.ndarray, save_path: str) -> bool:
    """
    保存图像到指定路径

    Args:
        image: 要保存的图像,BGR 格式的 numpy 数组
        save_path: 保存路径

    Returns:
        保存是否成功
    """
    path = Path(save_path)
    # 确保父级目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = _to_bgr_numpy(image)
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(path)
    return True


def _to_bgr_numpy(image: Any) -> np.ndarray:
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
            # 创建纯白背景
            background = Image.new("RGB", image.size, (255, 255, 255))
            # 提取 Alpha 通道作为掩码粘贴原图,实现透明转白色背景
            background.paste(image, mask=image.split()[3])
            rgb_image = np.asarray(background, dtype=np.uint8)
        else:
            rgb_image = image.convert("RGB")
            rgb_image = np.asarray(rgb_image, dtype=np.uint8)

        return cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise TypeError("图片必须是 PIL Image 或 numpy.ndarray 类型.")

    # 处理纯 Numpy 数组输入的维度和通道统一
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        # 直接截断 Alpha 通道作为保底方案
        image = image[..., :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "图片数组必须是形状为 (H, W), (H, W, 1), (H, W, 3) 或 (H, W, 4) 的数组."
        )

    return image.copy()
