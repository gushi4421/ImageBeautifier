"""
本模块实现了对图片的预处理、读取和保存功能
支持 BMP、PNG、JPG等常见图片格式
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2 as cv
from PIL import Image

from src.utils import to_bgr_numpy


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
    image_bgr = to_bgr_numpy(image)
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

    image_bgr = to_bgr_numpy(image)
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(path)
    return True
