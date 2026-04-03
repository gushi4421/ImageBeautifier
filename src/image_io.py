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


class ImageLoader:
    def load_image(self, file_path: str) -> np.ndarray:
        """
        加载图片
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"图片文件未找到: {file_path}")

        image = Image.open(path)
        image_bgr = self._to_bgr_numpy(image)
        return image_bgr

    def save_image(self, image: np.ndarray, save_path: str) -> bool:
        """
        保存图像到指定路径
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        image_bgr = self._to_bgr_numpy(image)
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        Image.fromarray(image_rgb).save(path)
        return True

    @staticmethod
    def _to_bgr_numpy(image: Any) -> np.ndarray:
        """
        将图片转化为BGR格式的numpy数组
        """
        if isinstance(image, Image.Image):
            rgb_image = image.convert("RGB")
            rgb_image = np.asarray(rgb_image, dtype=np.uint8)
            return cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

        if not isinstance(image, np.ndarray):
            raise TypeError("图片必须是PIL Image或numpy.ndarray类型")

        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "图片格式错误,图片数组必须是形状为(H, W), (H, W, 1), (H, W, 3)或(H, W, 4)的数组"
            )
        return image.copy()
