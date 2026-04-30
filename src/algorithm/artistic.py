"""
该模块实现艺术滤镜特效.
包含浮雕(Emboss)与毛玻璃(Frosted)效果.
"""

import cv2 as cv
import numpy as np


def emboss(image: np.ndarray) -> np.ndarray:
    """
    浮雕效果滤镜.

    Args:
        image: 输入图像.

    Returns:

    """
    kernel = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -1.0]],
        dtype=np.float32,
    )
    embossed_image = cv.filter2D(
        image.astype(np.float32),
        ddepth=-1,
        kernel=kernel,
        borderType=cv.BORDER_REPLICATE,
    )
    embossed_image += 128.0
    return np.clip(embossed_image, 0, 255).astype(np.uint8)


def frosted(image: np.ndarray, offset: int = 1) -> np.ndarray:
    """
    毛玻璃(磨砂)效果滤镜.

    通过在局部邻域内进行随机像素采样, 破坏图像的连续性, 产生磨砂质感.
    """
    h, w, _ = image.shape
    x_grid, y_grid = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )

    random_x = np.random.randint(-offset, offset + 1, size=(h, w)).astype(np.float32)
    random_y = np.random.randint(-offset, offset + 1, size=(h, w)).astype(np.float32)

    map_x = np.clip(x_grid + random_x, 0, w - 1)
    map_y = np.clip(y_grid + random_y, 0, h - 1)

    return cv.remap(
        image,
        map_x,
        map_y,
        interpolation=cv.INTER_NEAREST,
        borderMode=cv.BORDER_REPLICATE,
    )
