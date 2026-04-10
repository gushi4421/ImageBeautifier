"""
该模块实现了对图像的美化处理.
包含艺术滤镜特效: 浮雕(Emboss)与毛玻璃(Frosted).
"""

import numpy as np


def emboss(image: np.ndarray) -> np.ndarray:
    """
    浮雕效果滤镜.

    通过计算局部像素的差值来突出边缘, 模拟雕刻的凹凸感.
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
