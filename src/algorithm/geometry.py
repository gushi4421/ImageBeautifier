"""
本模块实现图片几何变换相关的算法.
采用纯数学方法(NumPy)实现底层的坐标映射与插值逻辑.

实现以下几种几何变换:
1. 平移 (Translate)
2. 旋转 (Rotate)
3. 缩放 (Zoom) - 含双线性插值
4. 翻转 (Flip)
5. 剪切 (Shear)
6. 加框 (AddBorder) - 将照片嵌入画框模板
7. 拼图 (Collage) - 将多张图片按行列网格排列
8. 水平拼接 (HConcat) - 将两张图片水平拼接
9. 垂直拼接 (VConcat) - 将两张图片垂直拼接
"""

import cv2 as cv
import numpy as np


def translate(image: np.ndarray, tx, ty) -> np.ndarray:
    """
    图像平移变换.

    Args:
        image: 原始图像.
        tx: 水平方向平移量 (正值向右).
        ty: 垂直方向平移量 (正值向下).

    Returns:
        平移后的图像, 尺寸与原图保持一致.
    """
    h, w, _ = image.shape
    transform = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return cv.warpAffine(
        image,
        transform,
        dsize=(w, h),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )


def rotate(image: np.ndarray, angle):
    """
    图像绕中心点旋转变换.

    Args:
        image: 原始图像.
        angle: 旋转的角度(角度制), 正值为逆时针旋转.

    Returns:
        旋转后的图像, 尺寸与原图保持一致.
    """
    h, w, _ = image.shape
    center = (w / 2.0, h / 2.0)
    transform = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(
        image,
        transform,
        dsize=(w, h),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )


def zoom(image: np.ndarray, x_scale, y_scale):
    """
    图像缩放变换 - 采用双线性插值算法.

    Args:
        image: 原始图像.
        x_scale: 水平方向(宽度)的缩放倍数.
        y_scale: 垂直方向(高度)的缩放倍数.

    Returns:
        缩放后的新图像, 尺寸为 (W*x_scale, H*y_scale).
    """
    h, w, _ = image.shape
    new_w = int(w * x_scale)
    new_h = int(h * y_scale)
    return cv.resize(image, dsize=(new_w, new_h), interpolation=cv.INTER_LINEAR)


def flip(image: np.ndarray) -> np.ndarray:
    """
    水平翻转(镜像).

    Args:
        image: 输入图像.

    Returns:
        水平翻转后的图像.

    """
    return cv.flip(image, 1)


def shear(image: np.ndarray, start_x, end_x, start_y, end_y):
    """
    图像裁剪(剪切).

    Args:
        image: 输入图像.
        start_x: 裁剪区域的起始水平坐标.
        end_x: 裁剪区域的结束水平坐标.
        start_y: 裁剪区域的起始垂直坐标.
        end_y: 裁剪区域的结束垂直坐标.

    Returns:
        裁剪后的图像.
    """
    h, w, c = image.shape
    # 限制范围在图像有效宽高内
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w, end_x)
    end_y = min(h, end_y)

    # 根据裁剪尺寸创建新画布
    target_image = np.zeros((end_y - start_y, end_x - start_x, c), np.uint8)

    # 像素拷贝
    for u in range(start_x, end_x):
        for v in range(start_y, end_y):
            target_image[v - start_y, u - start_x] = image[v, u]

    return target_image


def add_border(
    image: np.ndarray,
    frame: np.ndarray,
    scale: float = 0.1,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    将图片嵌入画框模板, 实现加框特效.

    画框模板作为基准, 原图按 scale 缩放嵌入画框内侧并居中放置,
    通过 offset 微调位置.

    Args:
        image: 用户的照片, 形状为 (H, W, C).
        frame: 画框背景模板, 形状通常大于 image.
        scale: 原图与画框内缩区域之间最短边的间隙比例 (0~1).
               0 表示最短边完全贴合画框, 值越大露出的画框区域越多.
        offset: 原图在内缩区域中的位置偏移量 (x, y), 取值范围 [-1, 1].
                (0, 0) 为居中, (1, 1) 紧贴左上角, (-1, -1) 紧贴右下角.

    Returns:
        合成后的加框图像.
    """
    if image.ndim != 3 or frame.ndim != 3:
        raise ValueError("仅支持三通道彩色图像进行加框合成")
    if scale < 0 or scale > 1:
        raise ValueError("scale 必须在 0 到 1 之间")

    f_h, f_w, _ = frame.shape
    i_h, i_w, _ = image.shape
    offset_x, offset_y = offset

    # 第一步: 根据 scale 计算间隙. 作用于画框较短边, 保证四个方向间隙一致
    gap = int(scale * min(f_h, f_w))

    # 第二步: 计算内缩后的可用嵌入区域
    inner_h = f_h - 2 * gap
    inner_w = f_w - 2 * gap

    # 第三步: 等比缩放照片, 使其刚好被内缩区域包住 (保持宽高比, 不变形)
    ratio = min(inner_w / i_w, inner_h / i_h)
    new_w = int(i_w * ratio)
    new_h = int(i_h * ratio)
    resized = zoom(image, new_w / i_w, new_h / i_h)

    # 第四步: 计算缩放后照片在内缩区域中的 slack 余量
    slack_x = inner_w - new_w
    slack_y = inner_h - new_h

    # 第五步: 根据 offset 分配 slack. offset ∈ [-1, 1]
    # 1 → 紧贴左/上, 0 → 居中, -1 → 紧贴右/下
    x = gap + int((1 - offset_x) / 2 * slack_x)
    y = gap + int((1 - offset_y) / 2 * slack_y)

    # 第六步: 将缩放后的照片嵌入画框副本
    canvas = frame.copy()
    canvas[y : y + new_h, x : x + new_w] = resized

    return canvas


def horizontal_collage(
    image1: np.ndarray, image2: np.ndarray, gap: int = 0
) -> np.ndarray:
    """
    水平拼接两张图像.

    两张图像垂直居中对齐, 尺寸不一致的区域以及间隙填充白色(255, 255, 255).

    Args:
        image1: 第一张图像.
        image2: 第二张图像.
        gap: 两张图像之间的间隙大小(像素), 默认为0.

    Returns:
        水平拼接后的图像.
    """
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape

    new_h = max(h1, h2)

    y1_top = (new_h - h1) // 2
    y1_bottom = new_h - h1 - y1_top
    image1_padded = cv.copyMakeBorder(
        image1,
        y1_top,
        y1_bottom,
        0,
        0,
        borderType=cv.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    y2_top = (new_h - h2) // 2
    y2_bottom = new_h - h2 - y2_top
    image2_padded = cv.copyMakeBorder(
        image2,
        y2_top,
        y2_bottom,
        0,
        0,
        borderType=cv.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    parts = [image1_padded]
    if gap > 0:
        parts.append(np.full((new_h, gap, 3), 255, dtype=np.uint8))
    parts.append(image2_padded)

    return cv.hconcat(parts)


def vertical_collage(
    image1: np.ndarray, image2: np.ndarray, gap: int = 0
) -> np.ndarray:
    """
    垂直拼接两张图像.

    两张图像水平居中对齐, 尺寸不一致的区域以及间隙填充白色(255, 255, 255).

    Args:
        image1: 第一张图像.
        image2: 第二张图像.
        gap: 两张图像之间的间隙大小(像素), 默认为0.

    Returns:
        垂直拼接后的图像.
    """
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape

    new_w = max(w1, w2)

    x1_left = (new_w - w1) // 2
    x1_right = new_w - w1 - x1_left
    image1_padded = cv.copyMakeBorder(
        image1,
        0,
        0,
        x1_left,
        x1_right,
        borderType=cv.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    x2_left = (new_w - w2) // 2
    x2_right = new_w - w2 - x2_left
    image2_padded = cv.copyMakeBorder(
        image2,
        0,
        0,
        x2_left,
        x2_right,
        borderType=cv.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    parts = [image1_padded]
    if gap > 0:
        parts.append(np.full((gap, new_w, 3), 255, dtype=np.uint8))
    parts.append(image2_padded)

    return cv.vconcat(parts)
