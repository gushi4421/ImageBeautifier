"""
本模块实现图片几何变换相关的算法.
采用纯数学方法(NumPy)实现底层的坐标映射与插值逻辑.

实现以下几种几何变换:
1. 平移 (Translate)
2. 旋转 (Rotate)
3. 缩放 (Zoom) - 含双线性插值
4. 翻转 (Flip)
5. 剪切 (Shear)
"""

import numpy as np
import math


def translate(image: np.ndarray, tx, ty) -> np.ndarray:
    """
    图像绕中心点旋转变换.

    Args:
        image: 原始图像.
        angle: 旋转的角度(角度制), 正值为逆时针旋转.

    Returns:
        旋转后的图像, 尺寸与原图保持一致.
    """
    h, w, _ = image.shape
    target_image = np.zeros_like(image)

    # 遍历目标图像的每一个像素点 (u, v)
    for u in range(w):
        for v in range(h):
            # 逆向映射: 寻找目标点在原图中的对应位置 (x, y)
            # 公式: u = x + tx -> x = u - tx
            x = u - tx
            y = v - ty

            # 边界检查: 只有当映射回原图的坐标合法时才赋值
            if 0 <= x < w and 0 <= y < h:
                target_image[v, u] = image[y, x]
    return target_image


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
    # 计算图像中心坐标
    center_x = w / 2.0
    center_y = h / 2.0
    # 角度转弧度
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    target_image = np.zeros_like(image)

    for u in range(w):
        for v in range(h):
            # 第一步: 将目标坐标平移到以中心为原点的系中
            u_centered = u - center_x
            v_centered = v - center_y

            # 第二步: 执行逆向旋转变换 (逆时针旋转 theta, 则逆向映射为旋转 -theta)
            # x' = u*cos(theta) + v*sin(theta)
            # y' = -u*sin(theta) + v*cos(theta)
            x_centered = u_centered * cos_theta + v_centered * sin_theta
            y_centered = -u_centered * sin_theta + v_centered * cos_theta

            # 第三步: 还原回原图坐标系
            x = x_centered + center_x
            y = y_centered + center_y

            # 四舍五入取整进行最近邻近似
            x = int(round(x))
            y = int(round(y))

            if 0 <= x < w and 0 <= y < h:
                target_image[v, u] = image[y, x]
    return target_image


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
    h, w, c = image.shape
    new_w = int(w * x_scale)
    new_h = int(h * y_scale)
    target_image = np.zeros((new_h, new_w, c), np.uint8)

    for u in range(new_w):
        for v in range(new_h):
            # 几何中心对齐映射公式: 找到目标点在原图中的浮点坐标
            src_x = (u + 0.5) / x_scale - 0.5
            src_y = (v + 0.5) / y_scale - 0.5

            # 获取浮点坐标的小数部分 (dx, dy) 用于权重计算
            dx = src_x - math.floor(src_x)
            dy = src_y - math.floor(src_y)

            # 获取左上角参考像素的整数坐标 (x1, y1)
            x1 = int(math.floor(src_x))
            y1 = int(math.floor(src_y))

            # 边界锁定, 防止索引溢出
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = min(x1 + 1, w - 1)  # 右侧像素坐标
            y2 = min(y1 + 1, h - 1)  # 下方像素坐标

            # 获取相邻四个像素点的值
            p11 = image[y1, x1].astype(np.float32)  # 左上
            p12 = image[y2, x1].astype(np.float32)  # 左下
            p21 = image[y1, x2].astype(np.float32)  # 右上
            p22 = image[y2, x2].astype(np.float32)  # 右下

            # 核心: 双线性插值计算
            # 1. 水平方向插值
            r1 = p11 * (1 - dx) + p21 * dx
            r2 = p12 * (1 - dx) + p22 * dx

            # 2. 垂直方向插值
            p = (1 - dy) * r1 + dy * r2

            target_image[v, u] = np.clip(p, 0, 255).astype(np.uint8)

    return target_image


def flip(image: np.ndarray) -> np.ndarray:
    """
    水平翻转(镜像).

    Args:
        image: 输入图像.

    Returns:
        水平翻转后的图像.

    """
    h, w, c = image.shape
    target_image = np.zeros_like(image)

    for u in range(w):
        for v in range(h):
            # 映射公式: u' = w - u - 1
            target_image[v, w - u - 1] = image[v, u]
    return target_image


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
