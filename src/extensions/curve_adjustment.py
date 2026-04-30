"""
曲线调整模块 — 非线性灰度映射.

传统亮度/对比度调节是线性的 (y = a*x + b), 无法针对特定亮度区域进行精细控制.
曲线调整使用三次 Catmull-Rom 样条在若干控制点之间插值, 生成平滑的非线性映射曲线,
实现暗部提亮、高光压缩、S 曲线增强等效果.

原理:
  1. 用户定义若干 (x, y) 控制点 (0~255)
  2. 插值生成一条平滑的映射曲线 LUT[0..255]
  3. 原图每个像素查表映射

用法:
    from extensions.curve_adjustment import apply_curve
    result = apply_curve(image, [(0,0), (64,40), (192,200), (255,255)])
    # → S 曲线: 暗部压暗, 亮部提亮
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """
    Catmull-Rom 样条插值.

    在 p1 和 p2 之间, 根据 t ∈ [0, 1] 计算插值点.
    曲线经过所有控制点, 且切线连续.

    Args:
        p0, p1, p2, p3: 四个连续控制点的值.
        t: 参数, [0, 1] 对应 p1 到 p2 的弧长比例.

    Returns:
        插值结果.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def _build_curve_lut(points: list[tuple[float, float]]) -> np.ndarray:
    """
    从控制点生成 256 项的查找表.

    Args:
        points: 控制点列表, 每个为 (input, output), 值域 [0, 255].
                至少需要 2 个点. 第一个点 x=0, 最后一个点 x=255.

    Returns:
        LUT 数组, (256,) float32, 值域 [0, 255].
    """
    p = sorted(points, key=lambda xy: xy[0])
    if p[0][0] != 0:
        p.insert(0, (0, 0))
    if p[-1][0] != 255:
        p.append((255, 255))

    lut = np.zeros(256, dtype=np.float32)
    n = len(p)

    for seg in range(n - 1):
        x0, y0 = p[seg]
        x1, y1 = p[seg + 1]

        if x1 == x0:
            continue

        # Catmull-Rom 需要 4 个点: p_{seg-1}, p_seg, p_{seg+1}, p_{seg+2}
        # 边界处理: 镜像外推
        p_before = p[max(seg - 1, 0)]
        p_after = p[min(seg + 2, n - 1)]

        for x in range(max(0, int(x0)), min(256, int(x1))):
            t = (x - x0) / (x1 - x0 + 1e-8)
            lut[x] = _catmull_rom(
                p_before[1],
                y0,
                y1,
                p_after[1],
                t,
            )

    # 边界点直接赋值
    lut[0] = p[0][1]
    lut[255] = p[-1][1]

    return np.clip(lut, 0, 255)


# ── 预设曲线 ──
PRESETS = {
    "s_curve": [(0, 0), (64, 40), (192, 215), (255, 255)],
    "brighten_shadows": [(0, 0), (64, 100), (192, 200), (255, 255)],
    "compress_highlights": [(0, 0), (128, 128), (192, 180), (255, 230)],
    "invert": [(0, 255), (255, 0)],
    "vintage_fade": [(0, 20), (128, 140), (255, 240)],
}


def apply_curve(
    image: np.ndarray,
    points: Optional[list[tuple[float, float]]] = None,
    preset: Optional[str] = None,
) -> np.ndarray:
    """
    对图像应用非线性曲线调整.

    Args:
        image:  输入图像, BGR 格式, (H, W, 3), uint8.
        points: 自定义控制点列表. 为 None 时使用 preset.
        preset: 预设曲线名称: "s_curve" / "brighten_shadows" /
                "compress_highlights" / "invert" / "vintage_fade".

    Returns:
        调整后的图像, (H, W, 3), uint8.

    预设说明:
        s_curve:               S 形曲线 — 暗部更暗, 亮部更亮 (增强氛围).
        brighten_shadows:      提亮暗部 — 保留亮部, 只提暗部.
        compress_highlights:   压缩高光 — 保留暗部, 压暗过曝.
        invert:                反相.
        vintage_fade:          褪色效果 — 暗部变灰, 模拟旧照片.
    """
    if preset and preset in PRESETS:
        pts = PRESETS[preset]
    elif points:
        pts = points
    else:
        raise ValueError("请提供 points 或 preset")

    lut = _build_curve_lut(pts)

    # 分通道应用 (对每个通道单独做曲线映射, 或转灰度做全局)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = cv2.LUT(image[:, :, c], lut.astype(np.uint8))

    return result
