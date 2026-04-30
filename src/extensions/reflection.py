"""
倒影图制作模块 — 模拟水面倒影效果.

原理: 垂直翻转原图下半部分 + 渐变透明叠加, 产生水面反射的视觉效果.
      "创新点": 支持调节倒影长度比例和渐变过渡的陡峭程度,
      比直接垂直翻转更自然.

用法:
    from extensions.reflection import create_reflection
    result = create_reflection(image, reflection_ratio=0.5, fade_strength=1.0)
"""

from __future__ import annotations

import numpy as np


def _linear_fade(height: int, reverse: bool = False) -> np.ndarray:
    """
    生成一维线性渐变权重.

    Args:
        height: 渐变长度 (像素).
        reverse: True 时从上往下由 1 渐变为 0; False 时相反.

    Returns:
        形状为 (height, 1) 的 float32 数组, 值域 [0, 1].
    """
    fade = np.linspace(1.0, 0.0, height, dtype=np.float32).reshape(-1, 1)
    if reverse:
        fade = 1.0 - fade
    return fade


def create_reflection(
    image: np.ndarray,
    reflection_ratio: float = 0.5,
    fade_exponent: float = 1.5,
    bg_color: tuple[int, int, int] = (200, 210, 230),
) -> np.ndarray:
    """
    为图像制作水面倒影效果.

    Args:
        image:       输入图像, BGR 格式, (H, W, 3), uint8.
        reflection_ratio: 倒影长度占原图高度的比例, 推荐 0.3~0.8.
        fade_exponent:    渐变指数. 1.0 为线性, >1.0 衰减更快, <1.0 衰减更慢.
        bg_color:         倒影底部背景色, BGR 元组, 默认浅灰蓝色模拟水面.

    Returns:
        倒影合成图, (H + int(H*ratio), W, 3), uint8, BGR 格式.
    """
    h, w = image.shape[:2]
    ref_h = int(h * reflection_ratio)

    # ── 1. 下半部分垂直翻转 ──
    bottom_section = image[h - ref_h :, :, :]  # 原图底部 ref_h 行
    reflection = np.flip(bottom_section, axis=0).astype(np.float32)

    # ── 2. 渐变蒙版: 从上到下 1 → 0 ──
    fade_mask = _linear_fade(ref_h)
    fade_mask = np.power(fade_mask, fade_exponent)  # 指数控制衰减曲线形状
    fade_mask_3c = np.tile(fade_mask, (1, w))  # (ref_h, W)

    # ── 3. 与背景色混合 ──
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    reflection_masked = (
        reflection * fade_mask_3c[..., None] + bg * (1.0 - fade_mask_3c[..., None])
    )
    reflection_masked = np.clip(reflection_masked, 0, 255).astype(np.uint8)

    # ── 4. 拼接 ──
    canvas = np.zeros((h + ref_h, w, 3), dtype=np.uint8)
    canvas[:h, :, :] = image
    canvas[h:, :, :] = reflection_masked

    return canvas
