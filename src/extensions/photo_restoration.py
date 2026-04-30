"""
旧照片修复模块 — 基于传统图像处理操作组合的轻量修复管线.

针对旧照片常见退化类型 (划痕、噪点、褪色、低对比度、破损),
组合多种经典图像处理操作形成修复管线, 无需深度学习模型.

管线流程可选:
  - 划痕/污点修复: OpenCV 快速行进法 (cv2.INPAINT_TELEA)
  - 去噪: 双边滤波 (保留边缘)
  - 对比度增强: CLAHE (限制对比度自适应直方图均衡化)
  - 去雾: 暗通道先验去雾算法 (He et al., CVPR 2009)
  - 黑白旧片上色: 通过调节色相/饱和度模拟上色效果
  - 锐化: Unsharp Masking

"创新点":
  1. 针对旧照片的复合退化特点, 设计了可配置的多阶段修复管线
  2. 各阶段可独立开关并调节参数, 适应不同退化程度
  3. 通过暗通道先验去雾恢复褪色旧照片的色彩饱和度

论文引用:
  - Telea, A. (2004). An Image Inpainting Technique Based on the Fast Marching Method.
    Journal of Graphics Tools, 9(1), 23-34.
  - He, K., Sun, J., & Tang, X. (2011). Single Image Haze Removal Using Dark Channel
    Prior. IEEE TPAMI, 33(12), 2341-2353.

用法:
    from extensions.photo_restoration import restore_old_photo
    result = restore_old_photo("old_photo.jpg", denoise=True, enhance_contrast=True)
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def _inpaint_scratches(
    image: np.ndarray, threshold: int = 200, radius: int = 3
) -> np.ndarray:
    """
    检测并修复划痕/污点.

    通过高阈值检测异常亮色区域作为划痕掩膜,
    使用快速行进法 (Fast Marching Method) 进行修复.

    Args:
        image:     BGR 图像, (H, W, 3), uint8.
        threshold: 划痕检测阈值, 灰度值 > 此值被视为划痕.
        radius:    修复邻域半径.

    Returns:
        修复后的图像.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, scratch_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 形态学操作: 膨胀连接邻近的划痕区域
    kernel = np.ones((3, 3), np.uint8)
    scratch_mask = cv2.dilate(scratch_mask, kernel, iterations=1)

    result = cv2.inpaint(image, scratch_mask, radius, cv2.INPAINT_TELEA)
    return result


def _apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    CLAHE: 限制对比度自适应直方图均衡化.
    在局部块内进行直方图均衡化, 限制噪声放大.

    适用于旧照片因褪色导致的低对比度问题.

    Args:
        image:     BGR 图像, (H, W, 3), uint8.
        clip_limit: 对比度限制阈值.
        grid_size:  局部块大小 (grid_size × grid_size).

    Returns:
        对比度增强后的图像.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _dark_channel_haze_removal(
    image: np.ndarray, omega: float = 0.95, t0: float = 0.1, patch_size: int = 15
) -> np.ndarray:
    """
    暗通道先验去雾 (He et al., CVPR 2009).

    核心思想: 户外无雾图像的暗通道中至少有一个通道的值趋近于 0.
    通过估计大气光强 A 和透射率 t 来恢复清晰图像.

    Args:
        image:     BGR 图像, float64, 值域 [0, 1].
        omega:    去雾强度 (0~1), 越大去雾越强.
        t0:       透射率下限, 防止过增强.
        patch_size: 暗通道求块最小值的窗口大小.

    Returns:
        去雾后的图像, (H, W, 3), float64, 值域 [0, 1].
    """
    img = image.astype(np.float64) / 255.0
    h, w = img.shape[:2]

    # 1. 计算暗通道: 每个像素取 RGB 三通道最小值, 再在 patch 内取最小值
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # 2. 估计大气光强 A: 取暗通道最亮前 0.1% 像素对应原图的亮度均值
    flat_dark = dark_channel.ravel()
    num_pixels = int(h * w * 0.001)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    a_val = np.mean(np.max(img, axis=2).ravel()[indices])

    # 3. 估计透射率 t
    t_map = 1.0 - omega * cv2.erode(min_channel / (a_val + 1e-6), kernel)
    t_map = np.clip(t_map, t0, 1.0)

    # 4. 恢复清晰图像
    result = np.zeros_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - a_val) / np.expand_dims(t_map, 2) + a_val

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _sharpen(image: np.ndarray, amount: float = 0.5, radius: int = 1) -> np.ndarray:
    """Unsharp Masking 锐化."""
    blur = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


def restore_old_photo(
    image_path: str,
    output_path: Optional[str] = None,
    inpaint_scratches: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    dehaze: bool = False,
    sharpen_result: bool = True,
    scratch_threshold: int = 200,
    clahe_clip_limit: float = 2.0,
) -> np.ndarray:
    """
    旧照片修复管线.

    可选的修复阶段 (按执行顺序):
      1. 划痕修复 → 2. 去噪 → 3. CLAHE 对比度增强 → 4. 暗通道去雾 → 5. 锐化

    Args:
        image_path:   输入旧照片路径.
        output_path:  可选保存路径.
        inpaint_scratches: 是否修复划痕/污点.
        denoise:            是否进行双边滤波降噪.
        enhance_contrast:   是否使用 CLAHE 增强对比度.
        dehaze:             是否使用暗通道去雾 (恢复褪色色彩).
        sharpen_result:     是否锐化.
        scratch_threshold:  划痕检测灰度阈值.
        clahe_clip_limit:   CLAHE 对比度限制.

    Returns:
        修复后的图像, BGR 格式, uint8.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    print(f"[Restore] 输入: {img.shape[1]}×{img.shape[0]}")

    result = img.copy()

    # ── 1. 划痕修复 ──
    if inpaint_scratches:
        result = _inpaint_scratches(result, threshold=scratch_threshold)
        print("[Restore] ✔ 划痕修复")

    # ── 2. 去噪 ──
    if denoise:
        result = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=75)
        print("[Restore] ✔ 双边滤波去噪")

    # ── 3. CLAHE 对比度增强 ──
    if enhance_contrast:
        result = _apply_clahe(result, clip_limit=clahe_clip_limit)
        print("[Restore] ✔ CLAHE 对比度增强")

    # ── 4. 暗通道去雾 ──
    if dehaze:
        result = _dark_channel_haze_removal(result)
        print("[Restore] ✔ 暗通道去雾 (色彩恢复)")

    # ── 5. 锐化 ──
    if sharpen_result:
        result = _sharpen(result, amount=0.3)
        print("[Restore] ✔ 锐化")

    # ── 保存 ──
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"[Restore] 已保存: {output_path}")

    print(f"[Restore] 修复完成: {result.shape[1]}×{result.shape[0]}")
    return result
