"""
超分辨率模块 — 基于 Real-ESRGAN (ICCV 2021)

Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
论文: https://arxiv.org/abs/2107.10833
代码: https://github.com/xinntao/Real-ESRGAN

本模块将 Real-ESRGAN 封装为简洁的接口, 支持 4 倍放大超分辨率.

"创新点":
  1. 超分前使用双边滤波进行预处理降噪, 避免放大图像中的噪声
  2. 超分后配合 unsharp masking 锐化, 增强边缘清晰度

用法:
    from extensions.super_resolution import super_resolve
    result = super_resolve("input.jpg", denoise=True, sharpen=True)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import cv2
import numpy as np

# 模块级缓存：存放已转为纯 ASCII 路径的模型副本，避免 OpenCV DNN 中文路径打不开
_SAFE_MODEL_PATHS: dict = {}


def _ascii_safe_path(path: str) -> str:
    """若 path 含非 ASCII 字符则复制到系统临时目录，返回可安全传给 OpenCV 的路径。"""
    if path in _SAFE_MODEL_PATHS:
        return _SAFE_MODEL_PATHS[path]
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        import shutil
        import tempfile

        safe_dir = os.path.join(tempfile.gettempdir(), "img_enhancer_weights")
        os.makedirs(safe_dir, exist_ok=True)
        safe_path = os.path.join(safe_dir, os.path.basename(path))
        if not os.path.exists(safe_path):
            shutil.copy2(path, safe_path)
        _SAFE_MODEL_PATHS[path] = safe_path
        return safe_path


def _pre_denoise(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    超分前预处理: 双边滤波降噪.

    双边滤波在平滑噪声的同时保留边缘, 避免 ESRGAN 将噪声放大.

    Args:
        image:  输入 BGR 图像, (H, W, 3), uint8.
        strength: 去噪强度 (0 无效果, 1 默认, 2 强).

    Returns:
        去噪后的图像.
    """
    if strength <= 0:
        return image
    d = int(5 + 10 * min(strength, 2.0))
    sigma_color = min(30 + 40 * strength, 150)
    sigma_space = d
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def _post_sharpen(
    image: np.ndarray, amount: float = 0.3, radius: int = 1
) -> np.ndarray:
    """
    超分后处理: Unsharp Masking 锐化.

    Args:
        image:  超分后的 BGR 图像, uint8.
        amount: 锐化强度.
        radius: 高斯模糊半径.

    Returns:
        锐化后的图像.
    """
    if amount <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


def super_resolve(
    image_path: str,
    output_path: Optional[str] = None,
    denoise: bool = True,
    sharpen: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    对图像进行 4 倍超分辨率.

    Args:
        image_path: 输入图像路径.
        output_path: 可选的保存路径.
        denoise:     是否在超分前进行双边滤波降噪.
        sharpen:     是否在超分后进行 unsharp masking 锐化.
        device:      推理设备, "cuda" 或 "cpu".

    Returns:
        超分后的图像, BGR 格式, uint8, 尺寸为原图的 4 倍.
    """
    # ── 加载图像 ──
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = img.shape[:2]
    print(f"[SR] 输入: {w}×{h}")

    # ── 预处理: 去噪 ──
    if denoise:
        img = _pre_denoise(img)
        print("[SR] 预处理: 双边滤波降噪")

    # ── 加载 OpenCV DNN 超分模型 ──
    try:
        from cv2 import dnn_superres
    except ImportError:
        raise ImportError(
            "需要安装 opencv-contrib-python: pip install opencv-contrib-python"
        )

    import urllib.request

    print("[SR] 准备 EDSR 模型...")

    # 自动下载轻量级 EDSR_x4 模型
    model_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "EDSR_x4.pb")

    if not os.path.exists(model_path):
        print("[SR] 正在下载 EDSR_x4.pb 模型文件...")
        # EDSR_x4 约 38MB
        url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("[SR] 模型下载完成.")
        except Exception as e:
            raise RuntimeError(f"下载模型失败: {e}。请手动下载 {url} 放入 {model_dir}")

    # ── 超分推理 ──
    print("[SR] 超分推理中 (OpenCV DNN)...")
    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_ascii_safe_path(model_path))
        sr.setModel("edsr", 4)
        if device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[SR] 设备: CUDA backend")
        else:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[SR] 设备: CPU backend")

        sr_result = sr.upsample(img)
    except Exception as e:
        raise RuntimeError(f"超分辨率推理失败: {e}")

    print(f"[SR] 输出: {sr_result.shape[1]}×{sr_result.shape[0]} (4×)")

    # ── 后处理: 锐化 ──
    if sharpen:
        sr_result = _post_sharpen(sr_result)
        print("[SR] 后处理: Unsharp Masking 锐化")

    # ── 保存 ──
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, sr_result)
        print(f"[SR] 已保存: {output_path}")

    return sr_result
