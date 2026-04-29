"""
图像拼接模块 — UDIS++ 推理封装

本模块是对 UDIS++ (ICCV 2023) 的 Python API 封装，整合了两阶段推理:
  Stage 1 — Warp: 全局单应性 + 局部薄板样条(TPS)变形对齐
  Stage 2 — Composition: 可微分融合, 消除拼接缝和重影

原始实现: https://github.com/nie-lang/UDIS2
原始作者: Lang Nie, Chunyu Lin, Kang Liao, Shuaicheng Liu, Yao Zhao
论文:    Parallax-Tolerant Unsupervised Deep Image Stitching (ICCV 2023)
协议:    Apache License 2.0
原始代码: src/extensions/udis_stitching/ (未修改)

用法:
    from extensions.stitch import stitch_images
    result = stitch_images("img1.jpg", "img2.jpg", device="cuda")
"""

from __future__ import annotations

import os
import glob
import sys
import threading
import warnings
from typing import Optional

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────
# UDIS++ 代码与权重路径
# ──────────────────────────────────────────────
_UDIS_DIR = os.path.join(os.path.dirname(__file__), "udis_stitching")
_WARP_DIR = os.path.join(_UDIS_DIR, "Warp", "Codes")
_COMP_DIR = os.path.join(_UDIS_DIR, "Composition", "Codes")
_WARP_MODEL_DIR = os.path.join(_UDIS_DIR, "Warp", "model")
_COMP_MODEL_DIR = os.path.join(_UDIS_DIR, "Composition", "model")

# 保护 sys.modules 操作, 防止并发导入时模块命名空间损坏
_IMPORT_LOCK = threading.Lock()


# ──────────────────────────────────────────────
# 模块缓存
# ──────────────────────────────────────────────
class _Modules:
    warp_net = None
    comp_net = None
    grid_res = None


def _ensure_warp_modules():
    """
    导入 Warp 阶段模块.

    注意: UDIS++ 的 Warp/Codes/utils/ 与项目的 src/utils.py 存在命名冲突.
    使用锁保护 + 导入后立即还原 sys.modules, 避免污染项目的模块命名空间.
    """
    if _Modules.warp_net is not None:
        return

    with _IMPORT_LOCK:
        if _Modules.warp_net is not None:
            return
        # 保存可能冲突的模块
        saved = {}
        for name in ("utils", "network", "grid_res"):
            if name in sys.modules:
                saved[name] = sys.modules.pop(name)

        sys.path.insert(0, _WARP_DIR)
        try:
            import network as wn
        finally:
            sys.path.remove(_WARP_DIR)
            # 清理 UDIS 导入留下的模块条目
            for name in ("utils", "network", "grid_res"):
                sys.modules.pop(name, None)
            # 还原项目原本的模块
            for name, mod in saved.items():
                sys.modules[name] = mod

        _Modules.warp_net = wn


def _ensure_comp_modules():
    if _Modules.comp_net is not None:
        return

    with _IMPORT_LOCK:
        if _Modules.comp_net is not None:
            return
        saved = {}
        for name in ("utils", "network"):
            if name in sys.modules:
                saved[name] = sys.modules.pop(name)

        sys.path.insert(0, _COMP_DIR)
        try:
            import network as cn
        finally:
            sys.path.remove(_COMP_DIR)
            for name in ("utils", "network"):
                sys.modules.pop(name, None)
            for name, mod in saved.items():
                sys.modules[name] = mod

        _Modules.comp_net = cn


# ──────────────────────────────────────────────
# 权重加载
# ──────────────────────────────────────────────
def _load_checkpoint(model_dir: str, device: torch.device) -> dict:
    ckpt_list = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not ckpt_list:
        raise FileNotFoundError(
            f"未找到预训练权重文件.\n" f"请将 .pth 权重文件放入: {model_dir}"
        )
    ckpt_path = ckpt_list[-1]
    print(f"[UDIS] 加载权重: {os.path.basename(ckpt_path)}")
    return torch.load(ckpt_path, map_location="cpu")


def _load_warp_network(device: torch.device) -> torch.nn.Module:
    _ensure_warp_modules()
    net = _Modules.warp_net.Network()
    ckpt = _load_checkpoint(_WARP_MODEL_DIR, device)
    net.load_state_dict(ckpt["model"])
    net = net.to(device)
    net.eval()
    return net


def _load_comp_network(device: torch.device) -> torch.nn.Module:
    _ensure_comp_modules()
    net = _Modules.comp_net.Network()
    ckpt = _load_checkpoint(_COMP_MODEL_DIR, device)
    net.load_state_dict(ckpt["model"])
    net = net.to(device)
    net.eval()
    return net


# ──────────────────────────────────────────────
# 图像 -> Tensor
# ──────────────────────────────────────────────
def _imread_to_tensor(path: str, device: torch.device) -> torch.Tensor:
    """读取 BGR 图像, 归一化到 [-1, 1], 返回 (1, 3, H, W)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    t = torch.from_numpy(np.transpose(img, [2, 0, 1])).unsqueeze(0)
    return t.to(device)


# ──────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────
def stitch_images(
    img1_path: str,
    img2_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> np.ndarray:
    """
    将两张有重叠区域的图像拼接为全景图.

    Args:
        img1_path:  第一张图像的路径 (参考帧).
        img2_path:  第二张图像的路径 (待对齐帧).
        output_path: 可选的保存路径. 为 None 则不保存到磁盘.
        device:     推理设备, "cuda" 或 "cpu".

    Returns:
        拼接后的图像, ndarray 类型, BGR 格式, 形状为 (H', W', 3).
    """
    # ── 设备 ──
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA 不可用, 回退到 CPU")
        device = "cpu"
    dev = torch.device(device)

    # ── 加载网络与模块 ──
    _ensure_warp_modules()
    _ensure_comp_modules()
    warp_net = _load_warp_network(dev)
    comp_net = _load_comp_network(dev)

    # ── 读取图像 ──
    img1_t = _imread_to_tensor(img1_path, dev)
    img2_t = _imread_to_tensor(img2_path, dev)

    print(f"[UDIS] 输入: {img1_t.shape[3]}×{img1_t.shape[2]}")

    # ══════════════════════════════════════════
    # Stage 1 — Warp
    # ══════════════════════════════════════════
    print("[UDIS] Stage 1/2 — Warp ...")
    with torch.no_grad():
        out1 = _Modules.warp_net.build_output_model(warp_net, img1_t, img2_t)

    warp1 = out1["final_warp1"]  # (1,3,H',W'), [-1,1]
    warp2 = out1["final_warp2"]  # (1,3,H',W'), [-1,1]
    mask1 = out1["final_warp1_mask"]  # (1,3,H',W'), [0,1]
    mask2 = out1["final_warp2_mask"]  # (1,3,H',W'), [0,1]

    del warp_net, img1_t, img2_t
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════
    # Stage 2 — Composition
    # ══════════════════════════════════════════
    print("[UDIS] Stage 2/2 — Composition ...")
    with torch.no_grad():
        out2 = _Modules.comp_net.build_model(comp_net, warp1, warp2, mask1, mask2)

    stitched_t = out2["stitched_image"]  # (1,3,H',W'), [-1,1]

    del comp_net, warp1, warp2, mask1, mask2
    torch.cuda.empty_cache()

    # ── Tensor → numpy BGR ──
    result = (stitched_t[0] + 1) * 127.5  # [-1,1] → [0,255]
    result = result.cpu().numpy().transpose(1, 2, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    H_out, W_out = result.shape[:2]

    print(f"[UDIS] 完成! 输出: {W_out}×{H_out}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"[UDIS] 已保存: {output_path}")

    return result
