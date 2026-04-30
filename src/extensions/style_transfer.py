"""
风格迁移模块 — 基于 Gatys et al. (CVPR 2016)

"A Neural Algorithm of Artistic Style"
论文: https://arxiv.org/abs/1508.06576

核心思想:
  1. 用 VGG-19 提取内容图像的内容特征 (第 conv4_2 层)
  2. 用 VGG-19 提取风格图像的风格特征 (多层的 Gram 矩阵)
  3. 用 L-BFGS 优化一张目标图像, 使其特征同时逼近内容特征和风格特征

用法:
    from extensions.style_transfer import stylize
    stylize("photo.jpg", "style.jpg", "output.jpg")
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T

# ── 预训练的 VGG-19 (只用到卷积层) ──
_vgg: Optional[nn.Sequential] = None
_device: Optional[torch.device] = None


def _get_vgg(device: torch.device) -> nn.Sequential:
    global _vgg
    if _vgg is None:
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg19 = vgg19.to(device).eval()
        for p in vgg19.parameters():
            p.requires_grad = False
        # 将 max_pool 改为 avg_pool, 效果更平滑
        layers = []
        for m in vgg19.children():
            if isinstance(m, nn.MaxPool2d):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(m)
        _vgg = nn.Sequential(*layers)
    return _vgg


# ── 内容/风格层配置 ──
CONTENT_LAYERS = {"conv4": 21}  # VGG-19 conv4_2 的索引
STYLE_LAYERS = {           # VGG-19 中风格层的索引
    "conv1": 1,   # conv1_1
    "conv2": 6,   # conv2_1
    "conv3": 11,  # conv3_1
    "conv4": 20,  # conv4_1
    "conv5": 29,  # conv5_1
}

STYLE_WEIGHTS = {"conv1": 1.0, "conv2": 0.8, "conv3": 0.6, "conv4": 0.4, "conv5": 0.2}


def _gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """计算特征图的 Gram 矩阵 (风格表示)."""
    b, c, h, w = feature_map.shape
    features = feature_map.view(c, h * w)
    gram = torch.mm(features, features.t()) / (c * h * w)
    return gram


def _preprocess(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 → 归一化 Tensor (1, 3, H, W)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img_rgb).unsqueeze(0).to(device)


def _postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Tensor → BGR uint8."""
    img = tensor[0].clone().detach().cpu().clamp_(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def stylize(
    content_path: str,
    style_path: str,
    output_path: str = "stylized.jpg",
    device: str = "cuda",
    num_steps: int = 500,
    content_weight: float = 1.0,
    style_weight: float = 1e6,
    tv_weight: float = 1e-3,
) -> np.ndarray:
    """
    将内容图像转换为指定风格图像的艺术风格.

    Args:
        content_path:   内容图像路径 (你照的那张).
        style_path:     风格图像路径 (梵高星空 / 浮世绘 / 莫奈).
        output_path:    输出保存路径.
        device:         "cuda" 或 "cpu".
        num_steps:      优化迭代次数 (100~1000, 越多效果越好, 越慢).
        content_weight: 内容保真度权重.
        style_weight:   风格匹配度权重.
        tv_weight:      平滑度正则权重.

    Returns:
        风格迁移结果, BGR 格式, uint8.
    """
    # ── 设备 ──
    global _device
    if device == "cuda" and not torch.cuda.is_available():
        print("[NST] CUDA 不可用, 回退到 CPU")
        device = "cpu"
    _device = dev = torch.device(device)

    # ── 加载图像 ──
    content_img = cv2.imread(content_path)
    style_img = cv2.imread(style_path)
    if content_img is None or style_img is None:
        raise FileNotFoundError("图像加载失败")
    print(f"[NST] 内容: {content_path}")
    print(f"[NST] 风格: {style_path}")
    print(f"[NST] 设备: {device.upper()}, 迭代: {num_steps}")

    content_tensor = _preprocess(content_img, dev)
    style_tensor = _preprocess(style_img, dev)

    # ── 加载 VGG-19 ──
    vgg = _get_vgg(dev)

    # ── 提取内容目标 ──
    content_targets = {}
    x = content_tensor
    for name, idx in CONTENT_LAYERS.items():
        x = vgg[: idx + 1](x)
        content_targets[name] = x.clone()

    # ── 提取风格目标 (Gram 矩阵) ──
    style_targets = {}
    x = style_tensor
    for name, idx in STYLE_LAYERS.items():
        x = vgg[: idx + 1](x)
        style_targets[name] = _gram_matrix(x)

    # ── 初始化优化目标 ──
    target = content_tensor.clone().requires_grad_(True)
    optimizer = optim.LBFGS([target], max_iter=1, line_search_fn="strong_wolfe")

    print("[NST] 优化中... (每 100 步打印一次)")

    # ── 优化循环 ──
    step = [0]

    def closure():
        optimizer.zero_grad()
        x = target
        total_loss = 0.0

        # 内容损失 (conv4_2)
        for name, idx in CONTENT_LAYERS.items():
            x_feat = vgg[: idx + 1](x)
            total_loss += content_weight * nn.MSELoss()(x_feat, content_targets[name])

        # 风格损失 (Gram 矩阵)
        for name, idx in STYLE_LAYERS.items():
            x_feat = vgg[: idx + 1](x)
            gram_x = _gram_matrix(x_feat)
            loss = nn.MSELoss()(gram_x, style_targets[name])
            total_loss += style_weight * STYLE_WEIGHTS[name] * loss

        # 全变分损失 (平滑正则)
        tv_loss = (
            torch.sum(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]))
            + torch.sum(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]))
        )
        total_loss += tv_weight * tv_loss

        total_loss.backward()
        step[0] += 1
        if step[0] % 100 == 0:
            print(f"[NST] Step {step[0]}/{num_steps}, loss={total_loss.item():.2f}")
        return total_loss

    for _ in range(num_steps):
        optimizer.step(closure)

    # ── 后处理 ──
    result = _postprocess(target)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"[NST] 完成! 已保存: {output_path}")

    return result
