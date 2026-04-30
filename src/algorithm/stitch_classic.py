"""
本模块实现基于经典特征匹配的图像拼接 (SIFT + RANSAC + 单应性变换 + 加权融合).

采用 OpenCV 的 SIFT 提取特征, 其余步骤 (匹配筛选、RANSAC 求解单应性、
图像变换与融合) 均使用 NumPy 手写实现.

管线:
  1. SIFT 特征提取 (cv2)
  2. 暴力匹配 + Lowe 比率测试筛选
  3. RANSAC 求解单应性矩阵 H
  4. 逆向映射 + 双线性插值做单应性变换
  5. 加权融合消除拼接缝
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


def extract_sift_features(
    image: np.ndarray,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    提取图像的 SIFT 特征点与描述子.

    Args:
        image: 输入图像, BGR 格式, 形状为 (H, W, 3).

    Returns:
        包含两个元素的元组:
        1. keypoints: SIFT 关键点列表.
        2. descriptors: 对应的描述子矩阵, 形状为 (N, 128).
    """
    # ── 在这里实现你的 SIFT 特征提取 ──
    pass


def match_keypoints(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> list[tuple[int, int]]:
    """
    对两张图的 SIFT 描述子进行暴力匹配, 并用 Lowe 比率测试筛选优质匹配.

    Args:
        desc1: 第一张图的描述子, (N1, 128).
        desc2: 第二张图的描述子, (N2, 128).
        ratio_thresh: Lowe 比率阈值, 只有最近距离 / 次近距离 <= 此值时保留.

    Returns:
        筛选后的匹配点对索引列表, 每个元素为 (idx1, idx2).
    """
    # ── 在这里实现你的暴力匹配 + Lowe 比率测试 ──
    pass


def compute_homography_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransac_reproj_thresh: float = 5.0,
    ransac_max_iters: int = 2000,
    ransac_confidence: float = 0.995,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    使用 RANSAC 迭代求解两张图像间的单应性矩阵 H.

    每次迭代:
      1. 随机采样 4 对匹配点.
      2. 通过 DLT (Direct Linear Transform) 求解 H.
      3. 计算所有匹配点在 H 下的投影误差, 统计内点.
      4. 保留内点最多的 H.

    Args:
        pts1: 第一张图的匹配点坐标, (N, 2).
        pts2: 第二张图的匹配点坐标, (N, 2).
        ransac_reproj_thresh: 内点判断的投影误差阈值 (像素).
        ransac_max_iters: RANSAC 最大迭代次数.
        ransac_confidence: 期望置信度, 用于提前终止.

    Returns:
        包含两个元素的元组:
        1. H: 3x3 单应性矩阵 (成功时), 或 None (失败时).
        2. inlier_mask: 布尔数组, (N,), True 表示该匹配点是内点.
    """
    # ── 在这里实现你的 RANSAC + DLT 求解 ──
    pass


def warp_perspective(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    对图像执行单应性变换 (逆向映射 + 双线性插值).

    遍历输出图像的每个像素 (u, v):
      1. 通过 H_inv 将 (u, v) 映射回原图坐标 (x, y).
      2. 若 (x, y) 在原图范围内, 用双线性插值取色; 否则填 0.

    Args:
        image: 输入图像, BGR 格式, (H, W, 3).
        H: 3x3 单应性矩阵 (从 img2 到 img1 的变换).
        output_size: 输出图像尺寸 (out_w, out_h).

    Returns:
        变换后的图像, (out_h, out_w, 3), uint8 类型.
    """
    # ── 在这里实现你的逆向映射 + 双线性插值 ──
    # 提示: 可以参考 geometry.py 中 zoom 函数的双线性插值写法
    pass


def linear_blend(
    warp1: np.ndarray,
    warp2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    """
    对两张变形后的图像进行加权融合.

    在重叠区域, 使用按像素到重叠区边界的距离进行线性加权;
    在非重叠区域, 直接取对应图像的值.

    Args:
        warp1: 第一张变换后的图像, (H, W, 3), uint8.
        warp2: 第二张变换后的图像, (H, W, 3), uint8.
        mask1: 第一张的有效区域 mask, (H, W), 二值 (0/255).
        mask2: 第二张的有效区域 mask, (H, W), 二值 (0/255).

    Returns:
        融合后的图像, (H, W, 3), uint8.
    """
    # ── 在这里实现你的加权融合 ──
    pass


def stitch_images_classic(
    img1_path: str,
    img2_path: str,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    使用经典 SIFT + RANSAC 管线拼接两张图像.

    Args:
        img1_path: 第一张图像路径 (参考帧).
        img2_path: 第二张图像路径 (待对齐帧).
        output_path: 可选保存路径.

    Returns:
        拼接后的全景图, BGR 格式, (H, W, 3), uint8.
    """
    # ── 1. 加载图像 ──
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("图像加载失败")
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"[ClassicStitch] 加载图像: {w1}×{h1} + {w2}×{h2}")

    # ── 2. SIFT 特征提取 ──
    kp1, desc1 = extract_sift_features(img1)
    kp2, desc2 = extract_sift_features(img2)
    print(f"[ClassicStitch] 特征点: img1={len(kp1)}, img2={len(kp2)}")

    # ── 3. 特征匹配 ──
    matches = match_keypoints(desc1, desc2)
    print(f"[ClassicStitch] 匹配点: {len(matches)}")

    if len(matches) < 4:
        raise RuntimeError(
            f"匹配点不足 ({len(matches)} < 4), 无法计算单应性矩阵"
        )

    # ── 4. RANSAC 求解单应性矩阵 H ──
    pts1 = np.float32([kp1[m[0]].pt for m in matches])
    pts2 = np.float32([kp2[m[1]].pt for m in matches])
    H, inlier_mask = compute_homography_ransac(pts1, pts2)
    num_inliers = np.sum(inlier_mask)
    print(f"[ClassicStitch] RANSAC 内点: {num_inliers}/{len(matches)}")

    if H is None:
        raise RuntimeError("RANSAC 未能求解出有效的单应性矩阵")

    # ── 5. 计算输出画布尺寸 ──
    corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners_img2_transformed = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((corners_img1, corners_img2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    out_w = x_max - x_min
    out_h = y_max - y_min
    print(f"[ClassicStitch] 输出画布: {out_w}×{out_h}")

    # ── 6. 构造平移矩阵 T, 使画布偏移到正坐标 ──
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    H_offset = T @ H  # img2 → img1 坐标 → 画布坐标

    # ── 7. 变形 (warp) ──
    warp1 = warp_perspective(img1, T, (out_w, out_h))  # img1 直接平移
    warp2 = warp_perspective(img2, H_offset, (out_w, out_h))  # img2 单应性变换 + 平移

    # ── 8. 生成有效区域 mask ──
    ones1 = np.ones((h1, w1), dtype=np.uint8) * 255
    ones2 = np.ones((h2, w2), dtype=np.uint8) * 255
    mask1 = warp_perspective(ones1, T, (out_w, out_h))
    mask2 = warp_perspective(ones2, H_offset, (out_w, out_h))

    # ── 9. 融合 ──
    result = linear_blend(warp1, warp2, mask1, mask2)

    # ── 10. 保存 ──
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"[ClassicStitch] 已保存: {output_path}")

    print(f"[ClassicStitch] 完成! 输出尺寸: {result.shape[1]}×{result.shape[0]}")
    return result


if __name__ == "__main__":
    # 简单测试
    import sys
    result = stitch_images_classic(
        "data/stitching/carpark_01.jpg",
        "data/stitching/carpark_02.jpg",
        output_path="test/classic_stitch_result.jpg",
    )
