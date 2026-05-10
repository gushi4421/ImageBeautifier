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
    """提取图像的 SIFT 特征点与描述子."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_keypoints(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> list[tuple[int, int]]:
    """暴力匹配 + Lowe 比率测试筛选优质匹配."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance <= ratio_thresh * n.distance:
            good_matches.append((m.queryIdx, m.trainIdx))

    return good_matches


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对点集做归一化, 使均值为0、平均距离为 sqrt(2), 提升 DLT 数值稳定性."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = math.sqrt(2) / mean_dist
    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def _dlt_homography(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """用 DLT 从 4 对匹配点计算 3x3 单应性矩阵."""
    A = np.zeros((8, 9), dtype=np.float64)
    for i in range(4):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H / H[2, 2]


def compute_homography_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransac_reproj_thresh: float = 5.0,
    ransac_max_iters: int = 2000,
    ransac_confidence: float = 0.995,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """使用 RANSAC 迭代求解两张图像间的单应性矩阵 H."""
    N = pts1.shape[0]
    if N < 4:
        return None, np.zeros(N, dtype=bool)

    # 归一化提升数值稳定性
    pts1_norm, T1 = _normalize_points(pts1)
    pts2_norm, T2 = _normalize_points(pts2)

    best_inlier_mask = np.zeros(N, dtype=bool)
    best_H = None
    max_inliers = 0
    iter_count = 0

    while iter_count < ransac_max_iters:
        # 随机采样 4 对匹配点
        sample_idx = np.random.choice(N, 4, replace=False)

        # DLT 计算 H (在归一化坐标系)
        H_norm = _dlt_homography(pts1_norm[sample_idx], pts2_norm[sample_idx])
        if H_norm is None:
            iter_count += 1
            continue

        # 反归一化
        H = np.linalg.inv(T2) @ H_norm @ T1
        H /= H[2, 2]

        # 统计内点: 计算所有点在 H 下的投影误差
        ones = np.ones((N, 1), dtype=np.float64)
        pts1_h = np.hstack([pts1, ones])

        proj = (H @ pts1_h.T).T
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]

        errors = np.sqrt(np.sum((proj[:, :2] - pts2) ** 2, axis=1))
        inlier_mask = errors < ransac_reproj_thresh
        num_inliers = np.sum(inlier_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_H = H

            # 自适应迭代次数
            inlier_ratio = num_inliers / N
            if inlier_ratio > 0:
                new_max_iters = math.log(1 - ransac_confidence) / math.log(
                    1 - inlier_ratio**4
                )
                ransac_max_iters = min(ransac_max_iters, int(new_max_iters) + 1)

        iter_count += 1

    # 用所有内点重新优化 H
    if best_H is not None and max_inliers >= 4:
        inlier_pts1 = pts1[best_inlier_mask]
        inlier_pts2 = pts2[best_inlier_mask]
        inlier_pts1_norm, T1_refit = _normalize_points(inlier_pts1)
        inlier_pts2_norm, T2_refit = _normalize_points(inlier_pts2)
        H_refit_norm = _dlt_homography(inlier_pts1_norm[:4], inlier_pts2_norm[:4])
        # 用所有内点通过最小二乘法重算: A h = 0 超定方程
        M = inlier_pts1_norm.shape[0]
        A = np.zeros((2 * M, 9), dtype=np.float64)
        for i in range(M):
            x1, y1 = inlier_pts1_norm[i]
            x2, y2 = inlier_pts2_norm[i]
            A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
            A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]
        H_refit_norm = h.reshape(3, 3)
        H_refit_norm /= H_refit_norm[2, 2]
        best_H = np.linalg.inv(T2_refit) @ H_refit_norm @ T1_refit
        best_H /= best_H[2, 2]

    return best_H, best_inlier_mask


def warp_perspective(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """逆向映射 + 双线性插值实现单应性变换."""
    h, w = image.shape[:2]
    out_w, out_h = output_size
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels == 1:
        result = np.zeros((out_h, out_w), dtype=np.uint8)
    else:
        result = np.zeros((out_h, out_w, channels), dtype=np.uint8)

    H_inv = np.linalg.inv(H)

    for u in range(out_w):
        for v in range(out_h):
            # 逆向映射: 输出坐标 -> 原图坐标
            vec = H_inv @ np.array([u, v, 1.0], dtype=np.float64)
            x = vec[0] / vec[2]
            y = vec[1] / vec[2]

            if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
                continue

            x1 = int(math.floor(x))
            y1 = int(math.floor(y))
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)

            dx = x - x1
            dy = y - y1

            # 双线性插值
            if channels == 1:
                p11 = float(image[y1, x1])
                p12 = float(image[y2, x1])
                p21 = float(image[y1, x2])
                p22 = float(image[y2, x2])
            else:
                p11 = image[y1, x1].astype(np.float64)
                p12 = image[y2, x1].astype(np.float64)
                p21 = image[y1, x2].astype(np.float64)
                p22 = image[y2, x2].astype(np.float64)

            r1 = p11 * (1 - dx) + p21 * dx
            r2 = p12 * (1 - dx) + p22 * dx
            p = (1 - dy) * r1 + dy * r2

            result[v, u] = np.clip(p, 0, 255).astype(np.uint8)

    return result


def linear_blend(
    warp1: np.ndarray,
    warp2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    """按像素到重叠区边界的距离进行加权融合."""
    out_h, out_w = warp1.shape[:2]
    result = np.zeros_like(warp1)

    bool1 = mask1 > 0
    bool2 = mask2 > 0

    # 距离变换: 有效区域内像素到最近无效像素的距离
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)

    # 防止除零
    sum_dist = dist1 + dist2
    sum_dist[sum_dist == 0] = 1.0

    w1 = dist1 / sum_dist
    w2 = dist2 / sum_dist

    for c in range(3):
        result[:, :, c] = (
            warp1[:, :, c].astype(np.float64) * w1
            + warp2[:, :, c].astype(np.float64) * w2
        )

    # 单图区域直接取对应图像的值
    only1 = bool1 & ~bool2
    only2 = bool2 & ~bool1
    result[only1] = warp1[only1]
    result[only2] = warp2[only2]

    return np.clip(result, 0, 255).astype(np.uint8)


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
        raise RuntimeError(f"匹配点不足 ({len(matches)} < 4), 无法计算单应性矩阵")

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
