"""
该模块实现像素级点操作.
包含灰度化、二值化(含OTSU)、伪彩色映射、假彩色增强、对比度调节、亮度调节、饱和度调节以及直方图均衡化.
"""

import numpy as np

from src.utils import generate_colormap_lut


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    将 BGR 图像转为单通道灰度图.

    Args:
        image: 输入的图像数据, 类型为 np.ndarray, 形状通常为 (H, W, 3).

    Returns:
        转换后的单通道灰度图像, 形状为 (H, W), 数据类型为 uint8.
    """
    # 获取图像尺寸,虽然此处未直接使用 h, w, 但常用于预处理检查
    h, w, _ = image.shape

    # 定义加权权重,对应 OpenCV 的 BGR 顺序权重
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)

    # 通过矩阵点积快速计算全图灰度值
    gray_image = np.dot(image, weights)

    # 确保数值在 0-255 之间并转换数据类型
    return np.clip(gray_image, 0, 255).astype(np.uint8)


def binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    对图像进行简单固定阈值二值化处理.
    若输入为彩色图, 自动转为灰度图再处理.

    Args:
        image: 输入的灰度图像或 BGR 彩色图像.
        threshold: 设定的分割阈值, 默认为 127.

    Returns:
        二值化后的图像, 像素值仅包含 0 和 255.
    """
    if image.ndim == 3:
        image = to_grayscale(image)
    binary_image = np.zeros_like(image)
    # 大于阈值的像素点设置为白色(255)
    binary_image[image > threshold] = 255
    return binary_image


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """
    OTSU自动阈值二值化算法实现.
    若输入为彩色图, 自动转为灰度图再处理.

    Args:
        image: 输入的待处理灰度图像或 BGR 彩色图像.

    Returns:
        使用自动阈值分割后的二值图像.
    """
    if image.ndim == 3:
        image = to_grayscale(image)
    total_pixels = image.size
    histogram = np.bincount(image.ravel(), minlength=256)
    probabilities = histogram / total_pixels
    intensity_levels = np.arange(256)
    sum_mean = np.sum(probabilities * intensity_levels)

    background_weight = 0.0
    background_sum = 0.0
    best_threshold = 0
    max_variance = 0.0

    for t in range(256):
        background_weight += probabilities[t]
        if background_weight == 0.0:
            continue
        if background_weight >= 1.0:
            break

        background_sum += probabilities[t] * intensity_levels[t]
        background_mean = background_sum / background_weight
        foreground_weight = 1 - background_weight
        foreground_mean = (sum_mean - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    binary_image = np.zeros_like(image)
    binary_image[image > best_threshold] = 255
    return binary_image


def apply_colormap(image: np.ndarray):
    """
    应用查找表进行图像伪彩色映射。
    """
    lut = generate_colormap_lut()
    color_image = lut[image]
    return color_image


def adjust_contrast(image: np.ndarray, alpha: float = 1.0):
    """
    调整图像对比度.

    Args:
        image: 输入图像.
        alpha: 对比度缩放因子, 大于 1.0 增强对比度, 小于 1.0 降低对比度.

    Returns:
        调整对比度后的图像.
    """
    # 转为浮点型防止计算溢出
    adjusted_image = image.astype(np.float32)
    # 线性变换: f(x) = alpha * x
    adjusted_image = adjusted_image * alpha
    # 截断处理并转回 uint8 类型
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def adjust_brightness(image: np.ndarray, beta: int = 0):
    """
    调整图像亮度.
    Args:
        image: 输入图像.
        beta: 亮度偏移量, 正值增加亮度, 负值降低亮度.

    Returns:
        调整亮度后的图像.
    """
    adjusted_image = image.astype(np.float32)
    # 线性变换: f(x) = x + beta
    adjusted_image = adjusted_image + beta
    # 确保数值合法性
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def histogram(image: np.ndarray) -> np.ndarray:
    """
    直方图均衡化实现.

    Args:
        image: 输入的低对比度灰度图像.

    Returns:
        亮度分布均匀化后的增强图像.
    """
    total_pixels = image.size
    # 统计每个灰度级(0-255)出现的频次
    hist_counts = np.bincount(image.ravel(), minlength=256)

    # 计算概率密度函数 (PDF: Probability Density Function)
    pdf = hist_counts / total_pixels

    # 计算累积分布函数 (CDF: Cumulative Distribution Function)
    cdf = np.cumsum(pdf)

    # 生成查找表 (LUT): 将 CDF 映射到 0-255 区间并取整
    lut = np.round(cdf * 255).astype(np.uint8)

    # 根据查找表重映射图像像素值
    color_image = lut[image]
    return color_image


def adjust_saturation(image: np.ndarray, alpha: float) -> np.ndarray:
    """
    使用线性插值调整图像饱和度.

    核心公式: output = input + alpha * (input - gray)
    - alpha > 0: 增大色彩偏离灰度中心的幅度, 提升饱和度.
    - alpha < 0: 减小色彩偏离幅度, 降低饱和度.
    - alpha = 0: 饱和度不变.

    Args:
        image: 输入的 BGR 三通道彩色图像.
        alpha: 饱和度调节因子, 通常取值范围在 -1 到 1 之间.

    Returns:
        调整饱和度后的图像.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("饱和度调整仅支持 BGR 三通道彩色图像")

    # 转为浮点防止计算溢出
    image_float = image.astype(np.float32)
    # 计算灰度图作为色彩中性基准
    gray_image = to_grayscale(image_float)
    # 扩展维度至 (H, W, 1) 以便与 (H, W, 3) 的彩色图逐通道广播
    gray_image = np.expand_dims(gray_image, axis=2)
    # 线性调整: 以灰度值为基准, 缩放色彩偏离量
    adjusted_image = image_float + alpha * (image_float - gray_image)

    return np.clip(adjusted_image, 0, 255).astype(np.uint8)


def adjust_sharpness(
    image: np.ndarray, amount: float = 1.0, radius: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """
    使用 Unsharp Masking (USM) 算法调整图像清晰度.

    原理: output = image + amount * (image - blurred)
    - 先对原图做高斯模糊得到平滑层
    - 原图减去平滑层得到细节层 (高频信息)
    - 将细节层按 amount 强度叠加回原图

    Args:
        image: 输入图像, BGR 三通道.
        amount: 锐化强度, 0 无变化, 1.0 中等, 更大值更锐利.
        radius: 高斯模糊核大小, 控制细节提取尺度, 建议 3~7.
        sigma: 高斯模糊标准差, 越大细节提取越粗.

    Returns:
        锐化后的图像, uint8 类型.
    """
    from src.algorithm.filter import gaussian_blurring

    image_float = image.astype(np.float32)
    blurred = gaussian_blurring(image, kernel_size=radius, sigma=sigma).astype(
        np.float32
    )
    sharpened = image_float + amount * (image_float - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def synthesize_false_color_image(bands_list: list[np.ndarray], r_index: int, g_index: int, b_index: int) -> np.ndarray:
    """
    通过将特定的多光谱灰度波段绑定到 RGB 通道，合成假彩色图像.
    
    Args:
        bands_list: 输入的单通道灰度波段列表. 每个波段形状为 (H, W), 类型 uint8.
                    假设列表的索引 0 代表红光(Red), 1 代表绿光(Green), 2 代表蓝光(Blue), 
                    3 代表近红外(NIR), 4 代表短波红外(SWIR) 等.
        r_index: 映射到 R 输出通道的波段列表索引.
        g_index: 映射到 G 输出通道的波段列表索引.
        b_index: 映射到 B 输出通道的波段列表索引.
        
    Returns:
        假彩色增强后的 RGB 图像, 形状为 (H, W, 3), 类型 uint8.
    """
    if len(bands_list) < 3:
        raise ValueError("合成假彩色至少需要 3 个输入波段.")

    num_bands = len(bands_list)
    if not (0 <= r_index < num_bands and 0 <= g_index < num_bands and 0 <= b_index < num_bands):
        raise ValueError("指定的波段索引越界.")

    h, w = bands_list[0].shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 将指定的灰度波段分别赋值到 RGB 三个通道
    color_image[:, :, 0] = bands_list[r_index]
    color_image[:, :, 1] = bands_list[g_index]
    color_image[:, :, 2] = bands_list[b_index]

    return color_image


def false_color_channel_swap(
    image: np.ndarray,
    r_src: int = 2,
    g_src: int = 0,
    b_src: int = 1,
) -> np.ndarray:
    """
    通道置换假彩色增强.

    将 BGR 图像的三通道拆分后按指定顺序重新排列到 RGB, 产生非自然但能突出
    特定信息的假彩色效果. 默认将 B 通道映射到 G、G 通道映射到 R、R 通道映射到 B.

    Args:
        image: 输入的 BGR 三通道彩色图像.
        r_src: 新 R 通道对应原图的通道索引 (0=B, 1=G, 2=R), 默认 2.
        g_src: 新 G 通道对应原图的通道索引, 默认 0.
        b_src: 新 B 通道对应原图的通道索引, 默认 1.

    Returns:
        假彩色增强后的 RGB 图像, 类型 uint8.
    """
    # 拆分 BGR 三个通道作为波段列表
    bands = [image[:, :, i] for i in range(3)]
    return synthesize_false_color_image(bands, r_src, g_src, b_src)