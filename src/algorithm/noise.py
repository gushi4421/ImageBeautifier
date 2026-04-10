"""
该模块用于实现对图像的加噪和去噪处理.
支持高斯噪声、椒盐噪声的生成, 以及均值滤波和中值滤波算法.
"""

import numpy as np


# 1. 加噪逻辑层
def add_noise(image: np.ndarray, mode: str = "gaussian", **kwargs) -> np.ndarray:
    """
    为图像添加特定类型的噪声.

    Args:
        image: 输入的原始图像.
        mode: 噪声模式, 支持 "gaussian" 和 "salt_pepper".
        kwargs: 噪声参数 (mean, sigma 或 prob).
    """
    support_types = ["gaussian", "salt_pepper"]
    if mode not in support_types:
        raise ValueError("不支持的噪声类型")

    if mode == "gaussian":
        # 获取高斯分布参数, 默认为均值0, 标准差20
        mean = kwargs.get("mean", 0.0)
        sigma = kwargs.get("sigma", 20.0)
        # 生成与图像尺度一致的高斯随机矩阵
        noise = np.random.normal(mean, sigma, image.shape)
        # 叠加噪声并进行溢出处理
        noisy_image = image.astype(np.float64) + noise
        return np.clip(noisy_image, 0, 255).astype(image.dtype)
        
    elif mode == "salt_pepper":
        prob = kwargs.get("prob", 0.1)
        if prob > 1.0:
            raise ValueError("噪声比例不能大于1.0")
            
        noise_image = image.copy()
        h, w, _ = image.shape
        # 生成随机概率矩阵, 用于决定哪些位置加噪
        random_matrix = np.random.rand(h, w)

        # 比例的一半设为椒噪声(黑色 0)
        noise_image[random_matrix < (prob / 2.0)] = [0, 0, 0]
        # 比例的另一半设为盐噪声(白色 255)
        # 修正原代码逻辑符号错误: 使用 = 赋值而非 - 运算
        noise_image[(random_matrix >= (prob / 2.0)) & (random_matrix < prob)] = [255, 255, 255]
        
        return noise_image


# 2. 去噪/滤波逻辑层
def remove_noise(
    image: np.ndarray, mode: str = "mean", kernal_size: int = 3
) -> np.ndarray:
    """
    空间域滤波函数.
    支持均值滤波(Mean Filter)和中值滤波(Median Filter).
    """
    if mode not in ["mean", "median"]:
        raise ValueError("不支持的滤波模式")
    if kernal_size % 2 == 0:
        raise ValueError("卷积核大小必须为奇数")

    h, w, c = image.shape
    # 计算填充大小, 保证输出图像尺寸不变
    pad_size = kernal_size // 2
    # 使用镜像填充(reflect)处理边缘像素
    padded_image = np.pad(
        image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="reflect"
    )
    
    filtered_image = np.zeros_like(image)
    
    # 空间卷积/滑动窗口操作
    for u in range(w):
        for v in range(h):
            # 切片提取当前窗口内的像素块
            window = padded_image[v : v + kernal_size, u : u + kernal_size, :]
            
            if mode == "mean":
                # 均值滤波: 计算窗口内所有像素的平均值
                filtered_image[v, u] = window.mean(axis=(0, 1)).astype(np.uint8)
            elif mode == "median":
                # 中值滤波: 提取中间值, 对椒盐噪声有极强的抑制作用
                filtered_image[v, u] = np.median(window, axis=(0, 1)).astype(np.uint8)
                
    return filtered_image


def better_salt_pepper_noise(image: np.ndarray, prob: float):
    """
    优化版的椒盐噪声生成, 通过随机坐标索引提高性能.
    """
    if prob > 1.0:
        raise ValueError("噪声比例不能大于1.0")
    noise_image = image.copy()
    h, w, _ = image.shape
    
    # 计算需要改变的像素总数
    total_pixels = int(h * w * prob)
    pepper_pixels = total_pixels // 2
    salt_pixels = total_pixels - pepper_pixels

    # 随机生成椒噪声坐标
    pepper_y = np.random.randint(0, h, pepper_pixels)
    pepper_x = np.random.randint(0, w, pepper_pixels)
    noise_image[pepper_y, pepper_x] = [0, 0, 0]

    # 随机生成盐噪声坐标
    salt_y = np.random.randint(0, h, salt_pixels)
    salt_x = np.random.randint(0, w, salt_pixels)
    noise_image[salt_y, salt_x] = [255, 255, 255]
    
    return noise_image