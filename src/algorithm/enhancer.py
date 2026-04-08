"""
该模块实现对图像的增强
"""
import numpy as np

def adjust_contrast(image:np.ndarray,alpha:float=1.0):
    adjusted_image=image.astype(np.float32)
    adjusted_image=adjusted_image*alpha
    return np.clip(adjusted_image,0,255).astype(np.uint8)

def adjust_brightness(image:np.ndarray,beta:int =0):
    adjusted_image=image.astype(np.float32)
    adjusted_image=adjusted_image+beta
    return np.clip(adjusted_image,0,255).astype(np.uint8)
