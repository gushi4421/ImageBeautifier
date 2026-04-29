import base64
import inspect
import io
from typing import Callable, Dict, Any, List
import numpy as np
import cv2


class ImageProcessorAdapter:
    """
    图像处理适配器，负责统一管理图像处理算法，提供对外的统一调用接口
    """

    def __init__(self):
        # 操作注册表：将字符串动作名映射到具体的处理函数
        self._actions: Dict[str, Callable] = {}

    def register_action(self, action_name: str, func: Callable):
        """
        注册一个处理动作
        Args:
            action_name: 前端调用的动作名称
            func: 对应的后端处理函数
        """
        self._actions[action_name] = func

    def process(
        self, action_name: str, images: List[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        执行图像处理
        Args:
            action_name: 动作名称
            images: 输入的图像列表（支持多图输入，如拼接）
            kwargs: 其他动态参数（如 angle, kernel_size 等）
        Returns:
            处理后的图像 ndarray
        """
        if action_name not in self._actions:
            raise ValueError(f"未注册的操作: {action_name}")

        func = self._actions[action_name]

        if len(images) == 1:
            # 通过 inspect 检查函数签名, 判断是单图还是多图接口
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            first_param = params[0] if params else ""
            if first_param and first_param not in ("self", "cls"):
                hint = sig.parameters[first_param].annotation
                takes_list = (
                    hint is not inspect.Parameter.empty
                    and getattr(hint, "__origin__", None) is list
                )
                if takes_list:
                    return func(images, **kwargs)
            return func(images[0], **kwargs)
        else:
            return func(images, **kwargs)

    @staticmethod
    def base64_to_image(base64_str: str) -> np.ndarray:
        """
        将 Base64 字符串转换为 OpenCV BGR 格式的 ndarray
        """
        # 移除 base64 的前缀如 'data:image/jpeg;base64,'
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def image_to_base64(image_ext: str, image: np.ndarray) -> str:
        """
        将 ndarray 转换为 Base64 格式字符串供前端显示
        Args:
            image_ext: 编码格式，如 '.png' 或 '.jpg'
            image: 待编码的图片
        """
        success, buffer = cv2.imencode(image_ext, image)
        if not success:
            raise ValueError("图片编码失败")

        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime_type = image_ext.lstrip(".").lower()
        if mime_type == "jpg":
            mime_type = "jpeg"

        return f"data:image/{mime_type};base64,{b64_str}"
