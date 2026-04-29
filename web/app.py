import sys
import os

# 将项目根目录加入到sys.path中，以解决找不到src模块的问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template
import logging

# 导入适配器
from src.adapters.processor_adapter import ImageProcessorAdapter

# 导入算法模块
from src.algorithm import filter as filter_algo
from src.algorithm import geometry as geometry_algo
from src.algorithm import noise as noise_algo
from src.algorithm import tone as tone_algo
from src.algorithm import artistic as artistic_algo
from src.algorithm import frequency as frequency_algo
from src.extensions import stitch as stitch_algo

import cv2
import tempfile
import uuid
import numpy as np

app = Flask(__name__, template_folder="../templates", static_folder="../static")
logging.basicConfig(level=logging.INFO)

# 初始化适配器并注册功能
processor = ImageProcessorAdapter()

# === 注册图像处理算法 ===
# filter.py
processor.register_action("mean_filter", filter_algo.mean_filter)
processor.register_action("median_filter", filter_algo.median_filter)
processor.register_action("gaussian_blurring", filter_algo.gaussian_blurring)
processor.register_action(
    "bilateral_filter",
    lambda img, **kwargs: filter_algo.bilateral_filter_manual(
        img,
        kernel_size=kwargs.get("kernel_size", 3),
        sigma_s=kwargs.get("sigma_s", 15.0),
        sigma_r=kwargs.get("sigma_r", 30.0),
    ),
)
# geometry.py
processor.register_action("translate", geometry_algo.translate)
processor.register_action("rotate", geometry_algo.rotate)
processor.register_action("zoom", geometry_algo.zoom)
processor.register_action("flip", geometry_algo.flip)
processor.register_action(
    "shear",
    lambda img, **kwargs: geometry_algo.shear(
        img,
        start_x=int(img.shape[1] * kwargs.get("start_x_ratio", 0) / 100.0),
        end_x=int(img.shape[1] * (100 - kwargs.get("end_x_ratio", 0)) / 100.0),
        start_y=int(img.shape[0] * kwargs.get("start_y_ratio", 0) / 100.0),
        end_y=int(img.shape[0] * (100 - kwargs.get("end_y_ratio", 0)) / 100.0),
    ),
)


def wrap_add_border(imgs, **kwargs):
    if len(imgs) < 2:
        return imgs[0]
    return geometry_algo.add_border(imgs[0], imgs[1], scale=kwargs.get("scale", 0.1))


processor.register_action("add_border", wrap_add_border)

# tone.py
processor.register_action("to_grayscale", tone_algo.to_grayscale)
processor.register_action("otsu_binarize", tone_algo.otsu_binarize)
processor.register_action("binarize", tone_algo.binarize)
processor.register_action("adjust_contrast", tone_algo.adjust_contrast)
processor.register_action("adjust_brightness", tone_algo.adjust_brightness)
processor.register_action("adjust_saturation", tone_algo.adjust_saturation)
processor.register_action("histogram_eq", tone_algo.histogram)
processor.register_action("adjust_sharpness", tone_algo.adjust_sharpness)
# processor.register_action("intelligent_fill_light", tone_algo.intelligent_fill_light)
# processor.register_action("adjust_highlight", tone_algo.adjust_highlight)
processor.register_action("apply_colormap", tone_algo.apply_colormap)
processor.register_action(
    "false_color_channel_swap",
    lambda img, **kwargs: tone_algo.false_color_channel_swap(
        img,
        r_src=kwargs.get("r_src", 2),
        g_src=kwargs.get("g_src", 0),
        b_src=kwargs.get("b_src", 1),
    ),
)


def wrap_synthesize_false_color(imgs, **kwargs):
    # This requires 3 images exactly
    if len(imgs) < 3:
        return imgs[0] if len(imgs) > 0 else np.zeros((10, 10, 3), dtype=np.uint8)
    bands = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs[:3]]
    return tone_algo.synthesize_false_color_image(
        bands, kwargs.get("r_src", 0), kwargs.get("g_src", 1), kwargs.get("b_src", 2)
    )


processor.register_action("synthesize_false_color_image", wrap_synthesize_false_color)
# artistic.py
processor.register_action("emboss", artistic_algo.emboss)
processor.register_action("frosted", artistic_algo.frosted)
# noise.py
processor.register_action(
    "add_noise_gaussian",
    lambda img, **kwargs: noise_algo.add_noise(img, mode="gaussian", **kwargs),
)
processor.register_action(
    "add_noise_salt_pepper",
    lambda img, **kwargs: noise_algo.add_noise(img, mode="salt_pepper", **kwargs),
)
# frequency.py (Using a wrapper to avoid missing positional parameters)
processor.register_action(
    "lowpass_filter",
    lambda img, **kwargs: frequency_algo.lowpass_filter(
        img, cutoff=kwargs.get("cutoff", 0.2), mode=kwargs.get("mode", "gaussian")
    ),
)
processor.register_action(
    "highpass_filter",
    lambda img, **kwargs: frequency_algo.highpass_filter(
        img, cutoff=kwargs.get("cutoff", 0.1), mode=kwargs.get("mode", "gaussian")
    ),
)
processor.register_action(
    "bandpass_filter",
    lambda img, **kwargs: frequency_algo.bandpass_filter(
        img,
        low_cut=kwargs.get("low_cut", 0.1),
        high_cut=kwargs.get("high_cut", 0.4),
        mode=kwargs.get("mode", "gaussian"),
    ),
)
processor.register_action(
    "bandreject_filter",
    lambda img, **kwargs: frequency_algo.bandreject_filter(
        img,
        low_cut=kwargs.get("low_cut", 0.1),
        high_cut=kwargs.get("high_cut", 0.4),
        mode=kwargs.get("mode", "gaussian"),
    ),
)


def wrap_stitch(imgs, **kwargs):
    if len(imgs) < 2:
        return imgs[0]
    import tempfile
    import cv2
    import os
    import numpy as np

    with tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False
    ) as f1, tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
        img1_path = f1.name
        img2_path = f2.name
        cv2.imwrite(img1_path, cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(img2_path, cv2.cvtColor(imgs[1], cv2.COLOR_RGB2BGR))

    try:
        # Note: stitch_images requires an output path in its signature?
        # stitch_images(img1_path: str, img2_path: str, output_path: Optional[str] = None) -> np.ndarray
        result = stitch_algo.stitch_images(img1_path, img2_path)
    finally:
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)

    return result


processor.register_action("stitch_images", wrap_stitch)
processor.register_action(
    "horizontal_collage",
    lambda imgs, **kwargs: geometry_algo.horizontal_collage(
        imgs[0], imgs[1], gap=kwargs.get("gap", 0)
    ),
)
processor.register_action(
    "vertical_collage",
    lambda imgs, **kwargs: geometry_algo.vertical_collage(
        imgs[0], imgs[1], gap=kwargs.get("gap", 0)
    ),
)


@app.route("/")
def index():
    """渲染前端主页面"""
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_image():
    """
    统一的图像处理接口
    接收前端传来的 JSON 数据:
    {
        "action": "mean_filter",
        "images": ["data:image/png;base64,..."],  # Base64数组，支持单图或多图
        "params": {
            "kernel_size": 5
        }
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "无效的请求数据"}), 400

        action = data.get("action")
        b64_images = data.get("images", [])
        params = data.get("params", {})

        if not action or not b64_images:
            return (
                jsonify(
                    {"success": False, "error": "缺少必要的 action 或 images 参数"}
                ),
                400,
            )

        # 解码 Base64 图片 -> numpy array
        np_images = [processor.base64_to_image(b64) for b64 in b64_images]

        # 调用适配器处理图片
        # 注意: 这里的参数会被动态解包，传递到底层注册的函数中
        result_img = processor.process(action, np_images, **params)

        # 把结果转回 Base64，默认传回 png 格式避免二次压缩带来的问题
        result_b64 = processor.image_to_base64(".png", result_img)

        return jsonify({"success": True, "result_image": result_b64})

    except Exception as e:
        logging.error(f"图像处理发生错误: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/process_pipeline", methods=["POST"])
def process_pipeline():
    """
    处理图像的流水线（多步操作串联），用于批量处理
    接收数据格式:
    {
        "images": ["base64_str_1", "base64_str_2"],
        "pipeline": [
            {"action": "mean_filter", "params": {"kernel_size": 3}},
            {"action": "to_grayscale", "params": {}}
        ]
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "无效的请求数据"}), 400

        b64_images = data.get("images", [])
        pipeline = data.get("pipeline", [])

        if not b64_images or not pipeline:
            return jsonify({"success": False, "error": "缺少 necessary params"}), 400

        results = []
        for b64 in b64_images:
            # 每张图经历整个 pipeline
            img = processor.base64_to_image(b64)
            for step in pipeline:
                action = step.get("action")
                params = step.get("params", {})
                img = processor.process(action, [img], **params)

            result_b64 = processor.image_to_base64(".png", img)
            results.append(result_b64)

        return jsonify({"success": True, "result_images": results})
    except Exception as e:
        logging.error(f"批量流水线处理发生错误: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # 开启调试模式运行，如果在开发阶段修改了 app.py 会自动重启
    app.run(host="0.0.0.0", port=5000, debug=True)
