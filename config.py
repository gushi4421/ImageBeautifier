"""
统一配置加载模块.

从项目根目录的 config.yaml 中读取配置, 提供便捷的取值接口.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_config: dict | None = None


def _load_config() -> dict:
    global _config
    if _config is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def get_output_dir(category: str) -> str:
    """读取 config.yaml 中 output 段下的目录路径."""
    cfg = _load_config()
    return cfg["output"][category]
