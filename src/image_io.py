from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Any


class ImageLoader:
    def __init__(self):
        pass
    

    def save_image(self, image, path):
        pass
    
    def read_image(self, path: str):
        pass

    @staticmethod
    def _to_numpy(image: Any)->np.ndarray:
        pass
