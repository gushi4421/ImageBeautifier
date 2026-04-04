from src.algorithm.geometry import (
    translate,
    rotate,
    zoom,
    flip,
    shear,
    affine_transform,
)
from src.management.image_io import load_image, save_image


def main():
    image = load_image("test.png")
    image = translate(image, 10, 100)
    save_image(image, "test_translate.png")


if __name__ == "__main__":
    main()
