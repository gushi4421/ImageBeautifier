from src.algorithm.geometry import (
    translate,
    rotate,
    zoom,
    flip,
    shear,
)
from src.io.image_io import load_image, save_image
from src.algorithm.noise import add_noise, remove_noise
import time
from src.algorithm.color import (
    to_grayscale,
    binarize,
    otsu_binarize,
    generate_colormap_lut,
    apply_colormap,
)
from src.algorithm.enhancer import adjust_brightness, adjust_contrast, histogram
from src.algorithm.beautifier import (
    emboss,
    frosted,
    gaussian_blurring,
    bilateral_filter_manual,
)


def main():
    image = load_image("test.png")
    translate_image = translate(image.copy(), 100, 100)
    zoom_image = zoom(image.copy(), 2, 1)
    rotate_image = rotate(image.copy(), 45)
    flip_image = flip(image.copy())
    shear_image = shear(image.copy(), start_x=100, start_y=100, end_x=200, end_y=150)
    gaussian_image = add_noise(image.copy(), mode="gaussian", mean=0.0, sigma=100.0)
    salt_and_pepper_image = add_noise(image.copy(), mode="salt_pepper", prob=0.2)
    mean_image = remove_noise(gaussian_image.copy(), mode="mean", kernal_size=3)
    median_image = remove_noise(gaussian_image.copy(), mode="median", kernal_size=3)
    gray_image = to_grayscale(image.copy())
    binary_image = otsu_binarize(gray_image.copy())
    color_image = apply_colormap(gray_image.copy())
    bright_image = adjust_brightness(image.copy(), beta=100)
    contrast_image = histogram(image.copy())
    emboss_image = emboss(image.copy())
    frosted_image = frosted(image.copy())
    gaussian_blurring_image = gaussian_blurring(image.copy())
    bilateral_image = bilateral_filter_manual(
        image.copy(), kernel_size=5, sigma_s=75, sigma_r=75
    )
    save_image(translate_image, "data/translate_image.png")
    save_image(zoom_image, "data/zoom_image.png")
    save_image(rotate_image, "data/rotate_image.png")
    save_image(flip_image, "data/flip_image.png")
    save_image(shear_image, "data/shear_image.png")
    save_image(gaussian_image, "data/gaussian_image.png")
    save_image(salt_and_pepper_image, "data/salt_pepper.png")
    save_image(mean_image, "data/mean_image.png")
    save_image(median_image, "data/median_image.png")
    save_image(gray_image, "data/gray_image.png")
    save_image(binary_image, "data/binary_image.png")
    save_image(color_image, "data/color_image.png")
    save_image(bright_image, "data/bright_image.png")
    save_image(contrast_image, "data/contrast_image.png")
    save_image(emboss_image, "data/emboss_image.png")
    save_image(frosted_image, "data/frosted_image.png")
    save_image(gaussian_blurring_image, "data/gaussian_blurring_image.png")
    save_image(bilateral_image, "data/bilateral_image.png")


if __name__ == "__main__":
    main()
