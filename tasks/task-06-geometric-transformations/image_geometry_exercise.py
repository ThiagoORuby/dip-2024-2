# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np


def apply_translation(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]

    translated_img = np.zeros_like(img)

    translated_img[dy:, dx:] = img[:h - dy, :w - dx]

    return translated_img


def apply_stretch(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = int(w*sx)
    new_h = int(h*sy)

    stretched_img = np.zeros((new_h, new_w), dtype=img.dtype)

    for y in range(new_h):
        for x in range(new_w):
            input_col = int(x/sx)
            input_row = int(y/sy)
            stretched_img[y, x] = img[input_row, input_col]

    return stretched_img


def apply_distortion(img: np.ndarray, k: float):
    h, w = img.shape[:2]
    center = w // 2, h // 2

    distorted_img = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            x_n = (x - center[0]) / center[0]
            y_n = (y - center[1]) / center[1]

            r2 = x_n**2 + y_n**2
            factor = 1 + k*r2

            x_d = int(round((x_n * factor * center[0]) + center[0]))
            y_d = int(round((y_n * factor * center[1]) + center[1]))

            if 0 <= x_d < w and 0 <= y_d < h:
                distorted_img[y, x] = img[y_d, x_d]

    return distorted_img

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Your implementation here
    return {
        "translated": apply_translation(img, 1, 1),
        "rotated": np.rot90(img, k=-1),
        "stretched": apply_stretch(img, 1.5, 1),
        "mirrored": img[:, ::-1],
        "distorted": apply_distortion(img, 1)
    }
