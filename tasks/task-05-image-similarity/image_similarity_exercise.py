# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np


def get_mse(i1: np.ndarray, i2: np.ndarray) -> float:
    error = np.square(i1 - i2)
    mse = float(np.mean(error))
    return mse

def get_psnr(i1: np.ndarray, i2: np.ndarray, max) -> float:
    mse = get_mse(i1, i2)

    if mse == 0:
        return float("inf")
    else:
        return 20 * np.log10(max) - 10*np.log10(mse)

def get_ssim(i1: np.ndarray, i2: np.ndarray, L: int) -> float:
    
    mean_i1 = np.mean(i1)
    mean_i2 = np.mean(i2)

    cov_matrix = np.cov(i1.flatten(), i2.flatten())
    var_i1 = cov_matrix[0, 0]
    var_i2 = cov_matrix[1, 1]
    cov = cov_matrix[0, 1]

    c1 = (0.01*L)**2
    c2 = (0.03*L)**2

    den = (2*mean_i1*mean_i2 + c1)*(2*cov + c2)
    num = (mean_i1**2 + mean_i2**2 + c1)*(var_i1 + var_i2 + c2)

    return den / num

def get_npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    
    mean_i1 = np.mean(i1)
    mean_i2 = np.mean(i2)

    den = np.sum((i1 - mean_i1)*(i2 - mean_i2))
    num = np.sqrt(np.sum((i1 - mean_i1)**2))*np.sqrt(np.sum((i2 - mean_i2)**2))

    return den / num

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here
    return {
        "mse": get_mse(i1, i2),
        "psnr": get_psnr(i1, i2, 255),
        "ssim": get_ssim(i1, i2, 255),
        "npcc": get_npcc(i1, i2)
    }
