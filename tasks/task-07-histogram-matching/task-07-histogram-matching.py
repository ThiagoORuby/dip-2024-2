# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski


def compute_cdf(img: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(img.flatten(),
                              bins=256, range=(0,256))
    pdf = hist / hist.sum()
    return pdf.cumsum()

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Your implementation here
    matched = []
    for channel in range(3): # R, G, B

        # captura o canal específico
        src_c = source_img[:, :, channel]
        ref_c = reference_img[:, :, channel]

        # gera CDF(s_k) e CDF(z_k)
        src_cdf = compute_cdf(src_c)
        ref_cdf = compute_cdf(ref_c)

        # array de mapeamento
        mapping = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            # mapeia s_k em z_k onde CDF(s_k) se aproxima de CDF(z_k)
            diff = np.abs(ref_cdf - src_cdf[i])
            mapping[i] = np.argmin(diff) # menor distancia

        matched_c = mapping[src_c]
        matched.append(matched_c)

    # retorna canais mesclados
    return cv.merge(matched)
