import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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


source_img = cv.imread("source.jpg")
source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
reference_img = cv.imread("reference.jpg")
reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)
matched_img =  match_histograms_rgb(source_img, reference_img)

images = {}
images["source"] = source_img
images["reference"] = reference_img
images["matched"] = matched_img


_, axs = plt.subplots(3, 4, figsize=(25, 20))

for i, (name, img) in enumerate(images.items()):
    axs[i, 0].imshow(img)
    axs[i, 0].set_title(name)
    axs[i, 0].axis("off")

for i, channel in enumerate(["red", "green", "blue"]):
    for j, (name, img) in enumerate(images.items()):
        hist, bins = np.histogram(img[:, :, i].flatten(),
                                  bins=256, range=(0,256))
        pdf = hist / hist.sum()
        axs[j, i+1].bar(bins[:-1], pdf, width=1,
                        color=f"tab:{channel}", alpha=0.6)
        axs[j, i+1].set_title(f"channel: {channel}")

plt.tight_layout()
plt.savefig("histograms.jpg")
