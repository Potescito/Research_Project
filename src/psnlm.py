"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

def psnlm(img: np.ndarray, g_sigma=0.2, h=2, templateWindowSize=9, searchWindowSize=7):
    """
    PSNLM implementation. First proposed by Yang et. al 2015.
    Image must be normalized.

    Args:
        img (np.ndarray): Image to be denoised (normalized). Expected shape (Frames, Height, Width)
        g_sigma (float, optional): Standard deviation for gaussian smoothing. Defaults to 0.2.
        h (int, optional): Parameter regulating filter strength. Higher h value removes noise better, but removes details of image also. Defaults to 2.
        templateWindowSize (int, optional): Size in pixels of the template patch. Should be odd. Defaults to 9.
        searchWindowSize (int, optional): Size in pixels of the window to search for patches. Should be odd. Defaults to 7.
    Returns:
        np.ndarray: Denoised image.
    """
    denoised_img = []
    for frame in img:
        # Extract background and compute standard deviation from it
        thresh = threshold_otsu(frame)
        bg_mask = frame < thresh
        std = np.std(frame * bg_mask)

        # Variance-stabilization (optimal for rician noise) -> here I sed generalized Anscombe-like transform
        fwd = 2 * np.sqrt(np.maximum(frame**2 - std**2, 0) + 3/8)
        
        # Smoothing -> here simple gaussian / can be anisotropic
        smt = gaussian_filter(fwd, sigma=g_sigma)

        # NLM 
        smt_min, smt_max = smt.min(), smt.max()
        smt_scaled = ((smt - smt_min) / (smt_max - smt_min) * 255).astype(np.uint8)
        denoised_scaled = cv2.fastNlMeansDenoising(smt_scaled, None, h, templateWindowSize, searchWindowSize)
        denoised = denoised_scaled.astype(np.float32) / 255.0 * (smt_max - smt_min) + smt_min

        # Inverse transform
        inv = np.sqrt((denoised / 2)**2 - 3/8 + std**2)
        denoised_img.append(inv)

    return np.stack(denoised_img, axis=0)

# %% Debugging
if __name__ == "__main__":
    from psnlm import psnlm
    from VideoProcessor import VideoProcessor
    import matplotlib.pyplot as plt
    import time

    ds = VideoProcessor(r"../data/dataset_2drt_video_only", nSubs=["sub001"], norm=True)
    imgs = ds.extract_frames(target="vcv")
    # imgs = ds.noise(imgs, type="gaussian", mean=0, std=0.1)
    n = list(imgs.keys())
    print(n)

    s = time.time()
    a = psnlm(imgs[n[0]])
    print(time.time() - s)

    plt.imshow(a[10], cmap="gray"), plt.colorbar()
    plt.show()
    plt.imshow(imgs[n[0]][10], cmap="gray"), plt.colorbar()
    plt.show()

# %%
