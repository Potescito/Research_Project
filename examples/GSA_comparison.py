"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
# %%
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import tabulate
import matplotlib.pyplot as plt
from src.VideoProcessor import VideoProcessor
from src.GSA import GSA
from src.metrics import NRMSE, PSNR, SSIM
from scipy.ndimage import zoom

if __name__ == "__main__":
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2)
    
    denoised_ds = {}

    PAN = np.sum(noisy_ds[n[0]][:10], axis=0) / 10
    LR_MS = noisy_ds[n[0]][:10]
    LR_MS = zoom(LR_MS, (1, 1/2, 1/2), order=1)

    HR_MS = GSA(LR_MS, PAN[None, :])

    psnr1 = PSNR(dataset[n[0]][:10], noisy_ds[n[0]][:10])
    psnr2 = PSNR(dataset[n[0]][:10], HR_MS)

    _, ax = plt.subplots(1, 4, figsize=(15, 15))
    ax[0].imshow(dataset[n[0]][0], cmap="gray"), ax[0].set_title("Original Frame"),  ax[0].axis("off")  
    ax[1].imshow(noisy_ds[n[0]][0], cmap="gray"), ax[1].set_title(f"Noisy Frame {psnr1:.3f}"), ax[1].axis("off")
    ax[2].imshow(LR_MS[0], cmap="gray"), ax[2].set_title("Downsampled Noisy Frame"), ax[2].axis("off")
    ax[3].imshow(HR_MS[0], cmap="gray"), ax[3].set_title(f"Upsampled Denoised Frame {psnr2:.3f}"), ax[3].axis("off")
    plt.tight_layout()
    plt.show()
# %%
