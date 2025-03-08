"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
# %%
import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import tabulate
import matplotlib.pyplot as plt
from src.VideoProcessor import VideoProcessor
from src.anisotropic import anisotropic_diffusion
from src.metrics import NRMSE, PSNR, SSIM

if __name__ == "__main__":
    dataset_path = r"../../data/dataset_2drt_video_only"
    vp = VideoProcessor(dataset_path, nSubs=["sub001"], norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2) # ref
    noise_psnr = PSNR(dataset[n[0]], noisy_ds[n[0]])

    denoised_ds = {}

    gamma_range = np.linspace(-0.1, 0.1, 16)
    kappa_range = np.linspace(-10, 10, 16)
    iter_range  = np.arange(1, 17, 1) # 16 iterations

    log = {'k': [], 'g': [], 'i': []}

    _, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(dataset[n[0]][10], cmap="gray"), ax[0].axis("off"), ax[0].set_title("Original")
    ax[1].imshow(noisy_ds[n[0]][10], cmap="gray"), ax[1].axis("off"), ax[1].set_title(f"Noise Added: {noise_type} / PSNR:{noise_psnr:.4f}")

    _, axg = plt.subplots(4, 4, figsize=(15, 15))
    _, axk = plt.subplots(4, 4, figsize=(15, 15))
    _, axi = plt.subplots(4, 4, figsize=(15, 15))

    video_name = n[0]
    # frames = noisy_ds[video_name]
    frames = dataset[video_name]
    print(video_name, frames.shape)
    for idx, g in enumerate(gamma_range):
        denoised_frames = np.stack(anisotropic_diffusion(frames, num_iter=11, kappa=2, gamma=g, option=2), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["g"].append([g, denoi_psnr])
        row, col = divmod(idx, 4)
        axg[row, col].imshow(denoised_frames[10], cmap="gray")
        axg[row, col].axis("off")
        axg[row, col].set_title(f"Gamma:{g:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, k in enumerate(kappa_range):
        denoised_frames = np.stack(anisotropic_diffusion(frames, num_iter=11, kappa=k, gamma=0.02, option=2), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["k"].append([k, denoi_psnr])
        row, col = divmod(idx, 4)
        axk[row, col].imshow(denoised_frames[10], cmap="gray")
        axk[row, col].axis("off")
        axk[row, col].set_title(f"Kappa:{k:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, i in enumerate(iter_range):
        denoised_frames = np.stack(anisotropic_diffusion(frames, num_iter=i, kappa=2, gamma=0.02, option=2), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["i"].append([i, denoi_psnr])
        row, col = divmod(idx, 4)
        axi[row, col].imshow(denoised_frames[10], cmap="gray")
        axi[row, col].axis("off")
        axi[row, col].set_title(f"Iter:{i:.4f}/PSNR:{denoi_psnr:.4f}")

    plt.tight_layout()
    plt.show()

