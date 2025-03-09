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
from src.psnlm import psnlm
from src.metrics import NRMSE, PSNR, SSIM

if __name__ == "__main__":
    dataset_path = r"../../data/dataset_2drt_video_only"
    vp = VideoProcessor(dataset_path, nSubs=["sub001"], norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "rician"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.1) # ref
    noise_psnr = PSNR(dataset[n[0]], noisy_ds[n[0]])

    denoised_ds = {}

    sigma = np.linspace(0.01, 2, 16)
    hache = np.linspace(1, 17, 16)
    temp_w  = np.arange(3, 22, 2) 
    temp_s  = np.arange(3, 22, 2) 

    log = {'s': [], 'h': [], 'tw': [], 'ts': []}

    _, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(dataset[n[0]][10], cmap="gray"), ax[0].axis("off"), ax[0].set_title("Original")
    ax[1].imshow(noisy_ds[n[0]][10], cmap="gray"), ax[1].axis("off"), ax[1].set_title(f"Noise Added: {noise_type} / PSNR:{noise_psnr:.4f}")

    _, axg = plt.subplots(4, 4, figsize=(15, 15))
    _, axk = plt.subplots(4, 4, figsize=(15, 15))
    _, axi = plt.subplots(4, 4, figsize=(15, 15))
    _, axj = plt.subplots(4, 4, figsize=(15, 15))

    video_name = n[0]
    frames = noisy_ds[video_name]
    # frames = dataset[video_name]
    print(video_name, frames.shape)
    for idx, g in enumerate(sigma):
        denoised_frames = np.stack(psnlm(frames, g_sigma=g, h=2, templateWindowSize=9, searchWindowSize=7), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["s"].append([g, denoi_psnr])
        row, col = divmod(idx, 4)
        axg[row, col].imshow(denoised_frames[10], cmap="gray")
        axg[row, col].axis("off")
        axg[row, col].set_title(f"Sigma:{g:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, k in enumerate(hache):
        denoised_frames = np.stack(psnlm(frames, g_sigma=0.2, h=k, templateWindowSize=9, searchWindowSize=7), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["h"].append([k, denoi_psnr])
        row, col = divmod(idx, 4)
        axk[row, col].imshow(denoised_frames[10], cmap="gray")
        axk[row, col].axis("off")
        axk[row, col].set_title(f"h:{k:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, i in enumerate(temp_w):
        denoised_frames = np.stack(psnlm(frames, g_sigma=0.2, h=2, templateWindowSize=i, searchWindowSize=7), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["tw"].append([i, denoi_psnr])
        row, col = divmod(idx, 4)
        axi[row, col].imshow(denoised_frames[10], cmap="gray")
        axi[row, col].axis("off")
        axi[row, col].set_title(f"t_w:{i:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, j in enumerate(temp_s):
        denoised_frames = np.stack(psnlm(frames, g_sigma=0.2, h=2, templateWindowSize=9, searchWindowSize=j), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["ts"].append([j, denoi_psnr])
        row, col = divmod(idx, 4)
        axj[row, col].imshow(denoised_frames[10], cmap="gray")
        axj[row, col].axis("off")
        axj[row, col].set_title(f"t_s:{j:.4f}/PSNR:{denoi_psnr:.4f}")

    plt.tight_layout()
    plt.show()

