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
from skimage.restoration import denoise_bilateral
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

    win_size = np.arange(3, 12, 2)
    sigma_color = np.linspace(-5, 5, 16)
    sigma_space  = np.linspace(-1, 1, 16) # 16 iterations

    log = {'w': [], 'sc': [], 'ss': []}

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
    for idx, g in enumerate(win_size):
        denoised_frames = np.stack(denoise_bilateral(frames, win_size=g, sigma_color=0.64, sigma_spatial=-0.18, channel_axis=0), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["w"].append([g, denoi_psnr])
        row, col = divmod(idx, 4)
        axg[row, col].imshow(denoised_frames[10], cmap="gray")
        axg[row, col].axis("off")
        axg[row, col].set_title(f"Win_size:{g:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, k in enumerate(sigma_color):
        denoised_frames = np.stack(denoise_bilateral(frames, win_size=3, sigma_color=k, sigma_spatial=-0.18, channel_axis=0), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["sc"].append([k, denoi_psnr])
        row, col = divmod(idx, 4)
        axk[row, col].imshow(denoised_frames[10], cmap="gray")
        axk[row, col].axis("off")
        axk[row, col].set_title(f"Sigma_c:{k:.4f}/PSNR:{denoi_psnr:.4f}")
    for idx, i in enumerate(sigma_space):
        denoised_frames = np.stack(denoise_bilateral(frames, win_size=3, sigma_color=0.64, sigma_spatial=i, channel_axis=0), axis=0)
        denoi_psnr = PSNR(dataset[video_name], denoised_frames)
        log["ss"].append([i, denoi_psnr])
        row, col = divmod(idx, 4)
        axi[row, col].imshow(denoised_frames[10], cmap="gray")
        axi[row, col].axis("off")
        axi[row, col].set_title(f"Sigma_s:{i:.4f}/PSNR:{denoi_psnr:.4f}")

    plt.tight_layout()
    plt.show()

