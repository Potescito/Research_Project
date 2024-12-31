"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
- Source JBL module: https://github.com/faebstn96/trainable-joint-bilateral-filter-source
"""
# %%
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import tabulate
import matplotlib.pyplot as plt
from src.VideoProcessor import VideoProcessor
from src.metrics import NRMSE, PSNR, SSIM
import torch
from joint_bilateral_filter_layer import JointBilateralFilter3d

if __name__ == "__main__":
    """Simple JBL denoising example."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2)
    
    # JBL parameters
    sigma_x = 1.5
    sigma_y = 1.5
    sigma_z = 1.0
    sigma_r = 0.9

    # let's test in a single video
    tensor_gt = torch.tensor(dataset[n[0]], dtype=torch.float32, device=dev)
    tensor_in = torch.tensor(noisy_ds[n[0]], dtype=torch.float32, device=dev)

    # Initialize filter layer.
    layer_JBF = JointBilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=True)

    # Guidance input
    tensor_guidance = tensor_in.clone().detach()

    # Fwd pass
    denoised_img = layer_JBF(tensor_in, tensor_guidance)

    # Bwd pass
    loss = denoised_img.mean()
    loss.backward()

    # Results
    vmin_img = 0
    vmax_img = 1
    idx_center = int(tensor_in.shape[4] / 2)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 3))
    axes[0].imshow(tensor_in[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[0].set_title('Noisy input', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(denoised_img[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[1].set_title('Filtered output', fontsize=14)
    axes[1].axis('off')
    axes[2].imshow(tensor_gt[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[2].set_title('Ground truth', fontsize=14)
    axes[2].axis('off')
    plt.show()

