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
import time

if __name__ == "__main__":
    """Trainable JBL denoising example."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2)
    
    # JBL module requires this input array shape -> (B, C, X, Y, Z)
    min_frames = min(video.shape[0] for video in dataset.values()) # To be able to stack videos of diff size
    
    resized_videos = []
    resi_no_videos = []
    for video, noisy in zip(dataset.values(), noisy_ds.values()):
        resized_video = video[:min_frames]  
        resized_videos.append(resized_video)
        
        resized_video = noisy[:min_frames]
        resi_no_videos.append(resized_video)

    stacked_video_array = np.stack(resized_videos, axis=0) # [B, CH, X, Y]
    stacked_video_arrayn = np.stack(resi_no_videos, axis=0)
    
    
    # JBL parameters (initialization if training)
    sigma_x = 1.0
    sigma_y = 1.0
    sigma_z = 1.0
    sigma_r = 0.01
    n_epochs = 100
    
    # Initialize filter layer.
    layer_JBF = JointBilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=True)
    
    # Test in batch of videos
    target = torch.tensor(stacked_video_array, dtype=torch.float32, device=dev)[:, None, ...]
    tensor_in = torch.tensor(stacked_video_arrayn, dtype=torch.float32, device=dev)[:, None, ...]
    print(target.shape, tensor_in.shape)
    tensor_in.requires_grad = True
    
    # Guidance input
    tensor_guidance = tensor_in.clone().detach().to(dev)
    tensor_guidance.requires_grad = True
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(layer_JBF.parameters(), lr=0.1)
    loss_function = torch.nn.MSELoss()
    
    start = time.time()
    # Training loop.
    for i in range(n_epochs):
        optimizer.zero_grad()

        prediction = layer_JBF(tensor_in, tensor_guidance)
        loss = loss_function(prediction, target)
        loss.backward()

        optimizer.step()
    timestamp = time.time() - start
    
    print("Sigma x: {}".format(layer_JBF.sigma_x))
    print("Sigma y: {}".format(layer_JBF.sigma_y))
    print("Sigma z: {}".format(layer_JBF.sigma_z))
    print("Sigma range: {}".format(layer_JBF.color_sigma))
    
    # Results
    vmin_img = 0
    vmax_img = 1
    idx_center = 10
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 3))
    axes[0].imshow(tensor_in[0, 0, idx_center, ...].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[0].set_title('Noisy input', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(prediction[0, 0, idx_center, ...].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[1].set_title('Filtered output', fontsize=14)
    axes[1].axis('off')
    axes[2].imshow(target[0, 0, idx_center, ...].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[2].set_title('Ground truth', fontsize=14)
    axes[2].axis('off')
    plt.show()
    print(timestamp)