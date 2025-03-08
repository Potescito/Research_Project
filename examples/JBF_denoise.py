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
    """Simple JBL denoising example."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "rician"
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
    # sigma_x = 3.9574406147003174 #1.8602454662322998 #1.2
    # sigma_y = -0.18808746337890625 #0.3205803632736206 #1.2
    # sigma_z = -0.17778436839580536 #0.31436747312545776 #1.0
    # sigma_r = 0.6407924294471741 #0.4918845295906067 #0.8
    
    sigma_x = 5.582812786102295
    sigma_y =  0.47077813744544983
    sigma_z = -0.09726442396640778
    sigma_r = 0.2528565227985382
    
    
    # Original image only ====
    # Sigma x: -2.9593870639801025
    # Sigma y: -3.461167097091675
    # Sigma z: -3.652005195617676
    # Sigma range: -2.040886878967285
    # ========================
    
    # Image with Rician noise std 0.2 ==
    # Sigma x: 4.652834892272949
    # Sigma y: 0.6620981693267822
    # Sigma z: 0.6317738890647888
    # Sigma range: 0.6902894377708435
    # =====================
    
    # Image with Rician noise std 0.1 ==
    # Sigma x: 5.582812786102295
    # Sigma y: 0.47077813744544983
    # Sigma z: -0.09726442396640778
    # Sigma range: 0.2528565227985382
    # ====================
    
    
    # Initialize filter layer.
    layer_JBF = JointBilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=True)
    
    # Test in batch of videos
    tensor_gt = torch.tensor(stacked_video_array, dtype=torch.float32, device=dev)[:, None, ...]
    tensor_in = torch.tensor(stacked_video_array, dtype=torch.float32, device=dev)[:, None, ...]
    print(tensor_gt.shape, tensor_in.shape)

    # Guidance input
    tensor_guidance = tensor_in.clone().detach().to(dev)

    start = time.time()
    # Fwd pass
    denoised_img = layer_JBF(tensor_in, tensor_guidance).to(dev)

    # Bwd pass
    loss = denoised_img.mean()
    loss.backward()
    timestamp = time.time() - start

    # Results
    vmin_img = 0 #normalized
    vmax_img = 1
    idx_center = 10
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 3))
    axes[0].imshow(tensor_in[0, 0, idx_center, ...].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[0].set_title('Noisy input', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(denoised_img[0, 0, idx_center, ...].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[1].set_title('Filtered output', fontsize=14)
    axes[1].axis('off')
    axes[2].imshow(tensor_gt[0, 0, idx_center, ...].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[2].set_title('Ground truth', fontsize=14)
    axes[2].axis('off')
    plt.show()
    print(timestamp)

