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
import torch
import matplotlib.pyplot as plt
from src.VideoProcessor import VideoProcessor
from src.metrics import NRMSE, PSNR, SSIM
from src.anisotropic import anisotropic_diffusion
from skimage.restoration import denoise_bilateral
from src.GSA import GSA
from joint_bilateral_filter_layer import JointBilateralFilter3d
import time
import src.utils as ut

if __name__ in ["__main__", "__mp_main__"]:
    """
    Script to make quantitative comparisons amongst denoising methods.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = r"../data/dataset_2drt_video_only"
    vp = VideoProcessor(dataset_path, nSubs=['sub001'], norm=True) # only (JBL issue)
    
    dataset = vp.extract_frames(target="vcv") # all vcv named videos
    n = list(dataset.keys()) # all names
    
    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2) # standard
    
    # ========================================================
    # Anisotropic Filtering 
    # ========================================================
    denoised_af = {}
    iterations = 11 
    kappa = 0.7
    gamma = 0.02
    opt = 2
    start = time.time()
    for i, (video_name, frames) in enumerate(noisy_ds.items()):
        print(f"AF [{i+1}/{len(noisy_ds)}]", video_name, frames.shape)

        denoised_frames = anisotropic_diffusion(frames, num_iter=iterations, 
                                                kappa=kappa, gamma=gamma, option=opt)
        denoised_af[video_name] = denoised_frames
    timestamp_af = time.time() - start
    print("SUMMARY: =======================================================")
    print(f"AF: {len(noisy_ds)} videos denoised in {timestamp_af} seconds.")
    print(f"Parameters: Iterations({iterations}), Kappa({kappa}), Gamma({gamma}), Algorithm({opt}).")
    
    # ========================================================
    # Bilateral Filtering 
    # ========================================================
    import warnings
    # denoise_bilateral not meant for multichannel images
    warnings.filterwarnings("ignore", module="skimage") 
    
    denoised_bf = {}
    win = 3
    sigma_c = 0.5
    sigma_s = -2
    start = time.time()
    for i, (video_name, frames) in enumerate(noisy_ds.items()):
        print(f"BF [{i+1}/{len(noisy_ds)}]", video_name, frames.shape)

        denoised_frames = denoise_bilateral(frames, win_size=win, sigma_color=sigma_c, 
                                            sigma_spatial=sigma_s, channel_axis=0)
        denoised_bf[video_name] = denoised_frames
    timestamp_bl = time.time() - start
    print("SUMMARY: =======================================================")
    print(f"BF: {len(noisy_ds)} videos denoised in {timestamp_bl} seconds.")
    print(f"Parameters: Window Size({win}), Sigma Color({sigma_c}), Sigma Spatial({sigma_s}).")
    
    # ========================================================
    # Gram-Schmidt Adaptive Spectral Sharpening
    # ========================================================
    denoised_gsa = {}
    pan_avg_length = 10
    down_f = 2
    start = time.time()
    for i, (video_name, frames) in enumerate(noisy_ds.items()):
        print(f"GSA [{i+1}/{len(noisy_ds)}]", video_name, frames.shape)
        
        # Section-wise (terribly slow)
        full = []
        for st in range(0, frames.shape[0], pan_avg_length):
            end = min(st + pan_avg_length, frames.shape[0])
            section = frames[st:end, ...]
            
            PAN = np.mean(section, axis=0)
            LR_MS = section[:, ::down_f, ::down_f] # atrificially downsampled version
        
            HR_MS = GSA(LR_MS, PAN[None, :])
            full.append(HR_MS)
            
        denoised_gsa[video_name] = np.vstack(full)
    timestamp_gsa = time.time() - start
    print("SUMMARY: =======================================================")
    print(f"GSA: {len(noisy_ds)} videos denoised in {timestamp_gsa} seconds.")
    print(f"Parameters: Frames for Panchromatic Average({pan_avg_length}), Downsampling Factor({down_f}).")

    # ========================================================
    # JBL with trained parameters from 1 subject (100 epochs)
    # ========================================================
    
    # Per each video / conserving original frame lenght / consistent comparison (GPU)
    denoised_jbl = {}
    
    sigma_x = 3.9574406147003174 
    sigma_y = -0.18808746337890625 
    sigma_z = -0.17778436839580536 
    sigma_r = 0.6407924294471741 
    dev = "cuda"
    
    layer_JBF = JointBilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=True)
    
    start = time.time()
    for i, (video_name, frames) in enumerate(noisy_ds.items()):
        print(f"T-JBF [{i+1}/{len(noisy_ds)}]", video_name, frames.shape)

        tensor_in = torch.tensor(frames, dtype=torch.float32, device=dev)[None, None, ...]
        tensor_guidance = tensor_in.clone().detach().to(dev)
        
        denoised_frames = layer_JBF(tensor_in, tensor_guidance)
        
        denoised_jbl[video_name] = denoised_frames.squeeze().detach().cpu().numpy()
    timestamp_jbl_gpu = time.time() - start
    print("SUMMARY: =======================================================")
    print(f"T-JBF: {len(noisy_ds)} videos denoised in {timestamp_jbl_gpu} seconds.")
    print(f"Parameters: Trained for 100 epochs. Sigma X({sigma_x}), Sigma Y({sigma_y}), Sigma Z({sigma_y}), Sigma R({sigma_r})), GPU/CPU({dev}).")
    
    # ======================================================
    # Comparisons 
    # ======================================================
    print("Computing comparisons...")
    log = []
    
    method_times = {
        "Anisotropic": timestamp_af,
        "Bilateral": timestamp_bl,
        "GSA": timestamp_gsa,
        "JBL": timestamp_jbl_gpu
    }
    
    for video_name in n:
        gt = dataset[video_name]
        noisy_frames = noisy_ds[video_name]

        noise_metrics = {metric.__name__: round(metric(gt, noisy_frames), 4) 
                         for metric in [NRMSE, PSNR, SSIM]}
        
        denoised_methods = {
            "Anisotropic": denoised_af[video_name],
            "Bilateral": denoised_bf[video_name],
            "GSA": denoised_gsa[video_name],
            "JBL": denoised_jbl[video_name]
        }                            

        for method, denoised_frames in denoised_methods.items():
                for metric in [NRMSE, PSNR, SSIM]:
                    log.append({
                        "Video Name": video_name,
                        "Method": method,
                        "Metric": metric.__name__,
                        "Noise": noise_metrics[metric.__name__],
                        "Denoised": round(metric(gt, denoised_frames), 4),
                        "Time Taken (s)": round(method_times[method], 2)
                    })
        
    df = pd.DataFrame(log)

    chart = tabulate.tabulate(
        df,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
    print(chart)

    df.to_csv("Metrics/denoising_metrics_comparison.csv", index=False)
    
    ut.save_ds("Metrics/noise.npz", noisy_ds)
    ut.save_ds("Metrics/denoised_af.npz", denoised_af)
    ut.save_ds("Metrics/denoised_bf.npz", denoised_bf)    
    ut.save_ds("Metrics/denoised_gsa.npz", denoised_gsa)    
    ut.save_ds("Metrics/denoised_jbl.npz", denoised_jbl)    