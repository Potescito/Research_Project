"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
# %%
import sys
sys.path.append("../")
import numpy as np
import torch
import matplotlib.pyplot as plt
import src.utils as ut
from src.VideoProcessor import VideoProcessor
from src.anisotropic import anisotropic_diffusion
from skimage.restoration import denoise_bilateral
from src.GSA import GSA
from src.psnlm import psnlm
from src.metrics import NRMSE, PSNR, SSIM
from joint_bilateral_filter_layer import JointBilateralFilter3d
import time

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif']
plt.rcParams["font.size"] = 12
plt.rcParams['figure.constrained_layout.use'] = True

if __name__ == "__main__":
    """
    Script to make quantitative comparisons amongst denoising methods in the original image
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 6)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    # Extracting frames =======================================================
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())
    # ==========================================================================

    # Addition of noise ========================================================
    noise_type = "rician"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.1)
    # ==========================================================================

    # Metrics ==================================================================
    metrics = [NRMSE, PSNR, SSIM]
    noise_metrics = {} # each will hold a dict with the metrics and their values
    adf_metrics = {}
    bf_metrics = {}
    nlm_metrics = {}
    gsa_metrics = {}
    tjbf_metrics = {}
    # ==========================================================================

    # Denoising placeholders ==================================================
    adf = {}
    bf = {}
    nlm = {}
    gsa = {}
    tjbf = {}
    # ==========================================================================

    # time placeholders ========================================================
    timestamp_af = 0
    timestamp_bl = 0
    timestamp_nlm = 0
    timestamp_gsa = 0
    timestamp_jbl_gpu = 0
    # ==========================================================================
    
    # TJBF params ==============================================================
    sigma_x = 5.582812786102295
    sigma_y = 0.47077813744544983
    sigma_z = -0.09726442396640778
    sigma_r = 0.2528565227985382
    layer_JBF = JointBilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=True) 

    for i, (video_name, frames) in enumerate(noisy_ds.items()):
        print(f"[{i+1}/{len(dataset)}]", video_name, frames.shape)

        # ========================================================
        # Noise
        # ========================================================
        noise_metrics[video_name] = {metric.__name__: round(metric(dataset[video_name], frames), 4) 
                                    for metric in metrics}

        # ========================================================
        # Anisotropic Filtering 
        # ========================================================
        s = time.time()
        d = anisotropic_diffusion(frames, num_iter=12, kappa=2, gamma=0.003, option=2)
        timestamp_af += time.time() - s
        adf[video_name] = d

        adf_metrics[video_name] = {metric.__name__: round(metric(dataset[video_name], d), 4) 
                                  for metric in metrics}

        # ========================================================
        # Bilateral Filtering 
        # ========================================================
        s = time.time()
        d = denoise_bilateral(frames, win_size=3, sigma_color=5, sigma_spatial=-1, channel_axis=0) # model selection?
        timestamp_bl += time.time() - s
        bf[video_name] = d

        bf_metrics[video_name] = {metric.__name__: round(metric(dataset[video_name], d), 4)
                                 for metric in metrics}

        # ========================================================
        # PSNLM Filtering
        # ========================================================
        s = time.time()
        d = psnlm(frames, g_sigma=0.25, h=10, templateWindowSize=21, searchWindowSize=3)
        timestamp_nlm += time.time() - s
        nlm[video_name] = d

        nlm_metrics[video_name] = {metric.__name__: round(metric(dataset[video_name], d), 4)
                                  for metric in metrics}
  
        # ========================================================
        # Gram-Schmidt Adaptive Spectral Sharpening
        # ========================================================
        s = time.time()
        full = []
        for st in range(0, frames.shape[0], 10):
            end = min(st + 10, frames.shape[0])
            section = frames[st:end, ...]
            PAN = np.mean(section, axis=0)
            LR_MS = section[:, ::2, ::2] # downgrading factor of 2
            HR_MS = GSA(LR_MS, PAN[None, :])
            full.append(HR_MS)
        timestamp_gsa += time.time() - s
        gsa[video_name] = np.vstack(full)

        gsa_metrics[video_name] = {metric.__name__: round(metric(dataset[video_name], gsa[video_name]), 4)
                                  for metric in metrics}

          
        # ========================================================
        # Trainable Joint Bilateral Filtering (T-JBF) [trained parameters from 1 subject (100 epochs)]
        # ========================================================
        tensor_in = torch.tensor(frames, dtype=torch.float32, device=dev)[None, None, ...]
        tensor_guidance = tensor_in.clone().detach().to(dev)
        s = time.time()
        denoised_frames = layer_JBF(tensor_in, tensor_guidance)
        timestamp_jbl_gpu += time.time() - s
        d = denoised_frames.squeeze().detach().cpu().numpy()
        tjbf[video_name] = d

        tjbf_metrics[video_name] = {metric.__name__: round(metric(dataset[video_name], d), 4)
                                   for metric in metrics}

    # ==========================================================================
    print("Denoising Finished")

    method_times = {
        "ADF": timestamp_af / len(dataset),
        "BF": timestamp_bl / len(dataset),
        "GSA": timestamp_gsa / len(dataset),
        "PSNLM": timestamp_nlm / len(dataset),
        "T-JBF": timestamp_jbl_gpu / len(dataset)
    }
    print(method_times)

    # ==========================================================================

    frame = 100
    # columns = 6
    # skip = 1
    # names = [n[i] for i in range(1, len(n), skip)[:columns]]
    names = ['sub003_2drt_03_vcv3_r1_video', 
             'sub001_2drt_02_vcv2_r1_video',
             'sub004_2drt_01_vcv1_r1_video'
             ]

    ds_list = [dataset, noisy_ds, adf, bf, nlm, gsa, tjbf]
    dataset_labels = ["GT", "NS", "ADF", "BF", "GSA", "PSNLM", "T-JBF"]
    metric_list = [noise_metrics, adf_metrics, bf_metrics, nlm_metrics, gsa_metrics, tjbf_metrics]

    fig, ax = plt.subplots(len(dataset_labels), len(names), figsize=(8, 18)) 

    for j, (ds, label) in enumerate(zip(ds_list, dataset_labels)):
        for i, name in enumerate(names):
            ax[j, i].imshow(ds[name][frame], cmap="gray")
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            if i == 0:
                ax[j, i].set_ylabel(label, fontsize=14, labelpad=10, rotation=90, va="center")

            if j == 0:
                nam = name.split("_video")[0]
                ax[j, i].set_title(f"{nam}\nFrame {frame}", fontsize=10, pad=10)

            if j > 0:
                nrmse = metric_list[j-1][name]["NRMSE"]
                psnr = metric_list[j-1][name]["PSNR"]
                ssim = metric_list[j-1][name]["SSIM"]  

                text = f"NRMSE: {nrmse:.2f}\nPSNR: {psnr:.1f}\nSSIM: {ssim:.3f}"
                if j == 1:
                    ax[j, i].text(3, 65, text, fontsize=10, color="white", va="top", ha="left")
                else:
                    ax[j, i].text(3, 65, text, fontsize=10, color="yellow", va="top", ha="left")
        if j > 1:
            ax[j, len(names)-1].annotate(
                f"Avg: {method_times[label]:.2f} s", xy=(1.05, 0.5), xycoords="axes fraction",
                fontsize=12, va="center", rotation=-90, bbox=dict(facecolor="white", 
                edgecolor="none", pad=3))  
                  
    fig.tight_layout()
    plt.show()
    # fig.savefig('../images/original.pdf')

    # Tables =============
    for i, mt in enumerate(metric_list):
        print(i)
        avg_nrmse = sum(entry["NRMSE"] for entry in mt.values()) / len(n)
        avg_psnr  = sum(entry["PSNR"] for entry in mt.values()) / len(n)
        avg_ssim  = sum(entry["SSIM"] for entry in mt.values()) / len(n)
        print(f"AVGNRMSE:{avg_nrmse}, AVGPSNR: {avg_psnr}, AVGSSIM: {avg_ssim}")
    
# %%
# VideoProcessor.video(nlm, output_dir="../data/output_original_nlm/", fps=60)