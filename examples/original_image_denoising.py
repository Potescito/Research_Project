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
from src.utils import imshow
from joint_bilateral_filter_layer import JointBilateralFilter3d
import time

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif']
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 5)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    adf = {}
    bf = {}
    nlm = {}
    gsa = {}
    tjbf1 = {}
    tjbf2 = {}

    timestamp_af = 0
    timestamp_bl = 0
    timestamp_nlm = 0
    timestamp_gsa = 0
    timestamp_jbl_gpu1 = 0
    timestamp_jbl_gpu2 = 0

    dev = "cuda"
    layer_JBF1 = JointBilateralFilter3d(5.582812786102295, 0.47077813744544983, -0.09726442396640778, 0.2528565227985382, use_gpu=True) # std 0.1 rician
    layer_JBF2 = JointBilateralFilter3d(4.652834892272949, 0.6620981693267822, 0.6317738890647888, 0.6902894377708435, use_gpu=True) # std 0.2 rician

    for i, (video_name, frames) in enumerate(dataset.items()):
        print(f"[{i+1}/{len(dataset)}]", video_name, frames.shape)
        # ADF
        s = time.time()
        d = anisotropic_diffusion(frames, num_iter=8, kappa=0.6, gamma=0.0067, option=2)
        timestamp_af += time.time() - s
        adf[video_name] = d
        # BF
        s = time.time()
        d = denoise_bilateral(frames, win_size=3, sigma_color=5, sigma_spatial=-1, channel_axis=0) # model selection?
        timestamp_bl += time.time() - s
        bf[video_name] = d
        # psNLM
        s = time.time()
        d = psnlm(frames, g_sigma=0.25, h=2, templateWindowSize=21, searchWindowSize=3)
        timestamp_nlm += time.time() - s
        nlm[video_name] = d
        # GSA
        s = time.time()
        full = []
        for st in range(0, frames.shape[0], 10):
            end = min(st + 10, frames.shape[0])
            section = frames[st:end, ...]
            PAN = np.mean(section, axis=0)
            LR_MS = section[::2, ::2] # downgrading factor of 2
            HR_MS = GSA(LR_MS, PAN[None, :])
            full.append(HR_MS)
        timestamp_gsa += time.time() - s
        gsa[video_name] = np.vstack(full)
        # TJBF std 0.1
        tensor_in = torch.tensor(frames, dtype=torch.float32, device=dev)[None, None, ...]
        tensor_guidance = tensor_in.clone().detach().to(dev)
        s = time.time()
        denoised_frames = layer_JBF1(tensor_in, tensor_guidance)
        timestamp_jbl_gpu1 += time.time() - s
        tjbf1[video_name] = denoised_frames.squeeze().detach().cpu().numpy()
        # TJBF std 0.2
        tensor_in = torch.tensor(frames, dtype=torch.float32, device=dev)[None, None, ...]
        tensor_guidance = tensor_in.clone().detach().to(dev)
        s = time.time()
        denoised_frames = layer_JBF2(tensor_in, tensor_guidance)
        timestamp_jbl_gpu2 += time.time() - s
        tjbf2[video_name] = denoised_frames.squeeze().detach().cpu().numpy()

    method_times = {
        "ADF": timestamp_af / len(dataset),
        "BF": timestamp_bl / len(dataset),
        "PSNLM": timestamp_nlm / len(dataset),
        "GSA": timestamp_gsa / len(dataset),
        "T-JBF_0.1": timestamp_jbl_gpu1 / len(dataset),
        "T-JBF_0.2": timestamp_jbl_gpu2 / len(dataset)
    }

    frame = 100
    columns = 6
    skip = 3
    signal_box_coords=(15, 35, 30, 30)
    noise_box_coords=(53, 0, 30, 30)
    ds_list = [dataset[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    adf_list = [adf[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    bf_list = [bf[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    nlm_list = [nlm[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    gsa_list = [gsa[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    tjbf_list1 = [tjbf1[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    tjbf_list2 = [tjbf2[n[i]][frame] for i in range(1, len(n), skip)[:columns]]
    names = [n[i] for i in range(1, len(n), 3)[:columns]]

    # imshow(ds_list, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30)) 
    # imshow(adf_list, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30))
    # imshow(bf_list, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30))
    # imshow(nlm_list, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30))
    # imshow(gsa_list, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30))
    # imshow(tjbf_list1, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30))
    # imshow(tjbf_list2, signal_box_coords=(15, 35, 30, 30), noise_box_coords=(53, 0, 30, 30))

    dataset_labels = ["GT", "ADF", "BF", "PSNLM", "GSA", "T-JBF_0.1", "T-JBF_0.2"]

    fig, ax = plt.subplots(len(dataset_labels), len(ds_list), figsize=(20, 20)) 

    for j, (ds, label) in enumerate(zip([ds_list, adf_list, bf_list, nlm_list, gsa_list, tjbf_list1, tjbf_list2], dataset_labels)):
        for i, name in enumerate(names):
            ax[j, i].imshow(ds[i], cmap="gray")
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ut.annotate_metrics(ds[i], ax[j, i], font_weight="normal", signal_box_coords=signal_box_coords, noise_box_coords=noise_box_coords)
            ut.draw_box(ax[j, i], signal_box_coords)
            ut.draw_box(ax[j, i], noise_box_coords)
            if i == 0:
                ax[j, i].set_ylabel(label, fontsize=14, labelpad=10, rotation=90, va="center")

            if j == 0:
                nam = name.split("_video")[0]
                ax[j, i].set_title(f"{nam}\nFrame {frame}", fontsize=10, pad=10)

        if j > 0:
            ax[j, len(ds_list)-1].annotate(
                f"Avg: {method_times[label]:.2f} s", xy=(1.05, 0.5), xycoords="axes fraction",
                fontsize=12, va="center", rotation=-90, bbox=dict(facecolor="white", 
                edgecolor="none", pad=3))  
                  
    fig.tight_layout()
    plt.show()
    # fig.savefig('../images/original.pdf')

# %%
