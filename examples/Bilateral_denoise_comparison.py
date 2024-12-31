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
from skimage.restoration import denoise_bilateral
from src.metrics import NRMSE, PSNR, SSIM

if __name__ == "__main__":
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 11)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2)
    
    denoised_ds = {}

    _, ax = plt.subplots(5, 3, figsize=(15, 15))
    for i, (video_name, frames) in enumerate(noisy_ds.items()):
        print(f"[{i+1}/{len(noisy_ds)}]", video_name, frames.shape)
                
        denoised_frames = denoise_bilateral(frames, win_size=3, sigma_color=0.5, sigma_spatial=-2, channel_axis=0) # model selection?
        denoised_ds[video_name] = np.stack(denoised_frames, axis=0)

        if i < 5:
            ax[i, 0].imshow(dataset[video_name][10], cmap="gray"), ax[i, 0].axis("off"), ax[i, 0].set_title("Original")
            ax[i, 1].imshow(noisy_ds[video_name][10], cmap="gray"), ax[i, 1].axis("off"), ax[i, 1].set_title(f"Noise Added: {noise_type}")
            ax[i, 2].imshow(denoised_ds[video_name][10], cmap="gray"), ax[i, 2].axis("off"), ax[i, 2].set_title("Denoised: Bilateral")
    plt.tight_layout()
    plt.show()


    # Metrics ==================================================================
    log = []
    for video_name in n:
        gt = dataset[video_name]
        noisy_frames = noisy_ds[video_name]
        denoised_frames = denoised_ds[video_name]

        for metric in [NRMSE, PSNR, SSIM]:
            log.append({
                "Video Name": video_name,
                "Metric": metric.__name__,
                "Noise": round(metric(gt, noisy_frames), 4),
                "Denoised": round(metric(gt, denoised_frames), 4)
            })

    df = pd.DataFrame(log)
    df = df.pivot_table(index=["Video Name"], columns=["Metric"], values=["Noise", "Denoised"])
    df.columns = [f'{col[1]} ({col[0]})' for col in df.columns] # sorting and also better layout :)
    df.reset_index(inplace=True)
    print(tabulate.tabulate(df, headers='keys', tablefmt='grid'))


    # df.to_csv(f"Metrics/metrics_bilat_{nSubs[0]}_{nSubs[-1]}_3_0.5_-2.csv", index=False)