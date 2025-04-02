"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import sys
sys.path.append('../')
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.AVDataset import AVDataset
from src.transforms import SlidingWindowTransform
from C_wav2vec2_AP import AudioFeatureExtractorFiLM
from C_cond_Unet import ConditionalUNet3D_FiLM

def train_one_epoch(model, audio_extractor, dataloader, criterion, optimizer, device):
    model.train()
    # Typically, freeze the audio extractor (or set to eval) if using a pretrained model.
    audio_extractor.eval()  
    running_loss = 0.0

    for waveforms, video_windows, _, _ in dataloader:
        # waveforms: (B, num_windows, window_audio)
        # video_windows: (B, num_windows, window_video, 1, H, W)
        B, num_windows = waveforms.shape[0], waveforms.shape[1]
        optimizer.zero_grad() # iterative as suggested by pytorch maybe?
        batch_loss = 0.0

        for i in range(num_windows):
            wave_i = waveforms[:, i, :]    # (B, window_audio)
            vid_i = video_windows[:, i, ...]  # shape: (B, window_video, 1, H, W)

            # Move to device.
            wave_i = wave_i.to(device)
            vid_i = vid_i.to(device)
            
            # Compute audio condition:
            # Audio extractor returns (B, window_video, feature_dim).
            audio_feats = audio_extractor(wave_i)  
            # Average over the time dimension (window_video) to get (B, cond_dim).
            cond = audio_feats.mean(dim=1)
            
            # Prepare video input for the U-Net:
            # Conditional U-Net expects video input shape (B, 1, T, H, W).
            # Our vid_i is (B, window_video, 1, H, W); we need to permute the dimensions.
            vid_input = vid_i.permute(0, 2, 1, 3, 4)  # becomes (B, 1, window_video, H, W)
            
            # Forward pass through the conditional U-Net.
            with torch.cuda.amp.autocast():
                output = model(vid_input, cond)  # Expected output shape: (B, 1, window_video, H, W)
                loss = criterion(output, vid_input)
            batch_loss += loss
            loss.backward()

        # Average loss over windows.
        batch_loss /= num_windows
        optimizer.step()
        running_loss += batch_loss.item() * B

    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, audio_extractor, dataloader, criterion, device):
    model.eval()
    audio_extractor.eval()
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for waveforms, video_windows, _, _ in dataloader:
            B, num_windows = waveforms.shape[0], waveforms.shape[1]
            batch_loss = 0.0
            for i in range(num_windows):
                wave_i = waveforms[:, i, :]
                if wave_i.dim() == 2:
                    wave_i = wave_i.unsqueeze(1)
                vid_i = video_windows[:, i, ...]
                wave_i = wave_i.to(device)
                vid_i = vid_i.to(device)
                audio_feats = audio_extractor(wave_i)
                cond = audio_feats.mean(dim=1)
                vid_input = vid_i.permute(0, 2, 1, 3, 4)
                output = model(vid_input, cond)
                loss = criterion(output, vid_input)
                batch_loss += loss
            batch_loss /= num_windows
            running_loss += batch_loss.item() * B
            total_samples += B
    return running_loss / total_samples


# ====================================================================
def main():

    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 51)]
    nSubsv = [f"sub{str(i).zfill(3)}" for i in range(51, 75)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.8e-3)
    parser.add_argument("--lr_step", type=int, default=10) # after every 10 epochs the lr is updated by gamma * lr
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--audio_root", type=str, default=audio_root)
    parser.add_argument("--video_root", type=str, default=video_root)
    parser.add_argument("--subs_t", type=list, default=nSubst)
    parser.add_argument("--subs_v", type=list, default=nSubsv)
    parser.add_argument("--filter_keyword", type=str, default=keyword)
    parser.add_argument("--video_max_frames", type=int, default=None)
    parser.add_argument("--audio_sampling_rate", type=int, default=16000)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--sw_window_duration", type=float, default=4.0, help="Sliding window duration in seconds")
    parser.add_argument("--sw_step_duration", type=float, default=4.0, help="Sliding window step in seconds")
    parser.add_argument("--video_fps", type=int, default=83)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cond_unet_sw")
    parser.add_argument("--log_dir", type=str, default="runs/cond_unet_sw")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()