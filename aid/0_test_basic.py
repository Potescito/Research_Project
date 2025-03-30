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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import save_waveforms, save_videos
from src.AVDataset import AVDataset
from src.transforms import TemporalWindowTransform, ContextualSamplingTransform
from aid.basic_net import BasicDenoisingNetwork


def test_model(model, test_loader, device, output_dir="../data/test_outputs/basic_net"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.L1Loss()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (waveform, frames, audio_paths, video_paths) in enumerate(test_loader):
            waveform = waveform.to(device)
            frames = frames.to(device)
            
            outputs = model(waveform, frames)
            loss = criterion(outputs, frames)
            running_loss += loss.item() * waveform.size(0)
            
            # sample_video = outputs[0].unsqueeze(0)  # shape: (1, F, 1, H, W)
            # sample_output_dir = os.path.join(output_dir, f"sub_{batch_idx}")
            
            # os.makedirs(sample_output_dir, exist_ok=True)
            save_videos(outputs, video_paths, output_dir=output_dir, fps=83)
    
    test_loss = running_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

# =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
audio_root = r"../data/audios_denoised_16khz"
video_root = r"../data/dataset_2drt_video_only"
keyword = "vcv"

nSubs_test = [f"sub{str(i).zfill(3)}" for i in range(75, 76)] # last subject
temporal_transform = TemporalWindowTransform(window_size_sec=28, audio_sample_rate=16000, video_fps=83)
contextual_transform = ContextualSamplingTransform(context_size=1, audio_sample_rate=16000, video_fps=83)

test_dataset = AVDataset(audio_root, video_root, subs=nSubs_test, filter_keyword=keyword, transform=temporal_transform)
    
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=AVDataset.collate)

# =======
model = BasicDenoisingNetwork(base_channels=32).to(device)

checkpoint_path = "checkpoints/basic_net/basic_net_50.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

test_loss = test_model(model, test_loader, device)
