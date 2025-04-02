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
from src.utils import save_videos
from src.AVDataset import AVDataset
from src.transforms import SlidingWindowTransform
from B_basic_net_sw import BasicDenoisingNetworkSlidingVideo

sw_transform = SlidingWindowTransform(4, 4)

def test_model(model, test_loader, device, output_dir="../data/test_outputs/basic_net_sw"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.L1Loss()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (waveform, frames, audio_paths, video_paths) in enumerate(test_loader):
            waveform = waveform.to(device)
            frames = frames.to(device)
            
            outputs = model(waveform, frames)
            print(outputs)
            loss = criterion(outputs, frames)
            running_loss += loss.item() * waveform.size(0)
            recons = sw_transform.overlap_add(outputs)
            save_videos(recons, video_paths, output_dir=output_dir, fps=83)
    
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


test_dataset = AVDataset(audio_root, video_root, subs=nSubs_test, filter_keyword=keyword, transform=None)
    
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))

# =======
window_audio = int(4*16000)
window_video = int(4*83)
model = BasicDenoisingNetworkSlidingVideo(base_channels=32,
                                            window_audio=window_audio,
                                            window_video=window_video).to(device)

checkpoint_path = "checkpoints/basic_net_sw_single/basic_net_sw_single1.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

test_loss = test_model(model, test_loader, device)
