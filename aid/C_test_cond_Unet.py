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
from C_wav2vec2_AP import AudioFeatureExtractorFiLM
from C_cond_Unet import ConditionalUNet3D_FiLM

sw_transform = SlidingWindowTransform(4, 4)

def test_model(model, audio_extractor, test_loader, device, output_dir="../data/test_outputs/cond_unet_sw"):
    model.eval()
    audio_extractor.eval()
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.L1Loss()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (waveform, frames, audio_paths, video_paths) in enumerate(test_loader):
            waveform_i = waveform[:,0, :].to(device)
            frames_i = frames[:,0, ...].to(device)
            print(waveform_i.shape, frames_i.shape)
            audio_feats = audio_extractor(waveform_i)  # (B, window_video, feature_dim)
            cond = audio_feats.mean(dim=1) # (B, cond_dim).
            vid_input = frames_i.permute(0, 2, 1, 3, 4)  # becomes B, 1, window_video, H, W
            outputs = model(vid_input, cond)  # Expected output shape: (B, 1, window_video, H, W)
            print(outputs)
            loss = criterion(outputs, vid_input)
            running_loss += loss.item() * waveform_i.size(0)
            recons = sw_transform.overlap_add(outputs.unsqueeze(3))
            print(recons.shape)
            save_videos(recons, video_paths, output_dir=output_dir, fps=83)
    
    test_loss = running_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, recons

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
audio_extractor = AudioFeatureExtractorFiLM(window_video=window_video, 
                                                pretrained_model_name="facebook/wav2vec2-base-960h"
                                                ).to(device)

cond_dim = audio_extractor.feature_dim
model = ConditionalUNet3D_FiLM(cond_dim=cond_dim, base_channels=32).to(device)

checkpoint_path = "checkpoints/cond_unet_sw/cond_unet_film_epoch50.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

test_loss, r = test_model(model, audio_extractor, test_loader, device)
