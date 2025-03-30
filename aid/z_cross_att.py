import sys
sys.path.append('../')
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchinfo
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from src.AVDataset import AVDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.is_available():
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)

# %% Dataset
audio_root = r"../data/audios_denoised_16khz"
video_root = r"../data/dataset_2drt_video_only"
nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
keyword = "vcv"
dataset = AVDataset(audio_root=audio_root, 
                    video_root=video_root, 
                    subs=nSubs, 
                    filter_keyword=keyword, 
                    video_max_frames=None, # batch
                    audio_sampling_rate=16000,
                    frame_skip=1)

print("Number of pairs:", len(dataset))

dataloader = DataLoader(dataset, batch_size=3, shuffle=False, pin_memory=True, collate_fn=AVDataset.collate) # Batch / Collation

for i, (waveform, frames, audio_path, video_path) in enumerate(dataloader):
        print("Audio shape:", waveform.shape) 
        print("Video frames shape:", frames.shape)
        print("Audio file:", audio_path)
        print("Video file:", video_path)
        print("===========")
        if i > 1:
            break

# Audio shape: torch.Size([5, 619695])
# Video frames shape: torch.Size([5, 3216, 1, 84, 84]) ... etc (this is an example)

# %% Model
class CrossModalDenoisingNet(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Audio encoder 
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.hidden_dim = hidden_dim
        # Video encoder
        self.video_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(64, hidden_dim, kernel_size=(3,3,3), padding=(1,1,1))
        )
        
        # Cross attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(768, hidden_dim)  # Wav2Vec2 output dim is 768
        self.value_proj = nn.Linear(768, hidden_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dim, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=(3,3,3), padding=(1,1,1))
        )

    def forward(self, video, audio):
        # Extract audio features
        audio_features = self.audio_encoder(audio).last_hidden_state  # [B, T, 768]
        
        # Extract video features
        B, F, C, H, W = video.shape
        video_features = self.video_encoder(video)  # [B, hidden_dim, F, H, W]
        
        # Reshape video features for attention
        video_features = video_features.permute(0,2,1,3,4)  # [B, F, hidden_dim, H, W]
        video_features = video_features.reshape(B, F, -1)  # [B, F, hidden_dim*H*W]
        
        # Cross attention
        queries = self.query_proj(video_features)
        keys = self.key_proj(audio_features)
        values = self.value_proj(audio_features)
        
        attention_weights = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attended_features = torch.matmul(attention_weights, values)
        
        # Reshape back for decoding
        attended_features = attended_features.reshape(B, F, self.hidden_dim, H, W).permute(0,2,1,3,4)
        
        # Decode
        denoised_video = self.decoder(attended_features)
        
        return denoised_video

model = CrossModalDenoisingNet().to(device)
for i, (waveform, frames, _, _) in enumerate(dataloader):
    waveform, frames = waveform.to(device), frames.to(device)
    denoised_frames = model(waveform, frames)
    print("Denoised frames shape:", denoised_frames.shape)
    break