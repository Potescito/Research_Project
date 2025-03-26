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
#
# %% Model
class CrossModalDenoisingModel(nn.Module):
    def __init__(self, audio_model_name="facebook/wav2vec2-base", image_channels=1, image_size=84):
        super(CrossModalDenoisingModel, self).__init__()
        
        # Audio feature extractor (Wav2Vec2)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.audio_feature_dim = self.audio_model.config.hidden_size
        
        # Image feature extractor (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv3d(image_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.image_feature_dim = 32 * (image_size // 4) * (image_size // 4)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.audio_feature_dim, num_heads=4, batch_first=True)
        
        # Denoising decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, audio_waveform, video_frames):
        # Extract audio features
        audio_input = self.audio_processor(audio_waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
        audio_features = self.audio_model(audio_input).last_hidden_state  # [batch, seq_len, audio_feature_dim]
        
        # Extract image features
        batch_size, frames, channels, height, width = video_frames.shape
        video_frames = video_frames.view(batch_size, channels, frames, height, width)  # [batch, 1, frames, 84, 84]
        image_features = self.image_encoder(video_frames)  # [batch, 32, frames/4, 21, 21]
        image_features = image_features.flatten(2).permute(0, 2, 1)  # [batch, frames/4 * 21 * 21, 32]
        
        # Cross-modal attention
        attended_features, _ = self.cross_attention(audio_features, image_features, image_features)  # [batch, seq_len, audio_feature_dim]
        
        # Reshape attended features for decoding
        attended_features = attended_features.permute(0, 2, 1).view(batch_size, 32, frames // 4, height // 4, width // 4)
        
        # Decode to denoised frames
        denoised_frames = self.decoder(attended_features)  # [batch, 1, frames, 84, 84]
        
        return denoised_frames

# Instantiate and test the model
model = CrossModalDenoisingModel().to(device)
for i, (waveform, frames, _, _) in enumerate(dataloader):
    waveform, frames = waveform.to(device), frames.to(device)
    denoised_frames = model(waveform, frames)
    print("Denoised frames shape:", denoised_frames.shape)
    break