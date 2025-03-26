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

# %% Model