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


I want to write a model that performs frame denoising in my set of MRI videos with the additional help of the speech information of the videos which is linked to the anatomical movements in the videos. The videos come in the shape [batch, frames, channel=1, 84, 84] and the audio comes in the shape [batch, samples]. I want to come with a solution that implements cross model attention to align the audio features with the image features and attent to the relevant parts to make the denoising. I must use the wav2vec2 to extract the audio features. Implement that. keep in mind that I have already written the dataloader. 