"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AudioFeatureExtractorFiLM(nn.Module):
    def __init__(self, window_video, pretrained_model_name="facebook/wav2vec2-base-960h", pre=True):
        """
        Args:
            window_video (int): Num of video frames in the sliding window. The output audio feature sequence will be pooled to have this many time steps.
            pretrained_model_name (str): Name of the pretrained wav2vec2 model.
            pre (bool): If True, uses the preprocessor and memory transfers, if not, it must be done outside.
        """
        super(AudioFeatureExtractorFiLM, self).__init__()
        self.pre = pre
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        for param in self.wav2vec2.parameters(): # inference, no fine tuning
            param.requires_grad = False
        
        self.feature_dim = self.wav2vec2.config.hidden_size  # (seq, 768 feat dim)
        
        # adaptive pooling to force the time dimension to match window video frames
        self.pool = nn.AdaptiveAvgPool1d(window_video)

        # here i just want to project the features to provide a learnable affine transformation of the features
        # may work to finetune the features later on? maybe to match the conditioning better?
        self.projection = nn.Linear(self.feature_dim, self.feature_dim)
    
    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Raw audio waveform of shape (B, window_audio). -> per window 
        Returns:
            torch.Tensor: Audio feature sequence of shape (B, window_video, feature_dim)
        """
        if self.pre:
            inputs = self.processor(waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt") # the input must be normalized [-1, 1]
            inputs = inputs.input_values.to(waveform.device)
        else:
            inputs = waveform
        outputs = self.wav2vec2(inputs)
        features = outputs.last_hidden_state  # encoder (B, T, feature_dim)
        
        # I want to align the time dimension with the video frames in the window.
        features = features.transpose(1, 2)  # (B, feature_dim, T)
        features = self.pool(features)       # (B, feature_dim, window_video)
        features = features.transpose(1, 2)   # (B, window_video, feature_dim)
        
        features = self.projection(features)  # (B, window_video, feature_dim) # maybe it is not needed
        return features

# %%
if __name__ == "__main__":
    import sys
    sys.path.append('../')

    import torch
    from aid.C_wav2vec2_AP import AudioFeatureExtractorFiLM
    from torch.utils.data import DataLoader
    from src.AVDataset import AVDataset
    from src.transforms import SlidingWindowTransform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"
    sw_transform = SlidingWindowTransform(4, 4)

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 3)]
    
    test_dataset = AVDataset(audio_root, video_root, subs=nSubst, filter_keyword=keyword, transform=None) 
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    print(len(test_dataset), len(test_loader))

    Amodel = AudioFeatureExtractorFiLM(window_video=332, pre=True).to(device)

    for i, (waveform, frames, audio_paths, video_paths) in enumerate(test_loader):
        print(i, "==========")
        print(waveform.shape)
        print(frames.shape)
        print(audio_paths)
        print(video_paths)
        feats = Amodel(waveform[:,0,:].to(device)) # tesk only with the first window
        print(waveform[:,0,:].shape, feats.shape)
        # feats = Amodel(waveform[:,0,:].unsqueeze(1)) # NOOOOO
        # print(waveform[:,0,:].unsqueeze(1).shape, feats.shape)
"""
cpu
12 3
Some weights of Wav2Vec2Model were not initialized from the model checkpoint at facebook/wav2vec2-base-960h and are newly initialized: ['wav2vec2.masked_spec_embed']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
0 ==========
torch.Size([4, 9, 64000])
torch.Size([4, 9, 332, 1, 84, 84])
('../data/audios_denoised_16khz/sub001/sub001_2drt_01_vcv1_r1_video.wav', '../data/audios_denoised_16khz/sub001/sub001_2drt_02_vcv2_r2_video.wav', '../data/audios_denoised_16khz/sub001/sub001_2drt_02_vcv2_r1_video.wav', '../data/audios_denoised_16khz/sub001/sub001_2drt_03_vcv3_r2_video.wav')
('../data/dataset_2drt_video_only/sub001/2drt/video/sub001_2drt_01_vcv1_r1_video.mp4', '../data/dataset_2drt_video_only/sub001/2drt/video/sub001_2drt_02_vcv2_r2_video.mp4', '../data/dataset_2drt_video_only/sub001/2drt/video/sub001_2drt_02_vcv2_r1_video.mp4', '../data/dataset_2drt_video_only/sub001/2drt/video/sub001_2drt_03_vcv3_r2_video.mp4')
torch.Size([4, 332, 768])
1 ==========
torch.Size([4, 10, 64000])
torch.Size([4, 10, 332, 1, 84, 84])
('../data/audios_denoised_16khz/sub001/sub001_2drt_01_vcv1_r2_video.wav', '../data/audios_denoised_16khz/sub001/sub001_2drt_03_vcv3_r1_video.wav', '../data/audios_denoised_16khz/sub002/sub002_2drt_02_vcv2_r1_video.wav', '../data/audios_denoised_16khz/sub002/sub002_2drt_01_vcv1_r1_video.wav')
('../data/dataset_2drt_video_only/sub001/2drt/video/sub001_2drt_01_vcv1_r2_video.mp4', '../data/dataset_2drt_video_only/sub001/2drt/video/sub001_2drt_03_vcv3_r1_video.mp4', '../data/dataset_2drt_video_only/sub002/2drt/video/sub002_2drt_02_vcv2_r1_video.mp4', '../data/dataset_2drt_video_only/sub002/2drt/video/sub002_2drt_01_vcv1_r1_video.mp4')
torch.Size([4, 332, 768])
2 ==========
torch.Size([4, 10, 64000])
torch.Size([4, 10, 332, 1, 84, 84])
('../data/audios_denoised_16khz/sub002/sub002_2drt_01_vcv1_r2_video.wav', '../data/audios_denoised_16khz/sub002/sub002_2drt_02_vcv2_r2_video.wav', '../data/audios_denoised_16khz/sub002/sub002_2drt_03_vcv3_r1_video.wav', '../data/audios_denoised_16khz/sub002/sub002_2drt_03_vcv3_r2_video.wav')
('../data/dataset_2drt_video_only/sub002/2drt/video/sub002_2drt_01_vcv1_r2_video.mp4', '../data/dataset_2drt_video_only/sub002/2drt/video/sub002_2drt_02_vcv2_r2_video.mp4', '../data/dataset_2drt_video_only/sub002/2drt/video/sub002_2drt_03_vcv3_r1_video.mp4', '../data/dataset_2drt_video_only/sub002/2drt/video/sub002_2drt_03_vcv3_r2_video.mp4')
torch.Size([4, 332, 768])

"""

# %%
