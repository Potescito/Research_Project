"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


# Audio Encoder for Sliding Window Approach
##############################################
class AudioEncoderSliding(nn.Module):
    def __init__(self, base_channels=32, window_audio=64000, window_video=332):
        """
        Processes each window and hen aggregates feats
        across time using adaptive pooling with the same number of
        time steps as video frames in the window. That aligns audio feats 
        temporarily with video feats.
        Args:
            base_channels (int): Number of feature channels.
            window_audio (int): Number of audio samples per window.
            window_video (int): Number of video frames per window.
        """
        super(AudioEncoderSliding, self).__init__()
        # I process audio as 1D signal.
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=5, stride=2, padding=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(window_video) # Adaptive pooling to get exactly window_video time steps.
        
    def forward(self, x):
        # x: (B, 1, window_audio)
        x = nn_functional.relu(self.conv1(x))
        x = nn_functional.relu(self.conv2(x))
        x = nn_functional.relu(self.conv3(x))
        x = self.adaptive_pool(x)  # (B, base_channels, window_video)
        return x


# Video Encoder for Sliding Window Approach
##############################################
class VideoEncoderSliding(nn.Module):
    """
    This also processes each video window individually.
    It uses 2D convolutions per frame! to extract features for each f.
    """
    def __init__(self, base_channels=32):
        super(VideoEncoderSliding, self).__init__()
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # x: (B, window_video, 1, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # here I just process batch*frames at once
        x = nn_functional.relu(self.conv1(x))
        x = nn_functional.relu(self.conv2(x))
        x = nn_functional.relu(self.conv3(x))
        x = x.view(B, T, -1, H, W)  # (B, window_video, base_channels, H, W)
        return x


# 3D Decoder for Sliding Window Approach
##############################################
class DecoderSliding(nn.Module):
    """
    Processes the fused feats and outputs a "denoised" segment
    for each window.
    """
    def __init__(self, base_channels=32):
        super(DecoderSliding, self).__init__()
        # 3D convolutions to decode fused features.
        self.conv3d_1 = nn.Conv3d(2 * base_channels, base_channels, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3d_2 = nn.Conv3d(base_channels, base_channels, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3d_3 = nn.Conv3d(base_channels, 1, kernel_size=(3,3,3), stride=1, padding=1)
        
    def forward(self, x):
        # x: (B, 2*base_channels, window_video, H, W)
        x = nn_functional.relu(self.conv3d_1(x))
        x = nn_functional.relu(self.conv3d_2(x))
        x = torch.sigmoid(self.conv3d_3(x)) # normalized pixel values
        return x  # (B, 1, window_video, H, W)


# Complete Denoising Network for Sliding Windows
##############################################
class BasicDenoisingNetworkSlidingVideo(nn.Module):
    def __init__(self, base_channels=32, window_audio=64000, window_video=332):
        super(BasicDenoisingNetworkSlidingVideo, self).__init__()
        self.audio_encoder = AudioEncoderSliding(base_channels=base_channels, 
                                                 window_audio=window_audio, 
                                                 window_video=window_video)
        self.video_encoder = VideoEncoderSliding(base_channels=base_channels)
        self.decoder = DecoderSliding(base_channels=base_channels)
        self.base_channels = base_channels
        
    def forward(self, waveform, frames):
        """
        Here I assume that theres no overlap for simplicity.!

        Args:
            waveform: (B, num_windows, window_audio)
            frames: (B, num_windows, window_video, 1, H, W)
        Returns:
            segments: (B, num_windows, window_video, 1, H, W) -- denoised video segments per window.
        """
        B, num_windows, window_audio = waveform.shape
        _, _, window_video, _, H, W = frames.shape # num_windows should be the same
        
        # Audio windows______________________________________
        # Reshape to (B*num_windows, window_audio) and add channel dimension.
        audio_in = waveform.view(B * num_windows, window_audio).unsqueeze(1)  # (B*num_windows, 1, window_audio)
        audio_feat = self.audio_encoder(audio_in)  # (B*num_windows, base_channels, window_video)
        # Reshape to (B, num_windows, base_channels, window_video)
        audio_feat = audio_feat.view(B, num_windows, self.base_channels, window_video)
        
        # Video windows_______________________________________
        # Reshape frames: (B, num_windows, window_video, 1, H, W) -> (B*num_windows, window_video, 1, H, W)
        video_in = frames.view(B * num_windows, window_video, 1, H, W)
        video_feat = self.video_encoder(video_in)  # (B*num_windows, window_video, base_channels, H, W)
        # Reshape to (B, num_windows, window_video, base_channels, H, W)
        video_feat = video_feat.view(B, num_windows, window_video, self.base_channels, H, W)
        
        # Fuse audio and video features_______________________
        audio_feat = audio_feat.permute(0, 1, 3, 2)  # (B, num_windows, window_video, base_channels)
        audio_feat_expanded = audio_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, H, W) # expansion to match video feats
        
        # fused: (B, num_windows, window_video, 2*base_channels, H, W)
        fused = torch.cat([video_feat, audio_feat_expanded], dim=3)
        fused = fused.view(B * num_windows, 2 * self.base_channels, window_video, H, W)
        
        # Decode each window segmen____________________________
        decoded = self.decoder(fused)  # (B*num_windows, 1, window_video, H, W)
        decoded = decoded.view(B, num_windows, window_video, 1, H, W)
        
        return decoded
    

# %% Debugging
if __name__ == "__main__":
    import torch
    import torchinfo
    from B_basic_net_sw import BasicDenoisingNetworkSlidingVideo

    B = 2
    num_windows = 5
    window_audio = 16000
    window_video = 83
    H, W = 84, 84
    device = "cuda"
    
    dummy_waveform = torch.randn((B, num_windows, window_audio), device=device)
    dummy_frames = torch.randn((B, num_windows, window_video, 1, H, W), device=device)
    
    net = BasicDenoisingNetworkSlidingVideo(base_channels=32, window_audio=window_audio, window_video=window_video).to(device)
    
    output_segments = net(dummy_waveform, dummy_frames)
    print("Output segments shape:", output_segments.shape) # (B, num_windows, window_video, 1, H, W)
    print(torchinfo.summary(net, input_size=(dummy_waveform.shape, dummy_frames.shape), device=device))
    
# %%
""" 
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
BasicDenoisingNetworkSlidingVideo        [2, 5, 83, 1, 84, 84]     --
├─AudioEncoderSliding: 1-1               [10, 32, 83]              --
│    └─Conv1d: 2-1                       [10, 32, 8000]            192
│    └─Conv1d: 2-2                       [10, 32, 4000]            5,152
│    └─Conv1d: 2-3                       [10, 32, 2000]            5,152
│    └─AdaptiveAvgPool1d: 2-4            [10, 32, 83]              --
├─VideoEncoderSliding: 1-2               [10, 83, 32, 84, 84]      --
│    └─Conv2d: 2-5                       [830, 32, 84, 84]         320
│    └─Conv2d: 2-6                       [830, 32, 84, 84]         9,248
│    └─Conv2d: 2-7                       [830, 32, 84, 84]         9,248
├─DecoderSliding: 1-3                    [10, 1, 83, 84, 84]       --
│    └─Conv3d: 2-8                       [10, 32, 83, 84, 84]      55,328
│    └─Conv3d: 2-9                       [10, 32, 83, 84, 84]      27,680
│    └─Conv3d: 2-10                      [10, 1, 83, 84, 84]       865
==========================================================================================
Total params: 113,185
Trainable params: 113,185
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 601.72
==========================================================================================
Input size (MB): 24.07
Forward/backward pass size (MB): 7578.99
Params size (MB): 0.45
Estimated Total Size (MB): 7603.51
==========================================================================================
"""