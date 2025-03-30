"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    """
    A simple audio encoder using 1D convolutions.
    Processes the audio waveform (B, T) and returns a feature vector (B, base_channels).
    """
    def __init__(self, in_channels=1, base_channels=32):
        super(AudioEncoder, self).__init__()
        # in_channels is 1 because we treat the waveform as (B, 1, T)
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=5, stride=2, padding=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output shape: (B, base_channels, 1) -> to aggregate features accross time
        
    def forward(self, waveform):
        # waveform shape: (B, T)
        x = waveform.unsqueeze(1)  # shape: (B, 1, T)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.global_pool(x)     # shape: (B, base_channels, 1)
        x = x.squeeze(-1)           # shape: (B, base_channels)
        return x
    

class VideoEncoder(nn.Module):
    """
    A simple video encoder using 2D convolutions.
    Processes the video frames (B, F, 1, H, W) and returns per-frame features (B, F, base_channels, H, W).
    """
    def __init__(self, in_channels=1, base_channels=32):
        super(VideoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, frames):
        # frames shape: (B, F, 1, H, W)
        B, F, C, H, W = frames.size()
        # Process each frame independently.
        frames = frames.view(B * F, C, H, W)
        x = nn.functional.relu(self.conv1(frames))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        # Reshape back to (B, F, base_channels, H, W)
        x = x.view(B, F, -1, H, W)
        return x
    

class Decoder(nn.Module):
    """
    A simple decoder that reconstructs denoised video frames.
    It takes the fused features (B, F, 2*base_channels, H, W) and outputs (B, F, 1, H, W).
    """
    def __init__(self, in_channels=64, base_channels=32):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # x shape: (B, F, in_channels, H, W)
        B, F, C, H, W = x.size()
        # Process each frame independently.
        x = x.view(B * F, C, H, W)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        # Use sigmoid to output normalized pixel values in [0, 1]
        x = torch.sigmoid(self.conv3(x))
        x = x.view(B, F, 1, H, W)
        return x


class BasicDenoisingNetwork(nn.Module):
    """
    Basic denoising network that processes synchronized audio and video segments.
    
    The network has an audio encoder (1D convs) and a video encoder (2D convs). The audio features
    are fused with per-frame video features by spatially replicating the audio feature vector and concatenating.
    The decoder then reconstructs the denoised video frames.
    """
    def __init__(self, base_channels=32):
        super(BasicDenoisingNetwork, self).__init__()
        self.audio_encoder = AudioEncoder(in_channels=1, base_channels=base_channels)
        self.video_encoder = VideoEncoder(in_channels=1, base_channels=base_channels)
        # After fusion, the number of channels is 2*base_channels -> simple concatenation
        self.decoder = Decoder(in_channels=base_channels*2, base_channels=base_channels)
    
    def forward(self, waveform, frames):
        """
        Args:
            waveform (torch.Tensor): Audio waveform of shape (B, T)
            frames (torch.Tensor): Video frames of shape (B, F, 1, H, W)
        
        Returns:
            torch.Tensor: Denoised video frames of shape (B, F, 1, H, W)
        """
        audio_feat = self.audio_encoder(waveform)  # (B, base_channels)
        video_feat = self.video_encoder(frames)      # (B, F, base_channels, H, W)
        
        B, F, C, H, W = video_feat.size()
        # Expand audio features to fuse with each video frame.
        audio_feat_expanded = audio_feat.unsqueeze(1).repeat(1, F, 1)  # (B, F, base_channels)
        # Reshape to (B, F, base_channels, 1, 1) and then expand spatially.
        audio_feat_expanded = audio_feat_expanded.unsqueeze(-1).unsqueeze(-1)
        audio_feat_expanded = audio_feat_expanded.expand(-1, -1, -1, H, W)
        
        fused = torch.cat([video_feat, audio_feat_expanded], dim=2)  # (B, F, 2*base_channels, H, W)
        
        denoised_frames = self.decoder(fused)  # (B, F, 1, H, W)
        return denoised_frames
    
# %% Debugging
if __name__ == "__main__":
    import torch
    import torchinfo
    from 0_basic_net import BasicDenoisingNetwork
    batch_size = 4
    audio_length = 16000  
    num_frames = 83     
    height, width = 84, 84

    dummy_waveform = torch.randn(batch_size, audio_length)
    dummy_frames = torch.randn(batch_size, num_frames, 1, height, width)

    net = BasicDenoisingNetwork(base_channels=32)

    denoised_video = net(dummy_waveform, dummy_frames) # one fwd pass
    print("Denoised video shape:", denoised_video.shape)  # Expected: (B, F, 1, H, W)
    print(torchinfo.summary(net, input_size=(dummy_waveform.shape, dummy_frames.shape)))
# %%
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
BasicDenoisingNetwork                    [4, 83, 1, 84, 84]        --
├─AudioEncoder: 1-1                      [4, 32]                   --
│    └─Conv1d: 2-1                       [4, 32, 8000]             192
│    └─Conv1d: 2-2                       [4, 32, 4000]             5,152
│    └─Conv1d: 2-3                       [4, 32, 2000]             5,152
│    └─AdaptiveAvgPool1d: 2-4            [4, 32, 1]                --
├─VideoEncoder: 1-2                      [4, 83, 32, 84, 84]       --
│    └─Conv2d: 2-5                       [332, 32, 84, 84]         320
│    └─Conv2d: 2-6                       [332, 32, 84, 84]         9,248
│    └─Conv2d: 2-7                       [332, 32, 84, 84]         9,248
├─Decoder: 1-3                           [4, 83, 1, 84, 84]        --
│    └─Conv2d: 2-8                       [332, 32, 84, 84]         18,464
│    └─Conv2d: 2-9                       [332, 32, 84, 84]         9,248
│    └─Conv2d: 2-10                      [332, 1, 84, 84]          289
==========================================================================================
Total params: 57,313
Trainable params: 57,313
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 109.80
==========================================================================================
Input size (MB): 9.63
Forward/backward pass size (MB): 3031.59
Params size (MB): 0.23
Estimated Total Size (MB): 3041.45
==========================================================================================
"""