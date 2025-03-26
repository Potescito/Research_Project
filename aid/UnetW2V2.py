import torch.nn as nn

class UnetW2V2(nn.Module):
    def __init__(self, base_channels=32):
        super(UnetW2V2, self).__init__()

        # Wav2Vec2 projection
        self.audio_proj = nn.Sequential(
            nn.Linear(768, 256),  # 768 is wav2vec2 feature dim
            nn.ReLU(inplace=True),
            nn.Linear(256, 84*84)  # Match video frame size
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2), # can be a bilinear interpolation too
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)
        
    def forward(self, video, audio):
        batch_size, num_frames = video.shape[0], video.shape[1] # shape: (batch_size, num_frames, channels, height, width)
        
        video = video.view(-1, video.shape[2], video.shape[3], video.shape[4]) # Reshape to process all frames at once -> (batch_size*num_frames, channels, height, width)
        
        # Encoder
        e1 = self.enc1(video)
        e2 = self.enc2(e1)
        
        # Bridge
        b = self.bridge(e2)
        
        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        
        # Output
        img_feat = self.final(d2)
        
        # Reshape back to include frames dimension
        img_feat = img_feat.view(batch_size, num_frames, -1, img_feat.shape[2], img_feat.shape[3])

        # Audio projection
        B, T, _ = audio.shape
        audio_proj = self.audio_proj(audio)  # [B, T, H*W]
        audio_proj = audio_proj.view(B, T, 1, 84, 84)  # Match video dimensions
        
        return img_feat, audio_proj

# %% Debug
if __name__ == "__main__":
    from UnetW2V2 import UnetW2V2
    model = UnetW2V2()
    print(model)