"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# FiLM Module for the params ---> FiLM: Visual Reasoning with a General Conditioning Layer by Ethan Perez et al.
##############################################
class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        """
        Args:
            cond_dim (int): Dimension of the input condition vector.
            out_channels (int): Number of channels to modulate.
        """
        super(FiLM, self).__init__()
        self.fc = nn.Linear(cond_dim, 2 * out_channels) # scale and shift params are just linear projections of audio feats
    
    def forward(self, condition):
        # condition: (B, cond_dim) -> avg over the wind, feats of audio to get general per window features
        out = self.fc(condition)  # (B, 2*out_channels)
        gamma, beta = out.chunk(2, dim=1)  # each (B, out_channels)
        
        # Reshape to (B, out_channels, 1, 1, 1) so that it can broadcast over (B, out_channels, T, H, W) for video
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma, beta


# Basic 3D Double Conv Block
##############################################
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # pytorch says that bias should be turned of if there is batchnorm
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # 2 convs as in initial Unet paper
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


# Downsampling Block with FiLM conditioning
##############################################
class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super(Down3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2) #spatial and temporal downsampling
        self.conv = DoubleConv3D(in_channels, out_channels)
        self.film = FiLM(cond_dim, out_channels)
    
    def forward(self, x, condition):
        x = self.pool(x)
        x = self.conv(x)
        gamma, beta = self.film(condition)
        x = gamma * x + beta # FiLM paper does it as simple as this
        return x


# Upsampling Block with FiLM conditioning
##############################################
class Up3D(nn.Module):
    def __init__(self, skip_channels, in_channels, out_channels, cond_dim, trilinear=True):
        super(Up3D, self).__init__()
        self.trilinear = trilinear
        up_channels = in_channels // 2
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) # this thing preserves the channel dimension
            self.reduce = nn.Conv3d(in_channels, up_channels, kernel_size=1)
            # self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            # self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2) # use deconvs if not interpolation (check patterns) -> halfs the channel dim
            self.up = nn.ConvTranspose3d(in_channels, up_channels, kernel_size=2, stride=2) # use deconvs if not interpolation (check patterns) -> halfs the channel dim
            # self.conv = DoubleConv3D(out_channels * 2, out_channels)
        self.conv = DoubleConv3D(skip_channels + up_channels, out_channels)
        self.film = FiLM(cond_dim, out_channels) # also here
    
    def forward(self, x1, x2, condition):
        # x1: from decoder (B, C, T, H, W); x2: skip connection from encoder.
        # print("x1a", x1.shape, x2.shape)
        if self.trilinear:
            x1 = self.up(x1)
            x1 = self.reduce(x1)
        else:
            x1 = self.up(x1)
        # print("x1b", x1.shape, x2.shape)
        # Pad if needed (in case the sizes dont match)
        diffT = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffT // 2, diffT - diffT // 2]) # zero pad

        x = torch.cat([x2, x1], dim=1) # concatenation along channel dim of the skip connection
        # print(x.shape, x1.shape, x2.shape)
        x = self.conv(x)
        # Apply FiLM modulation.
        gamma, beta = self.film(condition)
        x = gamma * x + beta
        return x


# Conditional 3D U-Net with FiLM
###################################################################################
class ConditionalUNet3D_FiLM(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, cond_dim=768, base_channels=32):
        """
        Args:
            n_channels (int): Number of input channels (1 grayscale ofc).
            n_classes (int): Number of output channels (1 for denoised output).
            cond_dim (int): Dimension of the condition vector (from audio features).
            base_channels (int): Number of channels in the first layer.
        """
        super(ConditionalUNet3D_FiLM, self).__init__()
        self.inc = DoubleConv3D(n_channels, base_channels)
        self.film0 = FiLM(cond_dim, base_channels) # FiLM after initial conv.
        
        self.down1 = Down3D(base_channels, base_channels * 2, cond_dim)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, cond_dim)
        # Bottleneck.
        self.bot = DoubleConv3D(base_channels * 4, base_channels * 8) # 256 channels, feat maps 
        self.film_bot = FiLM(cond_dim, base_channels * 8)

        self.up1 = Up3D(base_channels * 4, base_channels * 8, base_channels * 4, cond_dim)
        self.up2 = Up3D(base_channels * 2, base_channels * 4, base_channels * 2, cond_dim)
        self.up3 = Up3D(base_channels,     base_channels * 2, base_channels, cond_dim)
        self.outc = nn.Conv3d(base_channels, n_classes, kernel_size=1) # 1x1x1 conv to get the output channels
    
    def forward(self, x, condition):
        """
        Args:
            x (torch.Tensor): Input video window of shape (B, 1, T, H, W)
            condition (torch.Tensor): GLOBAL condition vector of shape (B, cond_dim) -> avg over the window of audio feats
        Returns:
            torch.Tensor: Denoised video of shape (B, 1, T, H, W) -> in fact n_classes but its 1
        """
        x1 = self.inc(x)
        gamma0, beta0 = self.film0(condition)
        x1 = gamma0 * x1 + beta0 # first film
        
        x2 = self.down1(x1, condition)
        x3 = self.down2(x2, condition)
        
        x_bot = self.bot(x3)
        gamma_bot, beta_bot = self.film_bot(condition)
        x_bot = gamma_bot * x_bot + beta_bot # bn film
        
        x = self.up1(x_bot, x3, condition)
        x = self.up2(x, x2, condition)
        x = self.up3(x, x1, condition)
        x = self.outc(x)

        x = torch.sigmoid(x) # optional to normalize the output to [0, 1]
        return x

# %% Debugging
if __name__ == "__main__":
    import torch
    import torchinfo
    from C_cond_Unet import ConditionalUNet3D_FiLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    T, H, W = 332, 84, 84   
    cond_dim = 768 # remember it is an avg over time   
    x = torch.randn(B, 1, T, H, W).to(device) # squeeze 3rd dim
    condition = torch.randn(B, cond_dim).to(device)
    
    model = ConditionalUNet3D_FiLM(cond_dim=cond_dim, base_channels=32).to(device)
    output = model(x, condition)
    print("Output shape:", output.shape)  # Expected: (B, 1, T, H, W)
# %%
    print(torchinfo.summary(model, input_size=(x.shape, condition.shape)))

# %%
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ConditionalUNet3D_FiLM                   [2, 1, 332, 84, 84]       --
├─DoubleConv3D: 1-1                      [2, 32, 332, 84, 84]      --
│    └─Sequential: 2-1                   [2, 32, 332, 84, 84]      --
│    │    └─Conv3d: 3-1                  [2, 32, 332, 84, 84]      864
│    │    └─BatchNorm3d: 3-2             [2, 32, 332, 84, 84]      64
│    │    └─ReLU: 3-3                    [2, 32, 332, 84, 84]      --
│    │    └─Conv3d: 3-4                  [2, 32, 332, 84, 84]      27,648
│    │    └─BatchNorm3d: 3-5             [2, 32, 332, 84, 84]      64
│    │    └─ReLU: 3-6                    [2, 32, 332, 84, 84]      --
├─FiLM: 1-2                              [2, 32, 1, 1, 1]          --
│    └─Linear: 2-2                       [2, 64]                   49,216
├─Down3D: 1-3                            [2, 64, 166, 42, 42]      --
│    └─MaxPool3d: 2-3                    [2, 32, 166, 42, 42]      --
│    └─DoubleConv3D: 2-4                 [2, 64, 166, 42, 42]      --
│    │    └─Sequential: 3-7              [2, 64, 166, 42, 42]      166,144
│    └─FiLM: 2-5                         [2, 64, 1, 1, 1]          --
│    │    └─Linear: 3-8                  [2, 128]                  98,432
├─Down3D: 1-4                            [2, 128, 83, 21, 21]      --
│    └─MaxPool3d: 2-6                    [2, 64, 83, 21, 21]       --
│    └─DoubleConv3D: 2-7                 [2, 128, 83, 21, 21]      --
│    │    └─Sequential: 3-9              [2, 128, 83, 21, 21]      664,064
│    └─FiLM: 2-8                         [2, 128, 1, 1, 1]         --
│    │    └─Linear: 3-10                 [2, 256]                  196,864
├─DoubleConv3D: 1-5                      [2, 256, 83, 21, 21]      --
│    └─Sequential: 2-9                   [2, 256, 83, 21, 21]      --
│    │    └─Conv3d: 3-11                 [2, 256, 83, 21, 21]      884,736
│    │    └─BatchNorm3d: 3-12            [2, 256, 83, 21, 21]      512
│    │    └─ReLU: 3-13                   [2, 256, 83, 21, 21]      --
│    │    └─Conv3d: 3-14                 [2, 256, 83, 21, 21]      1,769,472
│    │    └─BatchNorm3d: 3-15            [2, 256, 83, 21, 21]      512
│    │    └─ReLU: 3-16                   [2, 256, 83, 21, 21]      --
├─FiLM: 1-6                              [2, 256, 1, 1, 1]         --
│    └─Linear: 2-10                      [2, 512]                  393,728
├─Up3D: 1-7                              [2, 128, 83, 21, 21]      --
│    └─Upsample: 2-11                    [2, 256, 166, 42, 42]     --
│    └─Conv3d: 2-12                      [2, 128, 166, 42, 42]     32,896
│    └─DoubleConv3D: 2-13                [2, 128, 83, 21, 21]      --
│    │    └─Sequential: 3-17             [2, 128, 83, 21, 21]      1,327,616
│    └─FiLM: 2-14                        [2, 128, 1, 1, 1]         --
│    │    └─Linear: 3-18                 [2, 256]                  196,864
├─Up3D: 1-8                              [2, 64, 166, 42, 42]      --
│    └─Upsample: 2-15                    [2, 128, 166, 42, 42]     --
│    └─Conv3d: 2-16                      [2, 64, 166, 42, 42]      8,256
│    └─DoubleConv3D: 2-17                [2, 64, 166, 42, 42]      --
│    │    └─Sequential: 3-19             [2, 64, 166, 42, 42]      332,032
│    └─FiLM: 2-18                        [2, 64, 1, 1, 1]          --
│    │    └─Linear: 3-20                 [2, 128]                  98,432
├─Up3D: 1-9                              [2, 32, 332, 84, 84]      --
│    └─Upsample: 2-19                    [2, 64, 332, 84, 84]      --
│    └─Conv3d: 2-20                      [2, 32, 332, 84, 84]      2,080
│    └─DoubleConv3D: 2-21                [2, 32, 332, 84, 84]      --
│    │    └─Sequential: 3-21             [2, 32, 332, 84, 84]      83,072
│    └─FiLM: 2-22                        [2, 32, 1, 1, 1]          --
│    │    └─Linear: 3-22                 [2, 64]                   49,216
├─Conv3d: 1-10                           [2, 1, 332, 84, 84]       33
==========================================================================================
Total params: 6,382,817
Trainable params: 6,382,817
Non-trainable params: 0
Total mult-adds (Units.TERABYTES): 1.19
==========================================================================================
Input size (MB): 18.75
Forward/backward pass size (MB): 15329.94
Params size (MB): 25.53
Estimated Total Size (MB): 15374.22
==========================================================================================
"""