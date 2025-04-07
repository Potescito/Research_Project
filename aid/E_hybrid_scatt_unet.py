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


# FiLM Module - reused 
##############################################
class FiLM(nn.Module):
    """
        Args:
            cond_dim (int): Dimension of the input condition vector.
            out_channels (int): Number of channels to modulate.
    """
    def __init__(self, cond_dim, out_channels):
        super(FiLM, self).__init__()
        self.fc = nn.Linear(cond_dim, 2 * out_channels)
    
    def forward(self, condition):
        # condition: (B, cond_dim)
        out = self.fc(condition)  # (B, 2*out_channels)
        gamma, beta = out.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma, beta


# 3D Double Conv Block
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
        # self.film = FiLM(cond_dim, out_channels)
    
    def forward(self, x, condition):
        x = self.pool(x)
        x = self.conv(x)
        # gamma, beta = self.film(condition)
        # x = gamma * x + beta # FiLM paper does it as simple as this
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
        # self.film = FiLM(cond_dim, out_channels) # also here
    
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
        # gamma, beta = self.film(condition)
        # x = gamma * x + beta
        return x


# CAB
##############################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, video_channels, audio_dim, embed_dim, num_heads=4):
        """
        Args:
            video_channels (int): Number of channels in the video feature map.
            audio_dim (int): Dimension of the audio features.
            embed_dim (int): Embedding dimension for attention.
            num_heads (int): Number of attention heads.
        """
        super(CrossAttentionBlock, self).__init__()
        self.video_proj = nn.Conv3d(video_channels, embed_dim, kernel_size=1, bias=False) # 1x1x1 conv to project video feats 
        self.audio_proj = nn.Linear(audio_dim, embed_dim, bias=False) # fcn to project audio feats
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Conv3d(embed_dim, video_channels, kernel_size=1, bias=False) # sort of back projection
    
    def forward(self, video_feat, audio_feat):
        """
        Args:
            video_feat: (B, video_channels, T, H, W) -> transpose
            audio_feat: (B, T, audio_dim)  -- temporally downsampled to match T.
        Returns:
            (B, video_channels, T, H, W) with fused information.
        """
        B, C, T, H, W = video_feat.shape
        
        v_emb = self.video_proj(video_feat)  # (B, embed_dim, T, H, W)
        v_emb_flat = v_emb.view(B, -1, T * H * W).permute(0, 2, 1)  # (B, T*H*W, embed_dim) -> QUERIES
        
        a_emb = self.audio_proj(audio_feat)  # (B, T, embed_dim)
        a_emb_exp = a_emb.unsqueeze(-1).unsqueeze(-1).expand(B, T, a_emb.size(-1), H, W) # to match video feats shape
        a_emb_flat = a_emb_exp.contiguous().view(B, T * H * W, -1)  # (B, T*H*W, embed_dim)
        
        attn_out, _ = self.attn(v_emb_flat, a_emb_flat, a_emb_flat, need_weights=False)  # (B, T*H*W, embed_dim)
        attn_out = attn_out.permute(0, 2, 1).view(B, -1, T, H, W)  # (B, embed_dim, T, H, W) -> attended output
        out = self.out_proj(attn_out) 
        return out + video_feat # residual connection to keep info from the original features!

# SAB
###############################################
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Transformer-like architecture may help?
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        # x: (B, T, embed_dim) -> flattened temporal dimension.
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln(x + attn_out)
        x = self.ln(x + self.ffn(x))
        return x


# HybridAttentionUNet
##############################################
class HybridAttentionUNet(nn.Module):
    """
        Args:
            n_channels (int): Input channels.
            n_classes (int): Output channels.
            audio_dim (int): Audio feature dimension.
            base_channels (int): Base channels.
            embed_dim (int): Embedding dimension for cross-attention.
            num_heads (int): Attention heads.
        """
    def __init__(self, n_channels=1, n_classes=1, audio_dim=768, base_channels=32, embed_dim=128, num_heads=4):
        super(HybridAttentionUNet, self).__init__()
        # enc
        self.inc = DoubleConv3D(n_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2, cond_dim=audio_dim)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, cond_dim=audio_dim)

        self.self_attn_enc = SelfAttentionBlock(embed_dim=base_channels * 4, num_heads=num_heads)
        
        # bot
        self.bot = DoubleConv3D(base_channels * 4, base_channels * 8)
        # self.film_bot = FiLM(audio_dim, base_channels * 8) # should I take this out?

        self.cross_attn = CrossAttentionBlock(video_channels=base_channels * 8,
                                              audio_dim=audio_dim,
                                              embed_dim=embed_dim,
                                              num_heads=num_heads)
        
        # dec
        self.up1 = Up3D(skip_channels=base_channels * 4, in_channels=base_channels * 8, out_channels=base_channels * 4, cond_dim=audio_dim, trilinear=False)
        self.self_attn_dec1 = SelfAttentionBlock(embed_dim=base_channels * 4, num_heads=num_heads)
        self.up2 = Up3D(skip_channels=base_channels * 2, in_channels=base_channels * 4, out_channels=base_channels * 2, cond_dim=audio_dim, trilinear=False)
        self.self_attn_dec2 = SelfAttentionBlock(embed_dim=base_channels * 2, num_heads=num_heads)
        self.up3 = Up3D(skip_channels=base_channels, in_channels=base_channels * 2, out_channels=base_channels, cond_dim=audio_dim, trilinear=False)
        self.outc = nn.Conv3d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x, audio_condition):
        """
        x: video input of shape (B, 1, T, H, W)
        audio_condition: audio features (B, T, audio_dim)
        """
        cond_pool = audio_condition.mean(dim=1)
        
        x1 = self.inc(x)  # (B, base_channels, T, H, W) -> no film here to see if there's any effect 
        
        x2 = self.down1(x1, cond_pool)  # (B, base_channels*2, T/2, H/2, W/2)
        x3 = self.down2(x2, cond_pool)  # (B, base_channels*4, T/4, H/4, W/4)
        
        # SAB in encoder block 2 -> flaten
        B, C, T_enc, H_enc, W_enc = x3.shape
        x3_flat = x3.view(B, C, -1).permute(0, 2, 1)  # (B, T_enc*H_enc*W_enc, C)
        x3_attn = self.self_attn_enc(x3_flat)
        x3 = x3_attn.permute(0, 2, 1).view(B, C, T_enc, H_enc, W_enc)
        
        x_bot = self.bot(x3)  # (B, base_channels*8, T/4, H/4, W/4)
        # gamma_bot, bea_bot = self.film_bot(cond_pool)
        # x_bot = gamma_bot * x_bot + beta_bot
        
        # Downsample audio condition to match bottleneck time dimension T/4 (time of x_bot)
        T_bot = x_bot.size(2)
        cond_bot = F.adaptive_avg_pool1d(audio_condition.transpose(1,2), T_bot).transpose(1,2)
        
        # CAB at bottleneck
        x_bot = self.cross_attn(x_bot, cond_bot)
        
        x = self.up1(x_bot, x3, cond_pool)
        # SAB in decoder block 1.
        B, C, T_dec, H_dec, W_dec = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_attn = self.self_attn_dec1(x_flat)
        x = x_attn.permute(0, 2, 1).view(B, C, T_dec, H_dec, W_dec)
        
        x = self.up2(x, x2, cond_pool)
        B, C, T_dec, H_dec, W_dec = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_attn = self.self_attn_dec2(x_flat)
        x = x_attn.permute(0, 2, 1).view(B, C, T_dec, H_dec, W_dec)
        
        x = self.up3(x, x1, cond_pool)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x

# %% Debugging
if __name__ == "__main__":
    import torch
    import torchinfo
    from E_hybrid_scatt_unet import HybridAttentionUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    B = 2
    T, H, W = 166, 84, 84
    audio_dim = 768  
    x = torch.randn(B, 1, T, H, W).to(device)
    audio_condition = torch.randn(B, T, audio_dim).to(device) # avg is done inside
    
    model = HybridAttentionUNet(n_channels=1, n_classes=1, audio_dim=audio_dim, base_channels=32, embed_dim=128, num_heads=4).to(device)
    output = model(x, audio_condition)
    print("Output shape:", output.shape)  # Expected: (B, 1, T, H, W)

# %%
    print(torchinfo.summary(model, input_size=(x.shape, audio_condition.shape)))

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
HybridAttentionUNet                      [2, 1, 16, 84, 84]        --
├─DoubleConv3D: 1-1                      [2, 32, 16, 84, 84]       --
│    └─Sequential: 2-1                   [2, 32, 16, 84, 84]       --
│    │    └─Conv3d: 3-1                  [2, 32, 16, 84, 84]       864
│    │    └─BatchNorm3d: 3-2             [2, 32, 16, 84, 84]       64
│    │    └─ReLU: 3-3                    [2, 32, 16, 84, 84]       --
│    │    └─Conv3d: 3-4                  [2, 32, 16, 84, 84]       27,648
│    │    └─BatchNorm3d: 3-5             [2, 32, 16, 84, 84]       64
│    │    └─ReLU: 3-6                    [2, 32, 16, 84, 84]       --
├─Down3D: 1-2                            [2, 64, 8, 42, 42]        --
│    └─MaxPool3d: 2-2                    [2, 32, 8, 42, 42]        --
│    └─DoubleConv3D: 2-3                 [2, 64, 8, 42, 42]        --
│    │    └─Sequential: 3-7              [2, 64, 8, 42, 42]        166,144
│    └─FiLM: 2-4                         [2, 64, 1, 1, 1]          --
│    │    └─Linear: 3-8                  [2, 128]                  98,432
├─Down3D: 1-3                            [2, 128, 4, 21, 21]       --
│    └─MaxPool3d: 2-5                    [2, 64, 4, 21, 21]        --
│    └─DoubleConv3D: 2-6                 [2, 128, 4, 21, 21]       --
│    │    └─Sequential: 3-9              [2, 128, 4, 21, 21]       664,064
│    └─FiLM: 2-7                         [2, 128, 1, 1, 1]         --
│    │    └─Linear: 3-10                 [2, 256]                  196,864
├─SelfAttentionBlock: 1-4                [2, 1764, 128]            --
│    └─MultiheadAttention: 2-8           [2, 1764, 128]            66,048
│    └─LayerNorm: 2-9                    [2, 1764, 128]            256
│    └─Sequential: 2-10                  [2, 1764, 128]            --
│    │    └─Linear: 3-11                 [2, 1764, 128]            16,512
│    │    └─ReLU: 3-12                   [2, 1764, 128]            --
│    │    └─Linear: 3-13                 [2, 1764, 128]            16,512
│    └─LayerNorm: 2-11                   [2, 1764, 128]            (recursive)
├─DoubleConv3D: 1-5                      [2, 256, 4, 21, 21]       --
│    └─Sequential: 2-12                  [2, 256, 4, 21, 21]       --
│    │    └─Conv3d: 3-14                 [2, 256, 4, 21, 21]       884,736
│    │    └─BatchNorm3d: 3-15            [2, 256, 4, 21, 21]       512
│    │    └─ReLU: 3-16                   [2, 256, 4, 21, 21]       --
│    │    └─Conv3d: 3-17                 [2, 256, 4, 21, 21]       1,769,472
│    │    └─BatchNorm3d: 3-18            [2, 256, 4, 21, 21]       512
│    │    └─ReLU: 3-19                   [2, 256, 4, 21, 21]       --
├─FiLM: 1-6                              [2, 256, 1, 1, 1]         --
│    └─Linear: 2-13                      [2, 512]                  393,728
├─CrossAttentionBlock: 1-7               [2, 256, 4, 21, 21]       --
│    └─Conv3d: 2-14                      [2, 128, 4, 21, 21]       32,768
│    └─Linear: 2-15                      [2, 4, 128]               98,304
│    └─MultiheadAttention: 2-16          [2, 1764, 128]            66,048
│    └─Conv3d: 2-17                      [2, 256, 4, 21, 21]       32,768
├─Up3D: 1-8                              [2, 128, 4, 21, 21]       --
│    └─ConvTranspose3d: 2-18             [2, 128, 8, 42, 42]       262,272
│    └─DoubleConv3D: 2-19                [2, 128, 4, 21, 21]       --
│    │    └─Sequential: 3-20             [2, 128, 4, 21, 21]       1,327,616
│    └─FiLM: 2-20                        [2, 128, 1, 1, 1]         --
│    │    └─Linear: 3-21                 [2, 256]                  196,864
├─SelfAttentionBlock: 1-9                [2, 1764, 128]            --
│    └─MultiheadAttention: 2-21          [2, 1764, 128]            66,048
│    └─LayerNorm: 2-22                   [2, 1764, 128]            256
│    └─Sequential: 2-23                  [2, 1764, 128]            --
│    │    └─Linear: 3-22                 [2, 1764, 128]            16,512
│    │    └─ReLU: 3-23                   [2, 1764, 128]            --
│    │    └─Linear: 3-24                 [2, 1764, 128]            16,512
│    └─LayerNorm: 2-24                   [2, 1764, 128]            (recursive)
├─Up3D: 1-10                             [2, 64, 8, 42, 42]        --
│    └─ConvTranspose3d: 2-25             [2, 64, 8, 42, 42]        65,600
│    └─DoubleConv3D: 2-26                [2, 64, 8, 42, 42]        --
│    │    └─Sequential: 3-25             [2, 64, 8, 42, 42]        332,032
│    └─FiLM: 2-27                        [2, 64, 1, 1, 1]          --
│    │    └─Linear: 3-26                 [2, 128]                  98,432
├─SelfAttentionBlock: 1-11               [2, 14112, 64]            --
│    └─MultiheadAttention: 2-28          [2, 14112, 64]            16,640
│    └─LayerNorm: 2-29                   [2, 14112, 64]            128
│    └─Sequential: 2-30                  [2, 14112, 64]            --
│    │    └─Linear: 3-27                 [2, 14112, 64]            4,160
│    │    └─ReLU: 3-28                   [2, 14112, 64]            --
│    │    └─Linear: 3-29                 [2, 14112, 64]            4,160
│    └─LayerNorm: 2-31                   [2, 14112, 64]            (recursive)
├─Up3D: 1-12                             [2, 32, 16, 84, 84]       --
│    └─ConvTranspose3d: 2-32             [2, 32, 16, 84, 84]       16,416
│    └─DoubleConv3D: 2-33                [2, 32, 16, 84, 84]       --
│    │    └─Sequential: 3-30             [2, 32, 16, 84, 84]       83,072
│    └─FiLM: 2-34                        [2, 32, 1, 1, 1]          --
│    │    └─Linear: 3-31                 [2, 64]                   49,216
├─Conv3d: 1-13                           [2, 1, 16, 84, 84]        33
==========================================================================================
Total params: 7,088,289
Trainable params: 7,088,289
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 68.80
==========================================================================================
Input size (MB): 1.00
Forward/backward pass size (MB): 836.36
Params size (MB): 27.49
Estimated Total Size (MB): 864.86
==========================================================================================
"""

# %%
