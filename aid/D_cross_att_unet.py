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

##############################################
# Cross-Attention Block
##############################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, video_channels, audio_dim, embed_dim, num_heads=4):
        """
        Args:
            video_channels (int): Number of channels in the video feature map.
            audio_dim (int): Dimension of the audio features.
            embed_dim (int): Common embedding dimension for attention.
            num_heads (int): Number of attention heads.
        """
        super(CrossAttentionBlock, self).__init__()
        # Project video features to embedding space.
        self.video_proj = nn.Conv3d(video_channels, embed_dim, kernel_size=1, bias=False)
        # Project audio features.
        self.audio_proj = nn.Linear(audio_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Project back to video channels.
        self.out_proj = nn.Conv3d(embed_dim, video_channels, kernel_size=1, bias=False)
    
    def forward(self, video_feat, audio_feat):
        """
        Args:
            video_feat: (B, video_channels, T, H, W)
            audio_feat: (B, T, audio_dim)  -- assumed to be aligned temporally with video_feat.
        Returns:
            Tensor: (B, video_channels, T, H, W) with fused information.
        """
        B, C, T, H, W = video_feat.shape
        # Project video features.
        v_emb = self.video_proj(video_feat)  # (B, embed_dim, T, H, W)
        # Flatten spatial and temporal dims: (B, T*H*W, embed_dim)
        v_emb_flat = v_emb.view(B, -1, T * H * W).permute(0, 2, 1)
        
        # Process audio features.
        # Project audio features: (B, T, audio_dim) -> (B, T, embed_dim)
        a_emb = self.audio_proj(audio_feat)  
        # Expand audio along spatial dims.
        # First, unsqueeze to (B, T, embed_dim, 1, 1) and then expand:
        a_emb_exp = a_emb.unsqueeze(-1).unsqueeze(-1).expand(B, T, a_emb.size(-1), H, W)
        # Flatten to (B, T*H*W, embed_dim)
        a_emb_flat = a_emb_exp.contiguous().view(B, T * H * W, -1)
        
        # Now perform multihead attention: 
        # Query: v_emb_flat, Key/Value: a_emb_flat.
        attn_out, _ = self.attn(v_emb_flat, a_emb_flat, a_emb_flat)  # (B, T*H*W, embed_dim)
        # Reshape back to (B, embed_dim, T, H, W)
        attn_out = attn_out.permute(0, 2, 1).view(B, -1, T, H, W)
        # Project back to video_channels.
        out = self.out_proj(attn_out)
        # Residual connection.
        return out + video_feat

##############################################
# Conditional U-Net with Cross-Attention Fusion
##############################################
class ConditionalUNet3D_CrossAttn(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, audio_dim=768, base_channels=32, embed_dim=128, num_heads=4):
        """
        Args:
            n_channels (int): Number of input channels (e.g., 1 for grayscale).
            n_classes (int): Number of output channels.
            audio_dim (int): Dimension of the audio features.
            base_channels (int): Base number of channels for U-Net.
            embed_dim (int): Embedding dimension for cross-attention.
            num_heads (int): Number of attention heads.
        """
        super(ConditionalUNet3D_CrossAttn, self).__init__()
        self.inc = DoubleConv3D(n_channels, base_channels)
        # Downsampling blocks.
        self.down1 = Down3D(base_channels, base_channels * 2, cond_dim=audio_dim)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, cond_dim=audio_dim)
        self.bot = DoubleConv3D(base_channels * 4, base_channels * 8)
        # Apply cross-attention at the bottleneck.
        self.cross_attn = CrossAttentionBlock(video_channels=base_channels * 8,
                                              audio_dim=audio_dim,
                                              embed_dim=embed_dim,
                                              num_heads=num_heads)
        # Upsampling blocks.
        # For simplicity, we use the original Up3D blocks with FiLM removed;
        # you can also add cross-attention in the decoder if desired.
        self.up1 = Up3D(skip_channels=base_channels * 4, up_channels=base_channels * 8, out_channels=base_channels * 4, cond_dim=audio_dim, bilinear=False)
        self.up2 = Up3D(skip_channels=base_channels * 2, up_channels=base_channels * 4, out_channels=base_channels * 2, cond_dim=audio_dim, bilinear=False)
        self.up3 = Up3D(skip_channels=base_channels, up_channels=base_channels * 2, out_channels=base_channels, cond_dim=audio_dim, bilinear=False)
        self.outc = nn.Conv3d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x, audio_condition):
        """
        Args:
            x (torch.Tensor): Input video window of shape (B, 1, T, H, W).
            audio_condition (torch.Tensor): Audio features for cross-attention, shape (B, T, audio_dim).
        Returns:
            torch.Tensor: Denoised video of shape (B, n_classes, T, H, W).
        """
        x1 = self.inc(x)  # (B, base_channels, T, H, W)
        x2 = self.down1(x1, audio_condition)  # (B, base_channels*2, T/2, H/2, W/2)
        x3 = self.down2(x2, audio_condition)  # (B, base_channels*4, T/4, H/4, W/4)
        x_bot = self.bot(x3)  # (B, base_channels*8, T/4, H/4, W/4)
        # Apply cross-attention fusion at bottleneck.
        x_bot = self.cross_attn(x_bot, audio_condition)
        x = self.up1(x_bot, x3, audio_condition)
        x = self.up2(x, x2, audio_condition)
        x = self.up3(x, x1, audio_condition)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x

##############################################
# Helper modules reused from FiLM implementation
##############################################
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super(Down3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = DoubleConv3D(in_channels, out_channels)
        # Here, we use a simple FiLM block to condition the downsampled features.
        # Alternatively, you could use a simple linear projection.
        self.film = FiLM(cond_dim, out_channels)
    
    def forward(self, x, condition):
        x = self.pool(x)
        x = self.conv(x)
        gamma, beta = self.film(condition)
        return gamma * x + beta

class Up3D(nn.Module):
    def __init__(self, skip_channels, up_channels, out_channels, cond_dim, bilinear=True):
        super(Up3D, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.reduce = nn.Conv3d(up_channels, up_channels // 2, kernel_size=1)
            up_channels = up_channels // 2
        else:
            self.up = nn.ConvTranspose3d(up_channels, up_channels // 2, kernel_size=2, stride=2)
            up_channels = up_channels // 2
        
        self.conv = DoubleConv3D(skip_channels + up_channels, out_channels)
        self.film = FiLM(cond_dim, out_channels)
    
    def forward(self, x1, x2, condition):
        x1 = self.up(x1)
        if hasattr(self, 'reduce'):
            x1 = self.reduce(x1)
        diffT = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffT // 2, diffT - diffT // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        gamma, beta = self.film(condition)
        return gamma * x + beta

class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super(FiLM, self).__init__()
        self.fc = nn.Linear(cond_dim, 2 * out_channels)
    
    def forward(self, condition):
        out = self.fc(condition)
        gamma, beta = out.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma, beta

##############################################
# Example usage of ConditionalUNet3D_CrossAttn
##############################################
if __name__ == "__main__":
    B = 2
    T, H, W = 16, 84, 84
    audio_dim = 768  # for instance, the hidden size of wav2vec2
    x = torch.randn(B, 1, T, H, W)
    # Dummy audio features: assume they are time-aligned with video frames, shape (B, T, audio_dim)
    audio_feats = torch.randn(B, T, audio_dim)
    
    model = ConditionalUNet3D_CrossAttn(n_channels=1, n_classes=1, audio_dim=audio_dim, base_channels=32, embed_dim=128, num_heads=4)
    output = model(x, audio_feats)
    print("Output shape:", output.shape)  # Expected: (B, 1, T, H, W)
