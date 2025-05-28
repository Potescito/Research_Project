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
import math


# ====================================================================
# Timestep Embedding
# ====================================================================
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        """
        Generates sinusoidal timestep embeddings. Diffusion models need 
        to know at which point in the noising/denoising process they are 
        operating. This is typically done by feeding a timestep embedding 
        into the model. I use a sinusoidal positional embedding.

        Args:
            dim (int): The dimensionality of the embedding.
            max_period (int): The maximum period for the sinusoidal functions.
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        """
        Args:
            timesteps (torch.Tensor): A 1D tensor of timesteps (integers).
                                      Shape: (batch_size,)

        Returns:
            torch.Tensor: Timestep embeddings. Shape: (batch_size, dim)
        """
        if timesteps.ndim != 1:
            raise ValueError("Timesteps should be a 1D tensor.")

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If dim is odd, pad with a zero
        if self.dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))

        return embedding


# ====================================================================
# Simple Audio Encoder
# ====================================================================
class SimpleAudioEncoder(nn.Module):
    def __init__(self, input_channels=1, output_embedding_dim=512):
        """
        A simple 1D CNN-based audio encoder.

        Args:
            input_channels (int): Number of input channels for audio (default 1 for mono).
            output_embedding_dim (int): Dimensionality of the output audio embedding.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_embedding_dim = output_embedding_dim

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=2, padding=1), # (B, 64, L/2)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),            # (B, 128, L/4)
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),           # (B, 256, L/8)
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),           # (B, 512, L/16)
            nn.ReLU()
        )

        # Adaptive pooling to get a fixed-size output regardless of input length variations
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # (B, 512, 1)

        self.fc = nn.Linear(512, output_embedding_dim) # (B, output_embedding_dim)

    def forward(self, audio_segment):
        """
        Args:
            audio_segment (torch.Tensor): Batch of audio segments.
                                         Shape: (batch_size, segment_length) or
                                                (batch_size, input_channels, segment_length)
        Returns:
            torch.Tensor: Audio embeddings. Shape: (batch_size, output_embedding_dim)
        """
        if audio_segment.ndim == 2: # (batch_size, segment_length)
            # Add channel dimension: (batch_size, 1, segment_length)
            audio_segment = audio_segment.unsqueeze(1)
        elif audio_segment.ndim == 3: # (batch_size, input_channels, segment_length)
            pass # Already has channel dimension
        else:
            raise ValueError(f"Expected audio_segment to have 2 or 3 dimensions, got {audio_segment.ndim}")

        if audio_segment.shape[1] != self.input_channels:
             raise ValueError(f"Expected audio_segment to have {self.input_channels} channel(s), got {audio_segment.shape[1]}")


        x = self.conv_layers(audio_segment)
        x = self.adaptive_pool(x) # (batch_size, 512, 1)
        x = x.squeeze(-1) # (batch_size, 512)
        audio_embedding = self.fc(x) # (batch_size, output_embedding_dim)

        return audio_embedding


# ====================================================================
# Residual Block with Attention (flagged)
# ====================================================================
class ResidualBlock(nn.Module):
    """
    A residual block with attention mechanism. This block consists of two 2D 
    convolutional layers, each followed by normalization and SiLU. The output 
    of the second convolutional layer is added to the input (shortcut connection).
    If `use_attention` is set to True, an attention mechanism is applied
    to the output of the block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_emb_dim (int, optional): Dimensionality of the time embedding. Default is None.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
        groups (int, optional): Number of groups for GroupNorm. Default is 8.
        use_attention (bool, optional): Flag to indicate if an attention block should be used here. Default is False.
        attention_heads (int, optional): Number of attention heads for the attention block. Default is 4.
        audio_emb_dim (int, optional): Dimensionality of the audio embedding. Default is None.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 time_emb_dim=None, 
                 dropout_rate=0.1, 
                 groups=8, 
                 use_attention=False, 
                 attention_heads=8, 
                 audio_emb_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.use_attention = use_attention # Flag to indicate if an attention block should be used here

        # First convolution path
        self.norm1 = nn.GroupNorm(groups, in_channels) # better than BatchNorm for small batch sizes
        self.act1 = nn.SiLU() # GeLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection (if time_emb_dim is provided)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        # Second convolution path
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate) # more generalization + skip connection
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) # If channels differ, use a 1x1 convolution to match dimensions

        # Optional Attention Block (placeholder for now, to be detailed for audio)
        if self.use_attention:
            # This will be replaced by a proper CrossAttention block later
            self.attention = SpatioTemporalAttentionBlock(
                out_channels, # query_dim is the channel dim of the image features
                audio_emb_dim=audio_emb_dim, # context_dim
                num_heads=attention_heads,
                dropout=dropout_rate
            )
        else:
            self.attention = nn.Identity()

    def forward(self, x, time_emb=None, audio_emb=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            time_emb (torch.Tensor, optional): Time embedding tensor of shape (B, time_emb_dim). Default is None.
            audio_emb (torch.Tensor, optional): Audio embedding tensor of shape (B, audio_emb_dim). Default is None.
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        h = x # Store for shortcut

        # First part
        res_x = self.norm1(x)
        res_x = self.act1(res_x)
        res_x = self.conv1(res_x)

        # Incorporate time embedding
        if self.time_mlp is not None and time_emb is not None:
            time_condition = self.time_mlp(time_emb)
            # Add time_condition to h. (B, C) -> (B, C, 1, 1) for broadcasting
            res_x = res_x + time_condition.unsqueeze(-1).unsqueeze(-1)

        # Second part
        res_x = self.norm2(res_x)
        res_x = self.act2(res_x)
        res_x = self.dropout(res_x)
        res_x = self.conv2(res_x)

        # Apply shortcut
        x = res_x + self.shortcut(h) # Apply residual connection

        # Apply attention (if configured)
        if self.use_attention:
            x = self.attention(x, audio_emb=audio_emb) # audio_emb will be used by a real CrossAttention
        return x


# ====================================================================
# Attention Block
# ====================================================================
class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 query_dim, # Dimensionality of the query (image features)
                 context_dim, # Dimensionality of the context (audio features)
                 num_heads=8,
                 head_dim=None, # Dimensionality of each attention head
                 dropout=0.0):
        super().__init__()
        if head_dim is None:
            if query_dim % num_heads != 0:
                raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads}) if head_dim is not specified.")
            head_dim = query_dim // num_heads
        
        self.inner_dim = head_dim * num_heads # Total dimension across all heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5 # 1/sqrt(d_k)

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim), #in case the concatenation does not match query_dim 
            nn.Dropout(dropout)
        )

    def forward(self, query, context=None, context_mask=None):
        # query:   [batch_size, query_sequence_length, query_dim]
        # context: [batch_size, context_sequence_length, context_dim]
        # context_mask: [batch_size, context_sequence_length] (optional)

        if context is None:
            context = query # If no context is provided, use the query as context (self-attention)

        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention: [B, num_heads, seq_len, head_dim]
        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        # (B, num_heads, query_len, head_dim) @ (B, num_heads, head_dim, key_len) -> (B, num_heads, query_len, key_len)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if context_mask is not None:
            # Apply mask (for padding in context)
            # Mask should be broadcastable: (B, 1, 1, key_len)
            attention_scores = attention_scores.masked_fill(context_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)

        # (B, num_heads, query_len, key_len) @ (B, num_heads, key_len, head_dim) -> (B, num_heads, query_len, head_dim)
        out = torch.matmul(attention_probs, v)

        # Concatenate heads and project out
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1) # [B, query_len, inner_dim]
        return self.to_out(out)


class SpatioTemporalAttentionBlock(nn.Module):
    def __init__(self, 
                 channels, # Input channels of the image feature map (query_dim)
                 audio_emb_dim, # Dimension of the audio embedding (context_dim)
                 num_heads=8, 
                 head_dim=None, 
                 groups=8, 
                 dropout=0.1):
        super().__init__()

        self.norm = nn.GroupNorm(groups, channels)
        
        self.cross_attention = CrossAttentionLayer(
            query_dim=channels,
            context_dim=audio_emb_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        # Optional: Add a feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(), # Or SiLU
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x, audio_emb):
        # x: image features [B, C, H, W]
        # audio_emb: audio embedding [B, D_audio_emb] or [B, NumAudioTokens, D_audio_emb_per_token]
        
        B, C, H, W = x.shape

        x_norm = self.norm(x)
        
        # Reshape x for attention: [B, C, H, W] -> [B, H*W, C]
        query = x_norm.view(B, C, H * W).permute(0, 2, 1) # [B, HW, C]

        # Prepare context. If audio_emb is [B, D_audio_emb], we need to make it [B, 1, D_audio_emb]
        # for sequence-like input to attention.
        if audio_emb.ndim == 2: # [B, D_audio_emb]
            context = audio_emb.unsqueeze(1) # [B, 1, D_audio_emb]
        elif audio_emb.ndim == 3: # [B, NumAudioTokens, D_audio_emb_per_token]
            context = audio_emb
        else:
            raise ValueError(f"audio_emb has unexpected ndim: {audio_emb.ndim}")

        # Apply cross-attention
        attn_out = self.cross_attention(query, context) # Output: [B, HW, C]
        
        # Optional: Feed-forward network
        ffn_out = self.ffn(attn_out)
        attn_out = attn_out + ffn_out # Additive skip over FFN, or just use FFN output

        # Reshape back to image format: [B, HW, C] -> [B, C, H, W]
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        return x + attn_out # residual connection


# ====================================================================
# Downsample and Upsample Blocks (factor of 2)
# ====================================================================
class Downsample(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super().__init__()
        if channels_out is None:
            channels_out = channels_in
        # Using a strided convolution for downsampling & channel changing
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super().__init__()
        if channels_out is None:
            channels_out = channels_in
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


# ==========================================================================================================
# ==========================================================================================================
# UNet Architecture
# ==========================================================================================================
# ==========================================================================================================
class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,          # MRI is grayscale
                 out_channels=1,         # Predicting noise (same channels as input)
                 model_channels=64,      # Base number of channels
                 channel_multipliers=(1, 2, 4, 8), # Channel multiplier for each resolution level
                 num_residual_blocks=2,  # Number of residual blocks per level
                 time_emb_dim=256,       # Dimensionality of timestep embedding
                 audio_emb_dim=512,      # Dimensionality of audio embedding
                 dropout_rate=0.1,
                 attention_resolutions=(2,3), # At which levels (depths) to apply attention (1-indexed)
                                              # e.g., (2,3) means at the 2nd and 3rd down/up blocks
                 groups=8,              # Number of groups for GroupNorm
                 attention_heads=8      # Number of attention heads for attention blocks
                 ):
        super().__init__()

        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        self.audio_emb_dim = audio_emb_dim
        self.channel_multipliers = channel_multipliers
        self.num_residual_blocks = num_residual_blocks

        # 1. Initial Conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        current_ch = model_channels

        # 2. Downsampling Path
        self.down_blocks_res = nn.ModuleList()
        self.down_blocks_sample = nn.ModuleList()
        
        for level_idx, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult
            for _ in range(num_residual_blocks):
                use_att = (level_idx + 1) in attention_resolutions # +1 because levels are 1-indexed in common literature (flag)
                self.down_blocks_res.append(ResidualBlock(
                    current_ch, out_ch, time_emb_dim, dropout_rate, groups,
                    use_attention=use_att, attention_heads=attention_heads, audio_emb_dim=audio_emb_dim if use_att else None
                ))
                current_ch = out_ch
            if level_idx != len(channel_multipliers) - 1: # Don't add downsample after the last multiplier stage
                self.down_blocks_sample.append(Downsample(current_ch))  # current_ch is then the output of the last resblock at this level or before downsampling

        # 3. Bottleneck
        self.bottleneck_block1 = ResidualBlock(
            current_ch, current_ch, time_emb_dim, dropout_rate, groups,
            use_attention=True, attention_heads=attention_heads, audio_emb_dim=audio_emb_dim # Always use attention in bottleneck?
        )
        self.bottleneck_block2 = ResidualBlock(
            current_ch, current_ch, time_emb_dim, dropout_rate, groups
        )

        # 4. Upsampling Path
        self.up_blocks_res = nn.ModuleList()
        self.up_blocks_sample = nn.ModuleList()
        
        for level_idx, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch_up = model_channels * mult # Resblocks at this level should output this many channel
            skip_channels = model_channels * mult # Channels from corresponding down_block res outputs

            # The input to resblocks in up-path is `current_ch` (from previous upsample) + `skip_channels`
            res_block_in_channels = current_ch + skip_channels
            
            for _ in range(num_residual_blocks):
                use_att = (level_idx + 1) in attention_resolutions
                self.up_blocks_res.append(ResidualBlock(
                    res_block_in_channels, # After concat
                    out_ch_up,
                    time_emb_dim, dropout_rate, groups,
                    use_attention=use_att, attention_heads=attention_heads, audio_emb_dim=audio_emb_dim if use_att else None
                ))
                res_block_in_channels = out_ch_up # For next resblock in *same level* if num_res_blocks > 1 (no concat feats)
                                                                        
            current_ch = out_ch_up # This is the channel count before the upsample layer (if any)
            
            if level_idx != 0: # Don't add upsample if this is the last (highest-res) level
                self.up_blocks_sample.append(Upsample(current_ch)) # Upsamples to current_ch

        # 5. Final Conv (ResNetv2 style)
        self.final_norm = nn.GroupNorm(groups, model_channels) # model_channels is the expected output from last up_block
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time_emb, audio_emb=None):
        # x: (B, C_in, H, W) - noisy image
        # time_emb: (B, time_emb_dim) - timestep embedding
        # audio_emb: (B, audio_emb_dim) - audio embedding (optional for now, for structure)

        # Initial Conv
        h = self.init_conv(x) # (B, model_channels, H, W)
        skips_for_concat = [h] # outputs of res blocks before downsp (first is the initial conv feats) drop the intermediat ones

        # Downsampling Path
        temp_res_idx = 0
        for level_idx in range(len(self.channel_multipliers)):
            for _ in range(self.num_residual_blocks):
                h = self.down_blocks_res[temp_res_idx](h, time_emb, audio_emb)
                temp_res_idx += 1
            skips_for_concat.append(h) # store features before downsampling
            if level_idx != len(self.channel_multipliers) - 1:
                h = self.down_blocks_sample[level_idx](h)

        # Bottleneck
        h = self.bottleneck_block1(h, time_emb, audio_emb)
        h = self.bottleneck_block2(h, time_emb, audio_emb)

        # Upsampling Path
        temp_res_idx = 0
        temp_sample_idx = 0
        for level_idx in reversed(range(len(self.channel_multipliers))):
            skip_h = skips_for_concat.pop() # the last skip is used

            if h.shape[2:] != skip_h.shape[2:]:
                h = self._center_crop_if_needed(h, skip_h)
                pass
            
            h = torch.cat([h, skip_h], dim=1) # channel concat

            for _ in range(self.num_residual_blocks):
                h = self.up_blocks_res[temp_res_idx](h, time_emb, audio_emb)
                temp_res_idx += 1
                # After the first resblock in the up-path for this level, the input to the next resblock
                # (if num_res_blocks > 1 for this level) will not have the skip connection re-added.
                # The `res_block_in_channels` definition in __init__ for up_blocks_res needs care.
                # A common pattern: upsample -> concat -> ResBlock -> ResBlock ...
                # This means the first ResBlock takes concatenated, subsequent ones take output of previous.
                # My `up_blocks_res` init currently assumes all resblocks take concatenated input due to `res_block_in_channels`.
                # This is a common point of complexity in U-Net implementations.
                # For now, let's assume this simplified loop works with the current ResidualBlock def.
                # A more robust way is to define input channels for each up-resblock carefully.
                # OR, the first resblock in the up-level reduces channels from (upsampled+skip) to (level_out_ch). Subsequent ones maintain (level_out_ch).

            if level_idx != 0:
                h = self.up_blocks_sample[temp_sample_idx](h) # up_blocks_sample is in reverse order of creation
                temp_sample_idx +=1

        # Final Projection
        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h) # (B, out_channels, H, W)
        return h
    
    @staticmethod
    def _center_crop_if_needed(source_tensor, target_tensor):
        """ 
        Center crops source_tensor if its spatial dims are larger than target_tensor's. 
        
        Args:
            source_tensor (torch.Tensor): Tensor to be cropped. Shape (B, C, H, W).
            target_tensor (torch.Tensor): Tensor to match dimensions with.
        """
        target_h, target_w = target_tensor.shape[2], target_tensor.shape[3]
        source_h, source_w = source_tensor.shape[2], source_tensor.shape[3]

        if source_h == target_h and source_w == target_w:
            return source_tensor

        if source_h > target_h or source_w > target_w:
            # Calculate cropping amounts
            delta_h = source_h - target_h
            delta_w = source_w - target_w

            # Ensure deltas are not negative (should not happen if source > target)
            delta_h = max(0, delta_h)
            delta_w = max(0, delta_w)

            top = delta_h // 2
            bottom = delta_h - top # Accounts for odd differences
            left = delta_w // 2
            right = delta_w - left # Accounts for odd differences
            return source_tensor[:, :, top:source_h-bottom, left:source_w-right]


# %% Debugging
if __name__ == "__main__":
    from F_diffusion import *
    import matplotlib.pyplot as plt

    # Timestep embedding test
    timestep_embedder = TimestepEmbedding(dim=256)
    timesteps_tensor = torch.randint(0, 1000, (2,)) # Batch size of 2, random timesteps
    time_embeddings = timestep_embedder(timesteps_tensor)
    print("Timestep Embeddings Shape:", time_embeddings.shape) # Expected: torch.Size([2, 256])
    plt.pcolormesh(time_embeddings.detach().numpy())
# %% Debugging
if __name__ == "__main__":
    from F_diffusion import *
    import matplotlib.pyplot as plt

    # Assume audio_segment_length = 16000 (1 second at 16kHz)
    audio_data_sample = torch.randn(2, 16000) # Batch size of 2

    # My dataloader provides [batch_size, num_segments, segment_length] do it per segment
    audio_encoder = SimpleAudioEncoder(input_channels=1, output_embedding_dim=512)
    audio_embeddings = audio_encoder(audio_data_sample)
    print("Audio Embeddings Shape:", audio_embeddings.shape) # Expected: torch.Size([2, 512])
    plt.pcolormesh(audio_embeddings.detach().numpy())
# %% Debugging Unet
if __name__ == "__main__":
    from F_diffusion import *
    import torch
    import torchinfo

    batch_size = 2
    height, width = 84, 84
    input = torch.randn(batch_size, 1, height, width)
    timemb = torch.randn(batch_size, 256) 
    audioemb = torch.randn(batch_size, 512)

    unet_model = UNet()

    # Fwd pass
    output = unet_model(input, timemb, audioemb)
    print("Output Shape:", output.shape) # Expected: (batch_size, out_channels, height, width)

    print(torchinfo.summary(unet_model, input_size=(input.shape, timemb.shape, audioemb.shape), device="cuda"))


""" Without CrossAttention (placeholder)
==========================================================================================
Total params: 44,162,625
Trainable params: 44,162,625
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 30.38
==========================================================================================
Input size (MB): 0.06
Forward/backward pass size (MB): 358.04
Params size (MB): 174.28
Estimated Total Size (MB): 532.39
==========================================================================================
"""

""" With CrossAttention
=========================================================================================================
Total params: 50,986,561
Trainable params: 50,986,561
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 29.86
=========================================================================================================
Input size (MB): 0.06
Forward/backward pass size (MB): 494.11
Params size (MB): 203.95
Estimated Total Size (MB): 698.12
=========================================================================================================
"""