"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import sys
sys.path.append('../')

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.AVDataset import AVDataset 
from src.transforms import SlidingWindowTransform

from F_diffusion import UNet, SimpleAudioEncoder, TimestepEmbedding
from F_schedulers import get_diffusion_parameters, extract, cosine_beta_schedule

@torch.no_grad()
def sample_ddpm(
    unet_model,
    audio_encoder,
    timestep_embedder,
    diffusion_params,
    num_images, # How many images/sequences to generate (batch size for sampling)
    image_shape, # Tuple: (C, H, W) for a single image
    audio_segment_batch, # Batch of audio segments to condition on, shape: [num_images, AudioSegmentLength]
    total_timesteps_T,
    device
):
    unet_model.eval()
    audio_encoder.eval()
    timestep_embedder.eval()

    C, H, W = image_shape
    # Start with random noise x_T
    xt = torch.randn((num_images, C, H, W), device=device)

    # Get audio embeddings for the batch
    audio_emb_batch = audio_encoder(audio_segment_batch.to(device))

    print(f"Starting DDPM sampling for {total_timesteps_T} timesteps...")
    for t_val in reversed(range(total_timesteps_T)): # Loop from T-1 down to 0
        # Current timestep tensor for the batch
        timesteps_batch = torch.full((num_images,), t_val, device=device, dtype=torch.long)

        # Get timestep embeddings
        time_emb_batch = timestep_embedder(timesteps_batch)

        # Predict noise using the U-Net
        predicted_noise = unet_model(xt, time_emb_batch, audio_emb_batch)

        # Get pre-computed parameters for this timestep t_val
        # Note: diffusion_params are indexed from 0 to T-1
        alphas_cumprod_t = extract(diffusion_params['alphas_cumprod'], timesteps_batch, xt.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            diffusion_params['sqrt_one_minus_alphas_cumprod'], timesteps_batch, xt.shape
        )
        
        # Calculate x0_hat (model's prediction of the "clean" image)
        # x0_hat = (x_t - sqrt(1-alpha_bar_t) * pred_epsilon) / sqrt(alpha_bar_t)
        x0_hat = (xt - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / torch.sqrt(alphas_cumprod_t)
        # It's good practice to clamp x0_hat to the expected data range (e.g., [-1, 1] if you normalized your inputs)
        # For now, we'll proceed without explicit clamping here, but consider it if your inputs were normalized.
        x0_hat = torch.clamp(x0_hat, 0.0, 1.0) # Example if data was in [-1, 1]

        if t_val == 0:
            # For the last step (t=0), often the mean is taken directly, and no more noise is added.
            # The x0_hat is the result.
            xt = x0_hat
        else:
            # Get parameters for q(x_{t-1} | x_t, x0_hat)
            posterior_variance_t = extract(
                diffusion_params['posterior_variance'], timesteps_batch, xt.shape
            )
            posterior_mean_coef1_t = extract(
                diffusion_params['posterior_mean_coef1'], timesteps_batch, xt.shape
            )
            posterior_mean_coef2_t = extract(
                diffusion_params['posterior_mean_coef2'], timesteps_batch, xt.shape
            )

            # Calculate posterior mean
            posterior_mean = posterior_mean_coef1_t * x0_hat + posterior_mean_coef2_t * xt
            
            # Sample x_{t-1}
            noise = torch.randn_like(xt)
            xt = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        
        if (t_val) % 100 == 0 or t_val == total_timesteps_T -1 :
            print(f"  Sampling step {t_val}/{total_timesteps_T}")
            _, ax = plt.subplots(1, num_images, figsize=(15, 5))
            for i in range(num_images):
                img_to_show = xt[i].squeeze().cpu().numpy() 
                ax[i].imshow(img_to_show, cmap='gray')
                ax[i].set_title(f"Generated Image {i+1}")
                ax[i].axis('off')
            plt.show()

    print("Sampling complete.")
    # The final xt (which is x0 after the loop) is the generated image batch
    # If your original MRI data was in a certain range (e.g. 0 to 1, or 0 to 255),
    # you might want to unnormalize xt here.
    return xt


def main():
    # --- Assume models, diffusion_params are loaded and on DEVICE ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOTAL_TIMESTEPS_T = 1000
    audio_encoder = SimpleAudioEncoder(output_embedding_dim=512).to(DEVICE) 
    timestep_embedder = TimestepEmbedding(dim=256).to(DEVICE)
    unet_model = UNet(
        in_channels=1, 
        out_channels=1, 
        model_channels=64, 
        channel_multipliers=(1, 2, 4, 8), # Or your preferred config
        num_residual_blocks=2,
        time_emb_dim=256, # Must match TimestepEmbedding output
        audio_emb_dim=512, # Must match SimpleAudioEncoder output
        attention_resolutions=(2,3) 
    ).to(DEVICE)
    betas = cosine_beta_schedule(timesteps=TOTAL_TIMESTEPS_T)
    diffusion_params_dict = get_diffusion_parameters(betas=betas, device=DEVICE)

    checkpoint = torch.load("checkpoints/F_diffusionv1/checkpoint_epoch_50.pth", map_location=DEVICE) # Or your best checkpoint
    unet_model.load_state_dict(checkpoint['unet_model_state_dict'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
    print(f"Models loaded from epoch {checkpoint['epoch']} with loss {checkpoint.get('loss', 'N/A')}")

    num_samples_to_generate = 4
    example_audio_segment_length = 16000 
    dummy_audio_for_sampling = torch.randn(num_samples_to_generate, example_audio_segment_length).to(DEVICE) # real should be the audio

    generated_images = sample_ddpm(
        unet_model=unet_model,
        audio_encoder=audio_encoder,
        timestep_embedder=timestep_embedder,
        diffusion_params=diffusion_params_dict,
        num_images=num_samples_to_generate,
        image_shape=(1, 256, 256),
        audio_segment_batch=dummy_audio_for_sampling,
        total_timesteps_T=TOTAL_TIMESTEPS_T,
        device=DEVICE
    )

    print(f"Generated images shape: {generated_images.shape}")

    _, ax = plt.subplots(1, num_samples_to_generate, figsize=(15, 5))
    for i in range(num_samples_to_generate):
        img_to_show = generated_images[i].squeeze().cpu().numpy() # Remove channel dim, move to CPU
        ax[i].imshow(img_to_show, cmap='gray')
        ax[i].set_title(f"Generated Image {i+1}")
        ax[i].axis('off')
    plt.show()

# %%
if __name__ == "__main__":
    from F_inference import main
    main()