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
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.AVDataset import AVDataset 
from src.transforms import SlidingWindowTransform

from F_diffusion import UNet, SimpleAudioEncoder, TimestepEmbedding
from F_schedulers import get_diffusion_parameters, extract, cosine_beta_schedule


def train_diffusion_model(
    num_epochs,
    dataloader, 
    unet_model, 
    audio_encoder, 
    timestep_embedder, 
    diffusion_params, # dictionary from get_diffusion_parameters
    optimizer,
    device,
    total_timesteps_T, # T value used for get_diffusion_parameters (e.g., 1000)
    gradient_accumulation_steps=1, # Optional: for accumulating gradients
    writer = None,
    checkpoint_dir = "checkpoints/"
):
    """
    Training loop for the audio-visual diffusion model.
    """
    unet_model.train()
    audio_encoder.train()
    timestep_embedder.train() # in case I decide to use learnable timestep embeddings
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        num_samples_processed = 0 # To average loss correctly if batch sizes vary or steps are skipped
        
        for batch_idx, batch_data in enumerate(dataloader):
            video_segments = batch_data[1].to(device)
            audio_segments = batch_data[0].to(device)

            B, _, frames_per_segment, _, _, _  = video_segments.shape

            # Segment count might differ btwn video and audio
            min_num_synced_segments = min(video_segments.shape[1], audio_segments.shape[1])

            # --- Data Selection: One random frame per batch item, from a random synchronized segment ---
            
            # 1. Select a random synchronized segment index for each item in the batch
            #    These indices will be used for both video (up to min_num_synced_segments)
            #    and audio (up to min_num_synced_segments).
            rand_seg_indices = torch.randint(0, min_num_synced_segments, (B,), device=device)

            # 2. Select a random frame index from the chosen video segment for each item
            rand_frame_indices = torch.randint(0, frames_per_segment, (B,), device=device)

            # 3. Gather the selected x0 frames (target clean images for diffusion)
            #    x0_batch should be [B, C, H, W]
            batch_indices = torch.arange(B, device=device) # Helper for advanced indexing
            x0_batch = video_segments[batch_indices, rand_seg_indices, rand_frame_indices]
            # x0_batch now has shape [B, C, H, W]

            # 4. Gather the corresponding full audio segments
            #    audio_segment_for_x0 should be [B, AudioSegmentLength]
            audio_segment_for_x0 = audio_segments[batch_indices, rand_seg_indices]
            # audio_segment_for_x0 now has shape [B, AudioSegmentLength]

            # --- Audio Embedding ---
            # The audio embedding corresponds to the entire audio segment for the chosen frame's segment
            audio_emb_batch = audio_encoder(audio_segment_for_x0) # Expected shape: [B, D_audio_emb]

            # --- Forward Diffusion Process ---
            # 1. Sample random timesteps 't' for each item in the batch (from 0 to T-1)
            t_batch = torch.randint(0, total_timesteps_T, (B,), device=device).long()

            # 2. Get pre-computed schedule parameters for these timesteps
            sqrt_alphas_cumprod_t = extract(
                diffusion_params['sqrt_alphas_cumprod'], t_batch, x0_batch.shape
            )
            sqrt_one_minus_alphas_cumprod_t = extract(
                diffusion_params['sqrt_one_minus_alphas_cumprod'], t_batch, x0_batch.shape
            )

            # 3. Sample noise epsilon ~ N(0, I)
            epsilon_batch = torch.randn_like(x0_batch)

            # 4. Create the noisy image x_t for each x0
            #    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
            xt_batch = sqrt_alphas_cumprod_t * x0_batch + sqrt_one_minus_alphas_cumprod_t * epsilon_batch

            # --- Timestep Embedding ---
            time_emb_batch = timestep_embedder(t_batch) # Expected shape: [B, D_time_emb]

            # --- U-Net Prediction ---
            # The U-Net predicts the noise added to x_t, conditioned on t and audio
            predicted_noise_batch = unet_model(xt_batch, time_emb_batch, audio_emb_batch)

            # --- Loss Calculation ---
            loss = F.mse_loss(predicted_noise_batch, epsilon_batch) #l2
            
            # Normalize loss for gradient accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()

            # --- Optimization Step ---
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * (gradient_accumulation_steps if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader) else 0) # Re-scale accumulated loss for logging
            num_samples_processed += B # Count actual samples processed for loss averaging

            if (batch_idx + 1) % 50 == 0:
                current_avg_loss = total_loss / (batch_idx +1) # Rough average loss so far in epoch
                print(f"--Epoch [{epoch}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Current Avg Loss: {current_avg_loss:.4f}, Last Mini-Batch Loss: {loss.item():.4f}")
        
        # avg_epoch_loss = total_loss / len(dataloader) # If not using grad accum or careful batch counting
        avg_epoch_loss = total_loss / ( (len(dataloader) + gradient_accumulation_steps -1 ) // gradient_accumulation_steps ) # Avg loss over effective optimization steps
        print(f"Epoch [{epoch}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}. [Time: {time.time() - epoch_start:.2f}s]")
        if writer is not None:
            writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"ckp_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'unet_model_state_dict': unet_model.state_dict(),
                'audio_encoder_state_dict': audio_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)

    print("Training finished.")


def main():
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 51)]
    nSubsv = [f"sub{str(i).zfill(3)}" for i in range(51, 75)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--time_steps", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--lr", type=float, default=0.5e-4)
    parser.add_argument("--audio_root", type=str, default=audio_root)
    parser.add_argument("--video_root", type=str, default=video_root)
    parser.add_argument("--subs_t", type=list, default=nSubst)
    parser.add_argument("--subs_v", type=list, default=nSubsv)
    parser.add_argument("--filter_keyword", type=str, default=keyword)
    parser.add_argument("--video_max_frames", type=int, default=None)
    parser.add_argument("--audio_sampling_rate", type=int, default=16000)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--sw_window_duration", type=float, default=1, help="Sliding window duration in seconds")
    parser.add_argument("--sw_step_duration", type=float, default=1, help="Sliding window step in seconds")
    parser.add_argument("--video_fps", type=int, default=83)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/F_diffusionatt")
    parser.add_argument("--log_dir", type=str, default="runs/F_diffusionatt")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ______________________________________________________________________________________
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # ______________________________________________________________________________________
    sw_transform = SlidingWindowTransform(args.sw_window_duration, args.sw_step_duration,
                                          args.audio_sampling_rate, args.video_fps)
    
    dataset_t = AVDataset(
        audio_root=args.audio_root,
        video_root=args.video_root,
        subs=args.subs_t,
        filter_keyword=args.filter_keyword,
        transform=None,  # No extra transform raw data will be padded and then sliding window applied in collate
        video_max_frames=args.video_max_frames,
        audio_sampling_rate=args.audio_sampling_rate,
        frame_skip=args.frame_skip
    )

    # dataset_v = AVDataset(
    #     audio_root=args.audio_root,
    #     video_root=args.video_root,
    #     subs=args.subs_v,
    #     filter_keyword=args.filter_keyword,
    #     transform=None,  # No extra transform raw data will be padded and then sliding window applied in collate
    #     video_max_frames=args.video_max_frames,
    #     audio_sampling_rate=args.audio_sampling_rate,
    #     frame_skip=args.frame_skip
    # )

    #_______________________________________________________________________________________
    train_loader = DataLoader(dataset_t, batch_size=args.batch_size,
                              collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    # val_loader = DataLoader(dataset_v, batch_size=args.batch_size,
    #                         collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    print("Dataset loaded and collated.")

    #______________________________________________________________________________________
    audio_enc = SimpleAudioEncoder(output_embedding_dim=512).to(device) 
    time_emb = TimestepEmbedding(dim=256).to(device)
    #_____________________________________________________________________________________
    unet = UNet(
        in_channels=1, 
        out_channels=1, 
        model_channels=64, 
        channel_multipliers=(1, 2, 4, 8), # Or your preferred config
        num_residual_blocks=2,
        time_emb_dim=256, # Must match TimestepEmbedding output
        audio_emb_dim=512, # Must match SimpleAudioEncoder output
        attention_resolutions=(2,3) 
    ).to(device)
    
    #____________________________________________________________________________________
    optimizer = optim.AdamW(
        list(unet.parameters()) + list(audio_enc.parameters()), # Add other model params if they have learnable weights
        lr=args.lr
    )
    #_____________________________________________________________________________________
    betas = cosine_beta_schedule(timesteps=args.time_steps)
    diffusion_params_dict = get_diffusion_parameters(betas=betas, device=device)
    
    #_____________________________________________________________________________________
    num_epochs = args.epochs
    print("Training Diffusion Model...")
    
    train_diffusion_model(
        num_epochs=num_epochs,
        dataloader=train_loader, 
        unet_model=unet,
        audio_encoder=audio_enc,
        timestep_embedder=time_emb,
        diffusion_params=diffusion_params_dict,
        optimizer=optimizer,
        device=device,
        total_timesteps_T=args.time_steps,
        gradient_accumulation_steps=1,  # Adjust 
        writer=writer,
        checkpoint_dir=args.checkpoint_dir
    )
    writer.close()


if __name__ == "__main__":
    main()
