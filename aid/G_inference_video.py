"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import sys
sys.path.append('../') # Or adjust as needed to find your modules

import os
if sys.platform.startswith('win'):
    print("Running on Windows. Setting OpenCV backend priorities for MSMF and GStreamer.")
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "100"
    os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
else:
    print(f"Running on {sys.platform}. Skipping MSMF-specific OpenCV backend priority settings.")

import cv2
import torch
import torchaudio
import argparse

from src.AVDataset import AVDataset
from src.transforms import SlidingWindowTransform
from F_diffusion import UNet, TimestepEmbedding, SimpleAudioEncoder
from F_encoders import PretrainedAudioEncoder
from F_schedulers import get_diffusion_parameters, cosine_beta_schedule
from F_inference import sample_ddpm
import src.utils as ut 

@torch.no_grad()
def denoise_full_video(
    full_audio_waveform, # A single complete audio waveform tensor [NumSamples]
    full_video_frames,   # A single complete video tensor [NumFrames, C, H, W]
    unet_model,
    audio_encoder,
    timestep_embedder,
    diffusion_params,
    sw_transform_overlap, # A SlidingWindowTransform instance configured for OVERLAP
    device,
    total_timesteps_T,
    inference_batch_size=16  # To process chunks in batches if there are many
):
    """
    Denoises a full video sequence using an overlapping window approach.
    """
    print("1. Creating overlapping chunks from the full video and audio...")
    full_audio_waveform = full_audio_waveform.unsqueeze(0)
    full_video_frames = full_video_frames.unsqueeze(0)

    # Use the transform to get overlapping chunks
    audio_chunks, video_chunks = sw_transform_overlap(full_audio_waveform, full_video_frames)
    # video_chunks shape: [1, NumChunks, N_frames, C, H, W]
    # audio_chunks shape: [1, NumChunks, AudioChunkLen]
    
    # Remove the temporary batch dimension
    video_chunks = video_chunks.squeeze(0) # [NumChunks, N_frames, C, H, W]
    audio_chunks = audio_chunks.squeeze(0) # [NumChunks, AudioChunkLen]
    
    num_chunks = video_chunks.shape[0]
    print(f"   Created {num_chunks} overlapping chunks.")

    # Get shapes for the U-Net and sampler
    N_frames_seq, C, H, W = video_chunks.shape[1:]
    unet_input_shape = (N_frames_seq * C, H, W)

    denoised_chunks = []
    print(f"2. Denoising chunks in mini-batches of size {inference_batch_size}...")
    for i in range(0, num_chunks, inference_batch_size):
        # Create a mini-batch of chunks
        video_chunk_batch = video_chunks[i:i+inference_batch_size].to(device)
        audio_chunk_batch = audio_chunks[i:i+inference_batch_size].to(device)
        
        num_in_batch = video_chunk_batch.shape[0]
        print(f"   Processing chunks {i} to {i+num_in_batch-1}...")
        
        # Reshape video for U-Net: [B_inf, N, C, H, W] -> [B_inf, N*C, H, W]
        unet_input_video = video_chunk_batch.reshape(num_in_batch, N_frames_seq * C, H, W)
        
        # The sample_ddpm function from F_inference.py is fine, we just need to pass the correct shape
        denoised_stacked_batch = sample_ddpm(
            unet_model, audio_encoder, timestep_embedder, diffusion_params,
            num_images=num_in_batch,
            image_shape=unet_input_shape,
            audio_segment_batch=audio_chunk_batch,
            total_timesteps_T=total_timesteps_T,
            device=device
        ) # Output: [B_inf, N*C, H, W]
        
        # Reshape back to sequence: [B_inf, N*C, H, W] -> [B_inf, N, C, H, W]
        denoised_sequence_batch = denoised_stacked_batch.view(num_in_batch, N_frames_seq, C, H, W)
        denoised_chunks.append(denoised_sequence_batch)

    # Concatenate all denoised chunks from mini-batches
    all_denoised_chunks = torch.cat(denoised_chunks, dim=0) # [NumChunks, N, C, H, W]
    # Add the batch dimension back for the overlap-add function
    all_denoised_chunks = all_denoised_chunks.unsqueeze(0) # [1, NumChunks, N, C, H, W]

    print("3. Reconstructing full video using overlap-add...")
    reconstructed_video = sw_transform_overlap.overlap_add(all_denoised_chunks, window_func='rect')
    # reconstructed_video shape: [1, FinalLength, C, H, W]
    
    print("Reconstruction complete.")
    return reconstructed_video.squeeze(0) # Return single reconstructed video [FinalLength, C, H, W]


if __name__ == '__main__':
    # --- Setup Arguments and Device ---
    parser = argparse.ArgumentParser(description="Full Video Denoising Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input .mp4 video file.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the corresponding .wav audio file.")
    parser.add_argument("--output_path", type=str, default="denoised_video.mp4", help="Path to save the denoised video.")
    parser.add_argument("--n_frames_seq", type=int, default=5, help="Sequence length (N) the model was trained on.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Add any other necessary args from your training script (e.g., audio encoder type)
    args = parser.parse_args()
    
    DEVICE = torch.device(args.device)
    N_FRAMES_SEQ = args.n_frames_seq
    C_ORIGINAL = 1 # Grayscale
    TOTAL_TIMESTEPS = 1000 # Should match training

    # --- Load Models (Choose your best configuration) ---
    print("Loading models...")
    audio_enc = SimpleAudioEncoder(output_embedding_dim=512).to(DEVICE)
    # OR:
    # audio_enc = PretrainedAudioEncoder(model="WavLM", process=True).to(DEVICE)
    
    time_emb = TimestepEmbedding(dim=256).to(DEVICE)
    
    unet = UNet(
        in_channels=N_FRAMES_SEQ * C_ORIGINAL,
        out_channels=N_FRAMES_SEQ * C_ORIGINAL,
        # Ensure other UNet params match your trained model
        audio_emb_dim=audio_enc.get_output_dim(),
    ).to(DEVICE)

    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    unet.load_state_dict(checkpoint['unet_model_state_dict'])
    audio_enc.load_state_dict(checkpoint['audio_encoder_state_dict'])
    print(f"Models loaded from checkpoint epoch {checkpoint['epoch']}.")

    # --- Prepare Diffusion Parameters ---
    betas = cosine_beta_schedule(timesteps=TOTAL_TIMESTEPS)
    diffusion_params_dict = get_diffusion_parameters(betas=betas, device=DEVICE)

    # --- Prepare Sliding Window Transform for OVERLAPPING chunks ---
    VIDEO_FPS = 83
    AUDIO_SAMPLING_RATE = 16000
    WINDOW_DURATION = N_FRAMES_SEQ / VIDEO_FPS
    STEP_DURATION = 2 / VIDEO_FPS # e.g., step of 2 frames, creating overlap
    sw_transform_overlap = SlidingWindowTransform(
        window_duration=WINDOW_DURATION,
        step_duration=STEP_DURATION,
        audio_sample_rate=AUDIO_SAMPLING_RATE,
        video_fps=VIDEO_FPS
    )

    # --- Load Data (using AVDataset's loading logic as a template) ---
    # This part can be simplified since we are loading single files, not using the full Dataset class
    print(f"Loading data from {args.video_path} and {args.audio_path}")
    waveform, sr = torchaudio.load(args.audio_path)
    if sr != AUDIO_SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_SAMPLING_RATE)
        waveform = resampler(waveform)
    if waveform.size(0) > 1: 
        waveform = torch.mean(waveform, dim=0) # convert to mono if stereo
    waveform = waveform.squeeze()

    cap = cv2.VideoCapture(args.video_path)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_list.append(torch.from_numpy(frame).float() / 255.0)
    cap.release()
    video_frames_tensor = torch.stack(frames_list).unsqueeze(1) # [NumFrames, C=1, H, W]

    # --- Run Denoising ---
    denoised_video_tensor = denoise_full_video(
        full_audio_waveform=waveform,
        full_video_frames=video_frames_tensor,
        unet_model=unet,
        audio_encoder=audio_enc,
        timestep_embedder=time_emb,
        diffusion_params=diffusion_params_dict,
        sw_transform_overlap=sw_transform_overlap,
        device=DEVICE,
        total_timesteps_T=TOTAL_TIMESTEPS,
    ) # Output shape: [FinalLength, C, H, W]

    # --- Save Output Video ---
    print(f"Saving denoised video to {args.output_path}...")
    ut.save_videos(denoised_video_tensor.unsqueeze(), args.output_path, fps=VIDEO_FPS)
    print("Done!")