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
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.AVDataset import AVDataset
from src.transforms import SlidingWindowTransform
from C_wav2vec2_AP import AudioFeatureExtractorFiLM
from C_cond_Unet import ConditionalUNet3D_FiLM

def train_one_epoch(model, audio_extractor, dataloader, criterion, optimizer, device):
    model.train()
    # Typically, freeze the audio extractor (or set to eval) if using a pretrained model.
    audio_extractor.eval()  
    running_loss = 0.0

    for waveforms, video_windows, _, _ in dataloader:
        # waveforms: (B, num_windows, window_audio)
        # video_windows: (B, num_windows, window_video, 1, H, W)
        optimizer.zero_grad() # iterative as suggested by pytorch maybe?
        batch_loss = 0.0

        num_windows = min(waveforms.shape[1], video_windows.shape[1])

        for i in range(num_windows):
            wave_i = waveforms[:, i, :]    # (B, window_audio)
            vid_i = video_windows[:, i, ...]  # shape: (B, window_video, 1, H, W)

            # Move to device but maybe I can save one move here !
            wave_i = wave_i.to(device)
            vid_i = vid_i.to(device)
            
            # Audio  feat / condition
            audio_feats = audio_extractor(wave_i)  # (B, window_video, feature_dim)
            cond = audio_feats.mean(dim=1) # (B, cond_dim).
            
            # Prepare video input for the UNet --> expects video input shape (B, 1, T, H, W)
            vid_input = vid_i.permute(0, 2, 1, 3, 4)  # becomes B, 1, window_video, H, W
            
            # Fwd -> autocast for mixed precision (deprecated??)
            with torch.cuda.amp.autocast():
                output = model(vid_input, cond)  # Expected output shape: (B, 1, window_video, H, W)
                loss = criterion(output, vid_input)
            batch_loss += loss
            loss.backward()

        # Avg loss over windows
        batch_loss /= num_windows
        optimizer.step()
        running_loss += batch_loss.item() * waveforms.shape[0] # when using MSE or L1, it's an avg

    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, audio_extractor, dataloader, criterion, device):
    model.eval()
    audio_extractor.eval()
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for waveforms, video_windows, _, _ in dataloader:
            batch_loss = 0.0
            num_windows = min(waveforms.shape[1], video_windows.shape[1])
            for i in range(num_windows):
                wave_i = waveforms[:, i, :]
                vid_i = video_windows[:, i, ...]
                wave_i = wave_i.to(device)
                vid_i = vid_i.to(device)
                audio_feats = audio_extractor(wave_i)
                cond = audio_feats.mean(dim=1)
                vid_input = vid_i.permute(0, 2, 1, 3, 4)
                output = model(vid_input, cond)
                loss = criterion(output, vid_input)
                batch_loss += loss
            batch_loss /= num_windows
            running_loss += batch_loss.item() * waveforms.shape[0]
            total_samples += waveforms.shape[0]
    return running_loss / total_samples # len(dataloader.dataset)


# ====================================================================
def main():

    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 51)]
    nSubsv = [f"sub{str(i).zfill(3)}" for i in range(51, 75)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.8e-3) # not 1e-3
    parser.add_argument("--lr_step", type=int, default=10) # after every 10 epochs the lr is updated by gamma * lr
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--audio_root", type=str, default=audio_root)
    parser.add_argument("--video_root", type=str, default=video_root)
    parser.add_argument("--subs_t", type=list, default=nSubst)
    parser.add_argument("--subs_v", type=list, default=nSubsv)
    parser.add_argument("--filter_keyword", type=str, default=keyword)
    parser.add_argument("--video_max_frames", type=int, default=None)
    parser.add_argument("--audio_sampling_rate", type=int, default=16000)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--sw_window_duration", type=float, default=4.0, help="Sliding window duration in seconds")
    parser.add_argument("--sw_step_duration", type=float, default=4.0, help="Sliding window step in seconds")
    parser.add_argument("--video_fps", type=int, default=83)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cond_unet_sw")
    parser.add_argument("--log_dir", type=str, default="runs/cond_unet_sw")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ______________________________________________________________________________________
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    window_audio = int(args.sw_window_duration * args.audio_sampling_rate)
    window_video = int(args.sw_window_duration * args.video_fps)

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

    dataset_v = AVDataset(
        audio_root=args.audio_root,
        video_root=args.video_root,
        subs=args.subs_v,
        filter_keyword=args.filter_keyword,
        transform=None,  # No extra transform raw data will be padded and then sliding window applied in collate
        video_max_frames=args.video_max_frames,
        audio_sampling_rate=args.audio_sampling_rate,
        frame_skip=args.frame_skip
    )

    #_______________________________________________________________________________________
    train_loader = DataLoader(dataset_t, batch_size=args.batch_size,
                              collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    val_loader = DataLoader(dataset_v, batch_size=args.batch_size,
                            collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    print("hola")
    #______________________________________________________________________________________
    audio_extractor = AudioFeatureExtractorFiLM(window_video=window_video, 
                                                pretrained_model_name="facebook/wav2vec2-base-960h"
                                                ).to(device) # cond_dim is set to the wav2vec2 hidden size, typically 768.
    print("hola2")
    #______________________________________________________________________________________
    cond_dim = audio_extractor.feature_dim
    model = ConditionalUNet3D_FiLM(cond_dim=cond_dim, base_channels=args.base_channels).to(device)

    criterion = nn.L1Loss() # simple
    optimizer = optim.Adam(list(model.parameters()) + list(audio_extractor.parameters()), lr=args.lr) # audio has a linear layer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) # after maybe 10

    # ______________________________________________________________________________________
    num_epochs = args.epochs
    best_val_loss = float('inf')
    print("Training Conditional U-Net with FiLM conditioning...")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, audio_extractor, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, audio_extractor, val_loader, criterion, device)
        scheduler.step()
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        print(f"Epoch {epoch}/{num_epochs} - Time: {time.time()-epoch_start:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"cond_unet_film_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    writer.close()

if __name__ == "__main__":
    main()