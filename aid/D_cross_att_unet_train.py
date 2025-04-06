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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.AVDataset import AVDataset 
from src.transforms import SlidingWindowTransform
from C_wav2vec2_AP import AudioFeatureExtractorFiLM
from D_cross_att_unet import ConditionalUNet3D_CrossAttn
from C_train_cond_Unet_Loss import CompositeSSL

def train_one_epoch(model, audio_extractor, dataloader, criterion, optimizer, device):
    model.train()
    audio_extractor.eval()
    scaler = torch.amp.GradScaler(device=device)
    running_loss = 0.0

    for waveforms, video_windows, _, _ in dataloader:
        # waveforms: (B, num_windows, window_audio)
        # video_windows: (B, num_windows, window_video, 1, H, W)
        optimizer.zero_grad()
        batch_loss = 0.0

        num_windows = min(waveforms.shape[1], video_windows.shape[1])

        for i in range(num_windows):
            wave_i = waveforms[:, i, :]  # (B, window_audio)
            vid_i = video_windows[:, i, ...]  # (B, window_video, 1, H, W)
            
            # Move to device but maybe I can save one move here !
            wave_i = wave_i.to(device)
            vid_i = vid_i.to(device)
            
            # Audio feats -> avg is inside the net
            audio_feats = audio_extractor(wave_i)  # this is the condition -> we want the full sequence not averaged
            
            # Vid feats -> expects shape (B, 1, window_video, H, W)
            vid_input = vid_i.permute(0, 2, 1, 3, 4)  #(B, 1, window_video, H, W)
            
            # fwd pass with mixed precision.
            with torch.cuda.amp.autocast():
                output = model(vid_input, audio_feats)  # Expected: (B, 1, window_video, H, W)
                loss_i = criterion(output, vid_input)
            
            batch_loss += loss_i
            scaler.scale(loss_i).backward()
        
        # Average loss over windows.
        batch_loss /= num_windows
        scaler.step(optimizer)
        scaler.update()
        running_loss += batch_loss.item() * waveforms.shape[0] # batch

    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, audio_extractor, dataloader, criterion, device):
    model.eval()
    audio_extractor.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for waveforms, video_windows, _, _ in dataloader:
            B = waveforms.shape[0]
            batch_loss = 0.0

            num_windows = min(waveforms.shape[1], video_windows.shape[1])

            for i in range(num_windows):
                wave_i = waveforms[:, i, :]
                vid_i = video_windows[:, i, ...]
                wave_i = wave_i.to(device)
                vid_i = vid_i.to(device)
                audio_feats = audio_extractor(wave_i) # condition
                vid_input = vid_i.permute(0, 2, 1, 3, 4)
                output = model(vid_input, audio_feats)
                loss_i = criterion(output, vid_input)
                batch_loss += loss_i
            batch_loss /= num_windows
            running_loss += batch_loss.item() * B
            total_samples += B
    return running_loss / total_samples

def main():
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 51)]
    nSubsv = [f"sub{str(i).zfill(3)}" for i in range(51, 75)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.5e-3) # not 1e-3
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
    parser.add_argument("--sw_window_duration", type=float, default=2.0, help="Sliding window duration in seconds")
    parser.add_argument("--sw_step_duration", type=float, default=2.0, help="Sliding window step in seconds")
    parser.add_argument("--video_fps", type=int, default=83)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cross_att")
    parser.add_argument("--log_dir", type=str, default="runs/cross_att")
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
    print("Dataset loaded and collated.")

    #______________________________________________________________________________________
    audio_extractor = AudioFeatureExtractorFiLM(window_video=window_video,
                                                 pretrained_model_name="facebook/wav2vec2-base-960h",
                                                 freeze_wav2vec=True).to(device)
    print("Audio feature extractor loaded.")
    
    #_____________________________________________________________________________________
    cond_dim = audio_extractor.feature_dim
    model = ConditionalUNet3D_CrossAttn(n_channels=1, n_classes=1, audio_dim=cond_dim,
                                        base_channels=args.base_channels,
                                        embed_dim=128, num_heads=4).to(device)
    
    #____________________________________________________________________________________
    criterion = CompositeSSL(lambda_l1=1.0, lambda_ssim=0.5, lambda_temporal=0.05, lambda_tv=0.001).to(device)
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(list(model.parameters()) + list(audio_extractor.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    num_epochs = args.epochs
    best_val_loss = float('inf')
    print("Training Conditional U-Net with Cross-Attention Fusion...")
    
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
            checkpoint_path = os.path.join(args.checkpoint_dir, f"cross_att{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    writer.close()

if __name__ == "__main__":
    main()
