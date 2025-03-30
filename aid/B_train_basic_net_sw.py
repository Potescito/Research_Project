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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.AVDataset import AVDataset
from src.transforms import SlidingWindowTransform

from B_basic_net_sw import BasicDenoisingNetworkSlidingVideo

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Change if multi-node.
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    scaler = torch.amp.GradScaler(device=device)
    for waveforms, frames, _, _ in dataloader:
        waveforms = waveforms.to(device)  # (B, num_windows, window_audio)
        frames = frames.to(device) #  (B, num_windows, window_video, 1, H, W)
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # Fwd
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(waveforms, frames) #  (B, num_windows, window_video, 1, H, W)
            loss = criterion(outputs, frames) # I know it's wrong :(
        scaler.scale(loss).backward()
        scaler.step(optimizer) #.step()
        running_loss += loss.item() * waveforms.size(0) # MSE or L1, its an avg 
        scaler.update()
    return running_loss / len(dataloader.dataset) # true avg, not avg of avgs

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for waveforms, frames, _, _ in dataloader:
            waveforms = waveforms.to(device)
            frames = frames.to(device)
            outputs = model(waveforms, frames)
            loss = criterion(outputs, frames)
            running_loss += loss.item() * waveforms.size(0)
    return running_loss / len(dataloader.dataset)

def train(rank, world_size, args): # ranks are unique ids assigned to each process 0,1..gpus-1
    setup(rank, world_size) # world_size as 4 I assume single node this is so confusing
    device = torch.device(f"cuda:{rank}")
    
    # Create dataset and transform instances.
    dataset_t = AVDataset(
        audio_root=args.audio_root,
        video_root=args.video_root,
        subs=args.subs_t,
        filter_keyword=args.filter_keyword,
        transform=None,  # No extra transform; raw data will be padded and then sliding window applied in collate.
        video_max_frames=args.video_max_frames,
        audio_sampling_rate=args.audio_sampling_rate,
        frame_skip=args.frame_skip
    )

    dataset_v = AVDataset(
        audio_root=args.audio_root,
        video_root=args.video_root,
        subs=args.subs_v,
        filter_keyword=args.filter_keyword,
        transform=None,  # No extra transform; raw data will be padded and then sliding window applied in collate.
        video_max_frames=args.video_max_frames,
        audio_sampling_rate=args.audio_sampling_rate,
        frame_skip=args.frame_skip
    )

    sw_transform = SlidingWindowTransform(args.sw_window_duration, args.sw_step_duration, args.audio_sampling_rate, args.video_fps)
    
    # Distributed sampler.
    train_sampler = DistributedSampler(dataset_t, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset_t, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    

    val_sampler = DistributedSampler(dataset_v, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(dataset_v, batch_size=args.batch_size, sampler=val_sampler,
                            collate_fn=lambda batch: AVDataset.collate(batch, sw_transform))
    
    # Initialize the network.
    model = BasicDenoisingNetworkSlidingVideo(base_channels=args.base_channels,
                                                window_audio=args.window_audio,
                                                window_video=args.window_video).to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    num_epochs = args.epochs
    try:
        for epoch in range(1, num_epochs + 1):
            if rank == 0:
                print("hola")
            train_sampler.set_epoch(epoch)
            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            scheduler.step()
            
            if rank == 0:
                print(f"Epoch {epoch}/{num_epochs} - Time: {time.time() - epoch_start:.2f}s "
                    f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
                # Save checkpoint.
                checkpoint_path = os.path.join(args.checkpoint_dir, f"basic_net_sw{epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)
    finally:
        cleanup()


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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step", type=int, default=10)
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
    # Sliding window parameters for collate.
    parser.add_argument("--sw_window_duration", type=float, default=4.0, help="Sliding window duration in seconds")
    parser.add_argument("--sw_step_duration", type=float, default=4.0, help="Sliding window step in seconds")
    parser.add_argument("--video_fps", type=int, default=83)
    # Expected window lengths (should match sw_window_duration * sampling rate/fps)
    parser.add_argument("--window_audio", type=int, default=64000)
    parser.add_argument("--window_video", type=int, default=332)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/basic_net_sw")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    world_size = 4  # Number of GPUs available.
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True) # supposedly launches a process per GPU.

if __name__ == "__main__":
    main()