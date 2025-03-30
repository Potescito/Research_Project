"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for waveform, frames, audio_paths, video_paths in dataloader:
        waveform = waveform.to(device)
        frames = frames.to(device)
        
        optimizer.zero_grad()
        outputs = model(waveform, frames)
        loss = criterion(outputs, frames)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * waveform.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for waveform, frames, audio_paths, video_paths in dataloader:
            waveform = waveform.to(device)
            frames = frames.to(device)
            
            outputs = model(waveform, frames)
            loss = criterion(outputs, frames)
            running_loss += loss.item() * waveform.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def train_model(model, 
                train_loader, 
                val_loader, 
                num_epochs, 
                optimizer, 
                scheduler, 
                criterion, 
                device, 
                log_dir='runs/basic_net', 
                checkpoint_dir='checkpoints/basic_net'):
    writer = SummaryWriter(log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step()  # Update learning rate as per scheduler
        
        epoch_time = time.time() - start_time
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        print(f"Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"basic_net_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    
    writer.close()

if __name__ == "__main__":
    import sys
    sys.path.append('../')

    import torch.nn as nn
    import torch.optim as optim
    from train_basic import train_model
    from basic_net import BasicDenoisingNetwork
    from torch.utils.data import DataLoader
    from src.AVDataset import AVDataset
    from src.transforms import TemporalWindowTransform, ContextualSamplingTransform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 51)]
    nSubsv = [f"sub{str(i).zfill(3)}" for i in range(51, 75)]

    temporal_transform = TemporalWindowTransform(window_size_sec=1, audio_sample_rate=16000, video_fps=83)
    contextual_transform = ContextualSamplingTransform(context_size=1, audio_sample_rate=16000, video_fps=83)

    train_dataset = AVDataset(audio_root, video_root, subs=nSubst, filter_keyword=keyword, transform=temporal_transform)
    val_dataset = AVDataset(audio_root, video_root, subs=nSubsv, filter_keyword=keyword, transform=temporal_transform)
   
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=AVDataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=AVDataset.collate)
    
    model = BasicDenoisingNetwork(base_channels=32).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    num_epochs = 50
    
    train_model(model, train_loader, val_loader, num_epochs, optimizer, scheduler, criterion, device)