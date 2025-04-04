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


def gaussian_window(window_size, sigma, channel):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2)) #1d gaussian kernel
    g = g / g.sum()
    g = g.unsqueeze(1)
    window = g @ g.t()  # outer product to get 2d
    window = window.expand(channel, 1, window_size, window_size)
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0, channel=1, size_average=True):
    """
    A simple differentiable SSIM for a pair of images.
    Args:
        img1, img2: Tensors of shape (B, channel, H, W)
    Returns:
        SSIM index (if size_average=True, averaged over batch and spatial dims)
    """
    window = gaussian_window(window_size, sigma, channel).to(img1.device)
    # meas
    mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # variances
    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq #sq img mean - local mean
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2 # covariance

    C1 = (0.01 * data_range)**2 # from wiki: two vars to stabilize the division with weak denominators
    C2 = (0.03 * data_range)**2 # L is the dynamic range of the pixel values -> k vals are by default 0.01 and 0.03

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)) # per window
    if size_average: # avg over batch and spatial dims
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class CompositeSSL(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=1.0, lambda_temporal=0.2, lambda_tv=0.01):
        """
        Composite loss for self-supervised video denoising.
        Args:
            lambda_l1 (float): Weight for L1 loss.
            lambda_ssim (float): Weight for SSIM loss.
            lambda_temporal (float): Weight for temporal consistency loss.
            lambda_tv (float): Weight for total variation loss.
        """
        super(CompositeSSL, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_temporal = lambda_temporal
        self.lambda_tv = lambda_tv
        self.l1_loss = nn.L1Loss()
    
    def forward(self, output, target):
        """
        Args:
            output, target: Tensors of shape (B, 1, wind_video, H, W)
        Returns:
            Composite loss value.
        """
        # Pixel-wise L1 loss.
        loss_l1 = self.l1_loss(output, target)
        
        # SSIM loss computed per frame.
        _, C, T, _, _ = output.shape
        ssim_loss_sum = 0.0
        for t in range(T):
            frame_out = output[:, :, t, :, :]  # shape: (B, 1, H, W)
            frame_target = target[:, :, t, :, :]
            ssim_val = ssim(frame_out, frame_target, channel=C, size_average=True)
            ssim_loss_sum += (1 - ssim_val) # measure into loss for 0
        loss_ssim = ssim_loss_sum / T
        
        # Temporal consistency -> encourages output temporal differences to match target differences.
        temporal_diff_out = output[:, :, 1:, :, :] - output[:, :, :-1, :, :] #grad/diff wise
        temporal_diff_target = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        loss_temporal = self.l1_loss(temporal_diff_out, temporal_diff_target)
        
        # TV loss for spatial smoothness maybe? low weight
        tv_h = torch.mean(torch.abs(output[:, :, :, 1:, :] - output[:, :, :, :-1, :]))
        tv_w = torch.mean(torch.abs(output[:, :, :, :, 1:] - output[:, :, :, :, :-1]))
        loss_tv = tv_h + tv_w
        
        total_loss = (self.lambda_l1 * loss_l1 +
                      self.lambda_ssim * loss_ssim +
                      self.lambda_temporal * loss_temporal +
                      self.lambda_tv * loss_tv)
        return total_loss

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.9e-3) # not 1e-3
    parser.add_argument("--lr_step", type=int, default=5) # after every 10 epochs the lr is updated by gamma * lr
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cond_unet_sw_loss")
    parser.add_argument("--log_dir", type=str, default="runs/cond_unet_sw_loss")
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

    criterion = CompositeSSL(lambda_l1=1.0, lambda_ssim=1.0, lambda_temporal=0.2, lambda_tv=0.01).to(device)
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