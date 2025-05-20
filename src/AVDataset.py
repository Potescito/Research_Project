"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import os
import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

class AVDataset(Dataset):
    def __init__(self, 
                 audio_root, 
                 video_root, 
                 filter_keyword=None, 
                 subs=None,
                 transform=None,
                 video_max_frames=None,
                 audio_sampling_rate=16000, 
                 frame_skip=1):
        """
        Args:
            audio_root (str): Path to the audio root directory (e.g., ../data/audios_denoised_16khz).
            video_root (str): Path to the video root directory (e.g., ../data/dataset_2drt_video_only).
            subs (list, optional): List of subjects to extract the frames. Defaults to None (all).
            transform (callable, optional): Optional transform to be applied on a sample -> TemporalWindowTransform or ContextualSamplingTransform 
            filter_keyword (str, optional): If provided, only load files with this keyword in the filename.
            video_max_frames (int, optional): Maximum number of frames to load per video.
            audio_sampling_rate (int): Target sampling rate for audio.
            frame_skip (int): Number of frames to skip when loading video frames.
        """
        self.audio_root = audio_root
        self.video_root = video_root
        self.subs       = subs
        self.filter_keyword = filter_keyword
        self.video_max_frames = video_max_frames
        self.audio_sampling_rate = audio_sampling_rate
        self.frame_skip = frame_skip
        self.transform = transform
        
        self.pairs = []  # List to store (audio_file_path, video_file_path) pairs
        
        subjects = sorted(os.listdir(audio_root))
        for subj in subjects:
            if self.subs is not None and subj not in self.subs:
                continue
            audio_subj_path = os.path.join(audio_root, subj)
            video_subj_path = os.path.join(video_root, subj,  "2drt", "video")
            if not (os.path.isdir(audio_subj_path) and os.path.isdir(video_subj_path)):
                continue

            audio_files = [f for f in os.listdir(audio_subj_path) if f.endswith('.wav')] # List audio files in the subjects folder
            if self.filter_keyword:
                audio_files = [f for f in audio_files if self.filter_keyword in f]
            
            for audio_file in audio_files: # For each audio file, look for a corresponding video file (with .mp4 extension)
                base_name = os.path.splitext(audio_file)[0]
                video_file = base_name + ".mp4"
                if video_file in os.listdir(video_subj_path):
                    audio_full_path = os.path.join(audio_subj_path, audio_file)
                    video_full_path = os.path.join(video_subj_path, video_file)
                    self.pairs.append((audio_full_path, video_full_path))
                    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        audio_path, video_path = self.pairs[idx]
        
        waveform, sr = torchaudio.load(audio_path)  # (channels, num_samples) -> SR is 48000 not 16000 why?
        if sr != self.audio_sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.audio_sampling_rate)
            waveform = resampler(waveform)

        if waveform.size(0) == 1:
            waveform = waveform.squeeze(0)  #  num_samples
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            frame_count += 1
            if self.video_max_frames is not None and frame_count >= self.video_max_frames:
                break
        cap.release()

        frames = np.array(frames) # slow if I create the tensor directly, interesting! 
        frames = torch.from_numpy(frames).float() / 255.0  # (num_frames, H, W)
        frames = frames.unsqueeze(1)  # (num_frames, 1, H, W) -> channel dim 1 for 2d convs

        if self.transform is not None:
            waveform, frames = self.transform(waveform, frames) # temporal window or contextual sampling

        return waveform, frames, audio_path, video_path

    @staticmethod
    def collate(batch, sw_transform=None):
        waveforms, frames, audio_paths, video_paths = zip(*batch)

        max_audio_length = max([waveform.size(0) for waveform in waveforms])
        padded_waveforms = [torch.nn.functional.pad(waveform, (0, max_audio_length - waveform.size(0))) for waveform in waveforms]
        padded_waveforms = torch.stack(padded_waveforms)

        max_frames = max([frame.size(0) for frame in frames])    
        f = []
        for frame in frames:
            if frame.size(0) < max_frames:
                padding = torch.zeros(max_frames - frame.size(0), frame.size(1), frame.size(2), frame.size(3))
                frame = torch.cat((frame, padding), dim=0)
            f.append(frame)
        padded_frames = torch.stack(f)

        if sw_transform is not None:
            padded_waveforms, padded_frames = sw_transform(padded_waveforms, padded_frames)
        return padded_waveforms, padded_frames, audio_paths, video_paths

# %% Debugging Batching and Collate
if __name__ == "__main__":
    from AVDataset import AVDataset
    from torch.utils.data import DataLoader
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]

    dataset = AVDataset(audio_root=audio_root, 
                        video_root=video_root, 
                        subs=nSubs, 
                        filter_keyword="vcv",
                        transform=None,
                        video_max_frames=None,
                        audio_sampling_rate=16000,
                        frame_skip=1)
    print("Number of pairs:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=AVDataset.collate) # Batch / Collation
    for (waveform, frames, audio_path, video_path) in dataloader:
        print(waveform.shape, frames.shape, audio_path, video_path)
        break
    
# %% Debugging Transform 1
if __name__ == "__main__":
    from AVDataset import AVDataset
    from torch.utils.data import DataLoader
    from transforms import TemporalWindowTransform
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 3)]

    temporal_transform = TemporalWindowTransform(window_size_sec=2, audio_sample_rate=16000, video_fps=83)

    dataset = AVDataset(audio_root=audio_root, 
                        video_root=video_root, 
                        subs=nSubs, 
                        filter_keyword="vcv",
                        transform=temporal_transform,
                        video_max_frames=None,
                        audio_sampling_rate=16000,
                        frame_skip=1)
    print("Number of pairs:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=AVDataset.collate) # Batch / Collation
    for (waveform, frames, audio_path, video_path) in dataloader:
        print(waveform.shape, frames.shape, audio_path, video_path)
        break

# %% Debugging Transform 2
if __name__ == "__main__":
    from AVDataset import AVDataset
    from torch.utils.data import DataLoader
    from transforms import ContextualSamplingTransform
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 3)]

    contextual_transform = ContextualSamplingTransform(context_size=2, audio_sample_rate=16000, video_fps=83)

    dataset = AVDataset(audio_root=audio_root, 
                        video_root=video_root, 
                        subs=nSubs, 
                        filter_keyword="vcv",
                        transform=contextual_transform,
                        video_max_frames=None,
                        audio_sampling_rate=16000,
                        frame_skip=1)
    print("Number of pairs:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=AVDataset.collate) # Batch / Collation
    for (waveform, frames, audio_path, video_path) in dataloader:
        print(waveform.shape, frames.shape, audio_path, video_path)
        break

# %% Debugging Sliding Window
if __name__ == "__main__":
    from AVDataset import AVDataset
    from torch.utils.data import DataLoader
    from transforms import SlidingWindowTransform
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 3)]

    sw_transform = SlidingWindowTransform(window_duration=1, step_duration=0.5, audio_sample_rate=16000, video_fps=83)

    dataset = AVDataset(audio_root=audio_root, 
                        video_root=video_root, 
                        subs=nSubs, 
                        filter_keyword="vcv",
                        transform=None,
                        video_max_frames=None,
                        audio_sampling_rate=16000,
                        frame_skip=1)
    print("Number of pairs:", len(dataset))

    print("SW transform outside collate")
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=AVDataset.collate) # Batch / Collation
    for (waveform, frames, audio_path, video_path) in dataloader:
        print(waveform.shape, frames.shape, audio_path, video_path)
        padded_waveforms, padded_frames = sw_transform(waveform, frames)
        print(padded_waveforms.shape, padded_frames.shape, audio_path, video_path)
        video_reconst = sw_transform.overlap_add(padded_frames)
        print(video_reconst.shape)
        print("")
    
    print("SW transform inside collate")
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, 
                            collate_fn=lambda batch: AVDataset.collate(batch, sw_transform)) # Batch / Collation
    for (waveform, frames, audio_path, video_path) in dataloader:
        print(waveform.shape, frames.shape, audio_path, video_path)
        video_reconst = sw_transform.overlap_add(frames)
        print(video_reconst.shape)
        print("")
    
    print("Video Min/Max:", frames.min(), frames.max())
    print("Video Reconstr. Min/Max:", video_reconst.min(), video_reconst.max())

    # import utils as ut
    # ut.Video(frames[1, 0:5].reshape(-1, 1, 84, 84), mode=2, subrate=1)
    # ut.Audio(waveform[1, 0:5, :].reshape(1, -1))
