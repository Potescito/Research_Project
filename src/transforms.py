"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np
import torch

class TemporalWindowTransform:
    """
    Extracts a fixed-length temporal window from audio and video.
    
    This transform synchronizes the window between the modalities based on a common start time.
    For a given window size in seconds, it extracts the corresponding number of audio samples and video frames.
    
    Args:
        window_size_sec (float): The length of the temporal window in seconds.
        audio_sample_rate (int): Audio sample rate (samples per second). Default is 16000.
        video_fps (int): Video frame rate (frames per second). Default is 83.
    """
    def __init__(self, window_size_sec, audio_sample_rate=16000, video_fps=83):
        self.window_size_sec = window_size_sec
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        
    def __call__(self, waveform, frames):
        """
        Args:
            waveform (torch.Tensor): Audio waveform with shape (num_samples,).
            frames (torch.Tensor): Video frames with shape (num_frames, 1, H, W).
            
        Returns:
            tuple: (windowed_waveform, windowed_frames)
        """
        # Compute total duration for each modality.
        total_duration_audio = waveform.shape[0] / self.audio_sample_rate
        total_duration_video = frames.shape[0] / self.video_fps
        total_duration = min(total_duration_audio, total_duration_video) # they might differ slightly (floating point)
        
        if total_duration < self.window_size_sec:
            raise ValueError("Sequence too short for the requested temporal window.")
        
        # Pick a random start time ensuring the window fits in the sequence.
        start_time = np.random.uniform(0, total_duration - self.window_size_sec)
        
        start_audio = int(start_time * self.audio_sample_rate)
        start_video = int(start_time * self.video_fps)
        
        window_audio_length = int(self.window_size_sec * self.audio_sample_rate)
        window_video_length = int(self.window_size_sec * self.video_fps)
        
        end_audio = start_audio + window_audio_length
        end_video = start_video + window_video_length
        
        windowed_waveform = waveform[start_audio:end_audio]
        windowed_frames = frames[start_video:end_video]
        
        return windowed_waveform, windowed_frames


class ContextualSamplingTransform:
    """
    Samples a contextual window around a randomly chosen central video frame.
    
    For a given context size (i.e., number of frames before and after the central frame),
    this transform extracts the video frames in that window and the corresponding audio segment.
    
    Args:
        context_size (int): Number of frames to include before and after the central frame.
        audio_sample_rate (int): Audio sample rate (samples per second). Default is 16000.
        video_fps (int): Video frame rate (frames per second). Default is 83.
    """
    def __init__(self, context_size=1, audio_sample_rate=16000, video_fps=83):
        self.context_size = context_size
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        
    def __call__(self, waveform, frames):
        """
        Args:
            waveform (torch.Tensor): Audio waveform with shape (num_samples,).
            frames (torch.Tensor): Video frames with shape (num_frames, 1, H, W).
            
        Returns:
            tuple: (context_audio, context_frames, central_index)
        """
        num_frames = frames.shape[0]
        required_frames = 2 * self.context_size + 1
        
        if num_frames < required_frames:
            raise ValueError("Not enough frames for the requested context size.")
        
        # Choose a random central frame index ensuring enough context on both sides.
        central_index = np.random.randint(self.context_size, num_frames - self.context_size)
        context_frames = frames[central_index - self.context_size: central_index + self.context_size + 1]
        
        # Compute the corresponding audio time interval.
        start_time = (central_index - self.context_size) / self.video_fps
        end_time = (central_index + self.context_size + 1) / self.video_fps
        
        start_audio = int(start_time * self.audio_sample_rate)
        end_audio = int(end_time * self.audio_sample_rate)
        
        context_audio = waveform[start_audio:end_audio]
        
        return context_audio, context_frames
