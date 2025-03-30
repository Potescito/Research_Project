"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np
import torch

class SlidingWindowTransform:
    """
    Applies a fixed sliding window with overlap to both audio and video data.
    Always starts from the beginning and traverses the entire sequence.

    Args:
        window_duration (float): Duration of each window in seconds.
        step_duration (float): Step size (in seconds) between windows.
        audio_sample_rate (int): Audio sample rate (default: 16000).
        video_fps (int): Video frame rate (default: 83).
    """
    def __init__(self, window_duration, step_duration, audio_sample_rate=16000, video_fps=83):
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        
        self.window_audio = int(window_duration * audio_sample_rate)
        self.step_audio   = int(step_duration * audio_sample_rate)
        self.window_video = int(window_duration * video_fps)
        self.step_video   = int(step_duration * video_fps)
        
    def __call__(self, waveform, frames):
        """
        Args:
            waveform (torch.Tensor): Audio tensor of shape (batch, num_samples).
            frames (torch.Tensor): Video tensor of shape (batch, num_frames, 1, H, W).

        Returns:
            tuple: (audio_windows, video_windows) where:
                - audio_windows has shape (batch, num_windows, window_audio)
                - video_windows has shape (batch, num_windows, window_video, 1, H, W)
        """
        # Apply sliding window using unfold.
        audio_windows = waveform.unfold(dimension=1, size=self.window_audio, step=self.step_audio)
        video_windows = frames.unfold(dimension=1, size=self.window_video, step=self.step_video)
        video_windows = video_windows.permute(0, 1, 5, 2, 3, 4)
        return audio_windows, video_windows

    def overlap_add(self, video_windows, window_duration=None, step_duration=None):
        """
        Reconstructs a full video from overlapping window outputs using overlap-add.
        
        Args:
            video_windows (torch.Tensor):  (batch, num_windows, window_video, 1, H, W)
            window_duration (float): Duration of each window in seconds. Defaults to class.
            step_duration (float): Step size (in seconds) between windows. Defaults to class.
            
        Returns:
            torch.Tensor: Reconstructed video of shape (B, final_length, 1, H, W), 
                          where final_length = (num_windows - 1) * step_video + window_video.
        """
        if window_duration is None:
            window_duration = self.window_duration
        if step_duration is None:
            step_duration = self.step_duration

        B, num_windows, _, C, H, W = video_windows.shape
        final_length = (num_windows - 1) * step_duration + window_duration
        
        final_video = torch.zeros(B, final_length, C, H, W, device=video_windows.device)
        count = torch.zeros(B, final_length, 1, H, W, device=video_windows.device)
        
        for i in range(num_windows):
            start = i * step_duration
            end = start + window_duration
            final_video[:, start:end] += video_windows[:, i]
            count[:, start:end] += 1
            
        # Average the overlapping regions.
        final_video = final_video / count
        return final_video

# ====================================================================
# ====================================================================

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


# ====================================================================
# ====================================================================

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
