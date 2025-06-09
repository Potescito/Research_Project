"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
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

    def overlap_add(self, video_windows, window_func='rect'):
        """
        Reconstructs a full video from overlapping window outputs using a weighted
        overlap-add method with a specified windowing function.

        Args:
            video_windows (torch.Tensor): Tensor of video chunks, shape
                                          (batch, num_windows, window_video, C, H, W).
            window_func (str): The window function to use for smooth blending.
                               Options: 'hann', 'triang' (triangular), or 'rect' (no weighting).

        Returns:
            torch.Tensor: Reconstructed video of shape (B, final_length, C, H, W).
        """
        B, num_windows, window_video, C, H, W = video_windows.shape
        step_video = self.step_video
        
        final_length = (num_windows - 1) * step_video + window_video

        final_video = torch.zeros(B, final_length, C, H, W, device=video_windows.device)
        norm_buffer = torch.zeros(B, final_length, C, H, W, device=video_windows.device)

        if window_func == 'hann':
            win = torch.hann_window(window_video, periodic=True, device=video_windows.device)
        elif window_func == 'triang':
            win = torch.bartlett_window(window_video, periodic=True, device=video_windows.device) # can be triangular
        else:
            win = torch.ones(window_video, device=video_windows.device)

        # Reshape window for broadcasting against a video chunk
        win = win.view(1, window_video, 1, 1, 1)

        for i in range(num_windows):
            start = i * step_video
            end = start + window_video
            
            current_chunk = video_windows[:, i]
            windowed_chunk = current_chunk * win # window function
            
            final_video[:, start:end] += windowed_chunk # add to the sum buffer
            norm_buffer[:, start:end] += win # add the window to the normalization buffer

        norm_buffer[norm_buffer == 0] = 1.0 # div by zero protection
        
        reconstructed_video = final_video / norm_buffer
        return reconstructed_video
