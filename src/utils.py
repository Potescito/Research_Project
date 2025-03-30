"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import os
import torch
import torchaudio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif']
plt.rcParams["font.size"] = 12

def save_ds(path: str, dataset: dict):
    """ 
    Save a dataset as a .npz file.

    Args:
        path (str): Path to save the .npz file.
        dataset (dict): Dictionary containing the dataset.
    """
    np.savez(path, **dataset)


def load_ds(path: str):
    """ 
    Load a dataset from a .npz file.

    Args:
        path (str): Path to the .npz file.
    """
    ds = np.load(path, allow_pickle=True)
    ds = {key: ds[key].item() if ds[key].ndim==0 else ds[key] for key in ds}
    return ds


def save_waveforms(waveforms, names, output_dir='waveforms', sample_rate=16000):
    """
    Saves each waveform in the batch as an audio .wav file.
    
    Args:
        waveforms (torch.Tensor or np.ndarray): Batched waveforms with shape (batch_size, num_samples).
        names(list): List of names for each waveform -> can be the audio path of the dataloader.
        output_dir (str): Directory to save the audio .wav files.
        sample_rate (int): Sample rate for the saved audio.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(waveforms, 'cpu'):
        waveforms = waveforms.cpu().numpy()
    
    for i, waveform in enumerate(waveforms):
        if waveform.ndim == 1:
            waveform_to_save = np.expand_dims(waveform, axis=0) # must have channel dimension
        else:
            waveform_to_save = waveform
        
        name = names[i].split("/")[-1].split("\\")[-1].split(".")[0]

        file_path = os.path.join(output_dir, f'{name}.wav')
        waveform_tensor = torch.tensor(waveform_to_save, dtype=torch.float32)
        torchaudio.save(file_path, waveform_tensor, sample_rate)


def save_videos(videos, names, output_dir='videos', fps=83, codec='mp4v'):
    """
    Saves each video in the batch as an MP4 file using OpenCV's VideoWriter.
    
    Args:
        videos (torch.Tensor or np.ndarray): Batched videos with shape (batch_size, frames, 1, height, width).
        names(list): List of names for each video -> can be the video path of the dataloader.
        output_dir (str): Directory to save the video MP4 files.
        fps (int): Frames per second for the output video.
        codec (str): FourCC code for the video codec. Default is 'mp4v'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(videos, 'cpu'):
        videos = videos.cpu().numpy()
    
    batch_size = videos.shape[0]

    for i in range(batch_size):
        video = videos[i]  # shape: (frames, 1, height, width)
        video = np.squeeze(video, axis=1)  # now shape: (frames, height, width)
        video_uint8 = (video * 255).astype(np.uint8)
        
        # Determine height and width from the first frame.
        height, width = video_uint8[0].shape
        
        name = names[i].split("/")[-1].split("\\")[-1].split(".")[0]

        output_path = os.path.join(output_dir, f'{name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Create VideoWriter. If your codec struggles with grayscale, you could convert frames to BGR.
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
        print(output_path)
        for frame in video_uint8:
            out.write(frame)
        out.release()


def imshow(imgs: list,
           snr: bool = True,
           titles: list = None,
           fig_size: tuple = (20, 5),
           font_size: int = 15,
           font_color: str = "yellow",
           font_weight: str = "bold",
           signal_box_coords: tuple = (32, 32, 20, 20),
           noise_box_coords: tuple = (63, 0, 20, 20),
           box_linewidth: int = 2,
           box_edgecolor: str = "r"):
    """
    This function is an alternative to the imshow function from matplotlib.
    """
    f = plt.figure(figsize=fig_size)
    titles = [None] * len(imgs) if titles is None else titles

    for i in range(len(imgs)):
        ax = f.add_subplot(1, len(imgs), i+1)
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
        ax.set_title(titles[i])
        if snr:
            annotate_metrics(imgs[i], ax, signal_box_coords, noise_box_coords, font_size, font_color, font_weight)            
            draw_box(ax, signal_box_coords, box_linewidth, box_edgecolor)
            draw_box(ax, noise_box_coords, box_linewidth, box_edgecolor)
    plt.show()

def calc_snr(src: np.ndarray, signal_box_coords: tuple, noise_box_coords: tuple):
    """ 
    This function calculates the SNR value of an image from the given signal and noise box coordinates.
    It will be mean of the signal over the standard deviation of the noise.
    """
    signal_y, signal_x, signal_width, signal_height = signal_box_coords
    noise_y, noise_x, noise_width, noise_height = noise_box_coords

    if src.ndim == 3:
        signal = src[:, signal_y: signal_y + signal_height, signal_x:signal_x + signal_width]
        noise = src[:, noise_y: noise_y + noise_height, noise_x:noise_x + noise_width]
    else:
        signal = src[signal_y: signal_y + signal_height, signal_x:signal_x + signal_width]
        noise = src[noise_y: noise_y + noise_height, noise_x:noise_x + noise_width]
    return np.mean(signal) / np.std(noise)

def annotate_metrics(src : np.ndarray, 
                     ax : plt.Axes, 
                     signal_box_coords: tuple = (32, 32, 20, 20), 
                     noise_box_coords: tuple = (63, 0, 20, 20),  
                     font_size: int = 15, 
                     font_color: str = "yellow", 
                     font_weight: str = "bold"):
    """ 
    This function annotates the SNR value on the image. 
    """

    snr = calc_snr(src, signal_box_coords, noise_box_coords)    
    text = f"{snr:.3f}"

    ax.annotate(
        text,
        xy=(0, 1),
        xytext=(2, -2),
        fontsize=font_size,
        color=font_color,
        xycoords="axes fraction",
        textcoords="offset points",
        horizontalalignment="left",
        verticalalignment="top",
        fontweight=font_weight,
    )

def draw_box(ax, box_coords: tuple, box_linewidth: int = 2, box_edgecolor: str = "r"):
    """ 
    This function draws a box on the image.
    """
    y, x, width, height = box_coords
    rect = Rectangle(
        (x, y), width=width, height=height, linewidth=box_linewidth, edgecolor=box_edgecolor, facecolor="none"
    )
    ax.add_patch(rect)