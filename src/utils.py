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
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import Audio as disAu, HTML, display
except ImportError:
    print("IPython is not installed. Audio/Video display will not work.")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif']
plt.rcParams["font.size"] = 12


def Audio(data, rate: int=16000):
    """
    Display an audio waveform in a Jupyter notebook.

    Args:
        data (np.ndarray/torch.tensor): Audio data to be displayed (ch, samples).
        rate (int): Sample rate of the audio data.
    """
    data_play = data.detach().cpu().numpy() if hasattr(data, 'cpu') else data
    try:
        display(disAu(data_play, rate=rate), display_id=False)
    except Exception as e:
        print(f"Error displaying audio: {e}")
        print("Ensure that the data is in the correct format (ch, samples) or (samples) and try again.")
        raise e


def Video(data, subrate=10, mode=1, fps=83):
    """
    Display a video in a Jupyter notebook.

    Args:
        data (np.ndarray/torch.tensor): Video data to be displayed (frames, ch, H, W).
        mode (int): 1 for clip (faster), 2  for advanced viewer (slower).
        fps (int): Frames per second for the video.
    """
    frames_np = data.detach().cpu().numpy() if hasattr(data, 'cpu') else data # Shape: (F, C, H, W)
    if frames_np.shape[1] == 1: # Grayscale: (F, 1, H, W)
        processed_frames = np.transpose(frames_np, (0, 2, 3, 1)).squeeze(axis=3) # (F, H, W)
        cmap_choice = 'gray'
    elif frames_np.shape[1] == 3: # RGB: (F, C, H, W)
        processed_frames = np.transpose(frames_np, (0, 2, 3, 1)) # (F, H, W, C)
        cmap_choice = None
    else:
        raise ValueError(f"Unsupported channel size: {frames_np.shape[1]}")    

    # Subsample the video frames
    subsampled_anim_frames = processed_frames[::subrate]
    print(f"Original frame count for animation: {processed_frames.shape[0]}")
    print(f"Subsampled frame count for animation: {subsampled_anim_frames.shape[0]}")
    	
    # Normalize the subsampled data
    min_val_anim, max_val_anim = subsampled_anim_frames.min(), subsampled_anim_frames.max()
    if max_val_anim > min_val_anim:
        anim_frames_to_show = (subsampled_anim_frames - min_val_anim) / (max_val_anim - min_val_anim)
    else:
        anim_frames_to_show = np.zeros_like(subsampled_anim_frames)

    if anim_frames_to_show.shape[0] > 0:
        fig, ax = plt.subplots()
        plt.close(fig) # Close initial plot -> no display of static plot of the first frame 
        ax.set_axis_off()
        img_obj = ax.imshow(anim_frames_to_show[0], cmap=cmap_choice, vmin=0, vmax=1)
    
        def update_anim_frame(frame_idx):
            img_obj.set_data(anim_frames_to_show[frame_idx])
            return [img_obj]

        # Create animation with fewer frames
        # Interval: 1000ms / desired_fps. E.g., 1000/10fps = 100ms
        anim = FuncAnimation(fig, update_anim_frame, frames=anim_frames_to_show.shape[0], interval=1000/fps, blit=True)

        # Display the animation
        if mode == 1:
            # to_html5_video for better performance
            try:
                # Check if ffmpeg is available, otherwise to_html5_video might fail silently or error
                # A more robust check might be needed depending on your system
                import shutil
                if shutil.which('ffmpeg'):
                    print("Attempting to display with animation.to_html5_video() (requires ffmpeg)")
                    display(HTML(anim.to_html5_video()))
                else:
                    print("ffmpeg not found. Falling back to animation.to_jshtml() (can be slow).")
                    print("Install ffmpeg for better video animation performance in notebooks.")
                    display(HTML(anim.to_jshtml())) # Fallback, can be very slow for many frames
            except Exception as e:
                print(f"Error displaying animation: {e}")
                print("Consider installing ffmpeg or further reducing frame count.")
        elif mode == 2:
            try:
                display(HTML(anim.to_jshtml())) # to_jshtml is generally reliable
            except Exception as e:
                print(f"Error displaying animation: {e}")
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 1 for clip or 2 for advanced viewer.")
    else:
        raise ValueError("No frames to display after subsampling.")


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