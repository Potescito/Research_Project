"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def save_ds(path: str, dataset: dict):
    """ 
    Save a dataset as a .npz file.
    """
    np.savez(path, **dataset)

def load_ds(path: str):
    """ 
    Load a dataset from a .npz file.
    """
    ds = np.load(path, allow_pickle=True)
    ds = {key: ds[key].item() if ds[key].ndim==0 else ds[key] for key in ds}
    return ds

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