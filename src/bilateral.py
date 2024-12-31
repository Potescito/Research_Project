"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np
from scipy.ndimage import convolve

def bilateral_filter(image, sigma_spatial=0.9, sigma_intensity=0.5):
    """Bilateral filtering.

    Args:
        image (np.ndarray): Input image of shape (frames, height, width).
        sigma_spatial (float): Standard deviation for spatial Gaussian kernel. How much nearby pixels influence the filtering
        sigma_intensity (float): Standard deviation for intensity Gaussian kernel. How much pixels with similar intensity influence the filtering

    Returns:
        (np.ndarray): Denoised image of the same shape as the input.
    """
    frames, height, width = image.shape
    output = np.zeros_like(image)

    pad_width = int(3 * sigma_spatial)  # 3-sigma rule
    x, y = np.meshgrid(
        np.arange(-pad_width, pad_width + 1),
        np.arange(-pad_width, pad_width + 1)
    )
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2)) # gaussian, faster than calling it

    for c in range(frames):
        channel = image[c]
        padded_channel = np.pad(channel, pad_width, mode='reflect')
        
        neighborhoods = np.lib.stride_tricks.sliding_window_view(
            padded_channel, (2 * pad_width + 1, 2 * pad_width + 1)
        )
        neighborhoods = neighborhoods.reshape(height, width, -1)

        intensity_diffs = neighborhoods - channel[:, :, None]
        intensity_weights = np.exp(-(intensity_diffs**2) / (2 * sigma_intensity**2))

        combined_weights = spatial_weights.ravel()[None, None, :] * intensity_weights
        filtered_values = np.sum(combined_weights * neighborhoods, axis=2)
        normalization = np.sum(combined_weights, axis=2)
        filtered_channel = (filtered_values / normalization).reshape(height, width)

        output[c] = filtered_channel

    return output

# %% =====ß
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from VideoProcessor import VideoProcessor
    from bilateral import bilateral_filter
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())

    noise_type = "speckle"
    noisy_ds = vp.noise(dataset, type=noise_type, mean=0, std=0.2)

    denoised = bilateral_filter(noisy_ds[n[0]], sigma_spatial=0.9, sigma_intensity=0.5)
    plt.imshow(noisy_ds[n[0]][10], cmap="gray")
    plt.show()
    plt.imshow(denoised[10], cmap="gray")
    plt.show()