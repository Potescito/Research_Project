"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def NRMSE(gt: np.ndarray, dns: np.ndarray):
    """
    Normalized Root Mean Squared Error (NRMSE) metric.
    If all the frames are given, the NRMSE is computed across ALL frames.
    Standard error metric for image denoising.
    Args:
        gt (np.ndarray): Ground truth image.
        dns (np.ndarray): Denoised image.
    
    Returns:
        float: NMSE value.
    """
    gt = gt.flatten() / gt.max()
    dns = dns.flatten() / dns.max()
    
    squared_diff = np.sum((gt - dns)**2)
    nmse = squared_diff / np.sum(gt**2)

    return np.sqrt(nmse)

def SSIM(gt: np.ndarray, dns: np.ndarray):
    """
    Mean Structural Similarity Index -> (SSIM) metric.
    Better than simple MSE because it takes the texture into account.
    
    If all the frames are given, the SSIM is computed across ALL frames -> axis 0.
    Args:
        gt (np.ndarray): Ground truth image.
        dns (np.ndarray): Denoised image.
    
    Returns:
        float: SSIM value.
    """
    if gt.ndim == 3:
        axis = 0
    else:
        axis = None
    if gt.dtype != dns.dtype:
        dns = dns.astype(gt.dtype)
    data_range = max(gt.max(), dns.max()) - min(gt.min(), dns.min())
    return structural_similarity(gt, dns, data_range=data_range, channel_axis=axis)

def PSNR(gt: np.ndarray, dns: np.ndarray):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric. 
    If all the frames are given, the PSNR is computed across ALL frames.
    
    Args:
        gt (np.ndarray): Ground truth image.
        dns (np.ndarray): Denoised image.
    
    Returns:
        float: PSNR value.
    """
    gt = gt.flatten()
    dns = dns.flatten() 
    return peak_signal_noise_ratio(gt, dns)