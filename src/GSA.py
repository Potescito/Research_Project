"""
Research Project WiSe 2024/25
- Original Implementation: Fraunhofer IIS / OSS: 
  hernande (julian.andres.hernandez.potes@fraunhfer.de)

- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr

def GSA(LR_HS, HR_MS):
    """
    Gram-Schmidt Adaptive spectral sharpening (CS):
    B. Aiazzi et. al, "MS + Pan image fusion by 
    an enhanced Gram-Schmidt spectral sharpening", 
    Ref: https://www.earsel.org/symposia/2006-symposium-Warsaw/pdf/340.pdf

    Args:
        LR_HS (np.ndarray): Low Resolution HS Data
        HR_MS (np.ndarray): High Resolution MS Data
    
    Returns:
        np.ndarray: Fused High Resolution HS Data
    """
    ratioH = HR_MS.shape[1] // LR_HS.shape[1]
    ratioW = HR_MS.shape[2] // LR_HS.shape[2]
    
    hc = LR_HS.shape[0]
    mc, mh, mw = HR_MS.shape
    HR_HS = np.zeros((hc, mh, mw))
    
    # High resolution MS downsampling
    LR_MS = ndimage.zoom(HR_MS, (1, 1/ratioH, 1/ratioW), order=3) # P in paper (bilinear, bicubic?)

    # Correlation matrix btwn LR_MS and LR_HS images (pearson, linear) -> (mc, hc) shape
    A = np.array([[pearsonr(LR_MS.reshape(mc, -1)[i], LR_HS.reshape(hc, -1)[j])[0] \
        for j in range(hc)] for i in range(mc)])
    
    max_corr = np.argmax(A, axis=0) # (hc, ) -> per hc, which mc is more correlated
    
    # GSA per correlated bands
    for i in range(mc):
        correlated_hc = np.where(max_corr == i)[0] # find the hc bands correlated to the i-th band
        if len(correlated_hc) > 0:
            HR_HS[correlated_hc, ...] = _GSA(ratioH, ratioW, LR_HS[correlated_hc, ...], HR_MS[i, ...])
    return HR_HS
    
def _GSA(ratioH, ratioW, hs, ms):
    """
    Args:
        ratioH (int): spatial ratio in height
        ratioW (int): spatial ratio in width
        hs (np.ndarray): selected correlated hyperspectral bands images
        ms (np.ndarray): selected multispectral band image

    Returns:
        np.ndarray: fused HS image per correlated bands
    """
    # Upsample the hs (bicubic to capture more details)
    up_hs = ndimage.zoom(hs, (1, ratioH, ratioW), order=3)
    
    # Remove means (force independency?)
    hs_nomean = hs - np.mean(hs, axis=(1, 2), keepdims=True) # (hs, 1, 1) means
    ms_nomean = ms - np.mean(ms)
    up_hs_nomean = up_hs - np.mean(up_hs, axis=(1, 2), keepdims=True) 
    
    # Compute estimate (synthetic intensity) LR_MS and then compute alphas coeffs (check paper) -> least squares
    down_ms = ndimage.zoom(ms_nomean, (hs.shape[1] / ms.shape[0], hs.shape[2] / ms.shape[1]), order=1)
    
    hs_nomean_ext = np.concatenate((hs_nomean, np.ones((1, hs.shape[1], hs.shape[2]))), axis=0)
    alphas = np.linalg.lstsq(hs_nomean_ext.reshape(hs_nomean_ext.shape[0], -1).T, down_ms.ravel(), rcond=None)[0]
    
    uphs_nomean_ext = np.concatenate((up_hs_nomean, np.ones((1, up_hs_nomean.shape[1], up_hs_nomean.shape[2]))), axis=0)
    I = np.tensordot(alphas, uphs_nomean_ext, axes=([0], [0])) # sum over alpha[0] and uphs[0]->ch
    I = I - np.mean(I) # w/o mean and same shape of ms (this is the synthetic image) -> LR approx of the ms image (HR) -> initial estimate
    
    # High Resolution features
    delta = ms_nomean - I
    delta = np.tile(delta.ravel(), (hs.shape[0] + 1, 1)).T # shape (w*h, ch)
    
    # Regression Coeffs for each band of the upsampled LR-HS (relationship btwn I and each spectral band in hs)
    reg_coeffs = np.ones(hs.shape[0] + 1) # first coeff will be bias term here (WEIGHTS)
    for i in range(hs.shape[0]):
        cov = np.cov(I.ravel(), up_hs_nomean[i, ...].ravel()) # 00- var(I) (estimate), 11- var(hs), 01- cov(I,h), 10- cov(h,I) [we need how changes in hs are associated with changes in I]
        reg_coeffs[i + 1] = cov[0, 1]  / np.var(I) # degree to which the ch correlates linearly with the I
    
    # Fusion! -> fuse synthetic I with HR info and weight each band by its reg coeff
    fused_data = np.stack([I.ravel()] + [up_hs_nomean[i].ravel() for i in range(hs.shape[0])], axis=1) # I -> spacial details (estimate), UPHS -> spectral details
    reg_coeffs = np.tile(reg_coeffs, (fused_data.shape[0], 1)) # fused_data shape -> (w*h, ch)
    
    fused_data_hr = fused_data + delta*reg_coeffs
    fused_data_hr = fused_data_hr[:, 1:].T.reshape(*up_hs.shape)
    
    # Mean equalization
    fused_data_hr = fused_data_hr - np.mean(fused_data_hr, axis=(1, 2), keepdims=True) + np.mean(up_hs, axis=(1, 2), keepdims=True)
    return fused_data_hr