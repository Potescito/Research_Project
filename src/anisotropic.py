"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np

def anisotropic_diffusion(img: np.ndarray, num_iter: int=10, kappa: int=50, gamma: float=0.1, option: int=1):
    """Anisotropic diffusion filter for image denoising.

    Args:
        img (np.ndarray): Image to be denoised. Expected shape (Frames, Height, Width)
        num_iter (int, optional): Number of iterations. Defaults to 10.
        kappa (int, optional): Conduction coefficient, sensitivity to edges, lower preserves edges. Defaults to 50.
        gamma (float, optional): Speed of diffusion, higher: faster diffusion. Defaults to 0.1.
        option (int, optional): Algorithm to use, exponential = 1, inverse = 2. Defaults to 1.

    Returns:
        np.ndarray: Denoised image.
    """
    imgdtype = img.dtype
    if imgdtype != np.float32: 
        img = img.astype(np.float32) # for faster computations in float32
    
    for _ in range(num_iter):
        # Gradients
        deltaN = np.roll(img, -1, axis=1) - img
        deltaS = np.roll(img, 1, axis=1) - img
        deltaE = np.roll(img, -1, axis=2) - img
        deltaW = np.roll(img, 1, axis=2) - img

        # Compute diffusion coefficients -> perona malik difussion
        if option == 1:
            cN = np.exp(-(deltaN/kappa)**2)
            cS = np.exp(-(deltaS/kappa)**2)
            cE = np.exp(-(deltaE/kappa)**2)
            cW = np.exp(-(deltaW/kappa)**2)
        elif option == 2:
            cN = 1 / (1 + (deltaN/kappa)**2)
            cS = 1 / (1 + (deltaS/kappa)**2)
            cE = 1 / (1 + (deltaE/kappa)**2)
            cW = 1 / (1 + (deltaW/kappa)**2)

        img += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
    return img.astype(imgdtype)