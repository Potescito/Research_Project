"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
- Adapted from https://huggingface.co/blog/annotated-diffusion
"""
import torch
import torch.nn.functional as F

# =========================================
# Schedulers
# =========================================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    
    Args:
        timesteps (int): Number of diffusion steps.
        s (float): Scaling factor for the cosine schedule, default is 0.008.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# =========================================
# Helpers
# =========================================
def get_alphas_and_cumprod(betas):
    """
    Calculates alphas and their cumulative products from betas.
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


def get_diffusion_parameters(betas, device=torch.device("cpu")):
    """
    Pre-computes all necessary parameters for the diffusion process.

    Args:
        timesteps (int): Total number of diffusion timesteps (T).
        betas (torch.Tensor): Beta schedule tensor of shape (T,).
        device (str or torch.device): Device to store the parameters on.

    Returns:
        dict: A dictionary containing various diffusion parameters:
              'betas', 'alphas', 'alphas_cumprod',
              'alphas_cumprod_prev' (for sampling),
              'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
              'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
              'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
              'posterior_log_variance_clipped', 'posterior_mean_coef1',
              'posterior_mean_coef2'.
              All values are torch.Tensors on the specified device.
    """
    betas = betas.to(device)
    alphas, alphas_cumprod = get_alphas_and_cumprod(betas)

    # Previous cumulative product (for sampling x_0 from x_1) at t0 then 1
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Pad at the beginning

    # --- Parameters for q(x_t | x_0) ---
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod) # Used by some loss functions

    # --- Parameters for q(x_{t-1} | x_t, x_0) (the posterior for DDPM sampling) ---
    # (Used in reverse process for DDPM: predicts x_0 from x_t, then samples x_{t-1})
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))  # clip this to avoid issues where beta_t is too small (especially at t=0)
    
    # Posterior mean coefficients:
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    # Alternative parameters for sampling (often used in DDIM or simplified DDPM derivations)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)

    return {
        'betas': betas,                                          
        'alphas': alphas,                                         
        'alphas_cumprod': alphas_cumprod,                         
        'alphas_cumprod_prev': alphas_cumprod_prev,               
        
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,               
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'log_one_minus_alphas_cumprod': log_one_minus_alphas_cumprod,
        
        'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
        
        'posterior_variance': posterior_variance,                 
        'posterior_log_variance_clipped': posterior_log_variance_clipped,
        'posterior_mean_coef1': posterior_mean_coef1,
        'posterior_mean_coef2': posterior_mean_coef2,
    }


def extract(schedule_params, t, x_shape):
    """
    Extracts values from a pre-computed schedule for a batch of timesteps.

    Args:
        schedule_params (torch.Tensor): Tensor from the pre-computed schedule 
                                        (e.g., sqrt_alphas_cumprod), shape (T,).
        t (torch.Tensor): Batch of timesteps, shape (batch_size,).
        x_shape (tuple): Shape of the data tensor x (e.g., (batch_size, C, H, W)).

    Returns:
        torch.Tensor: Extracted values, reshaped to be broadcastable with x.
                      Shape: (batch_size, 1, 1, 1) for 4D x.
    """
    batch_size = t.shape[0]
    out = schedule_params.to(t.device).gather(0, t.long()) # Gather values using timesteps as indices for eah sample in the batch
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))) # Reshape for broadcasting


# %% Debugging
if __name__ == "__main__":
    from F_schedulers import *
    timesteps = 1000
    betas = cosine_beta_schedule(timesteps)
    diffusion_params = get_diffusion_parameters(betas)

    print("Betas:", diffusion_params['betas'])
    print("Alphas cumulative product:", diffusion_params['alphas_cumprod'])
    print("Posterior variance:", diffusion_params['posterior_variance'])

    t = torch.tensor([0, 100, 500])  # Example timesteps
    x_0 = torch.randn(3, 1, 64, 64)  # Example data shape (B, C, H, W)
    
    sqrt_alphas_cumprod_t = extract(diffusion_params['sqrt_alphas_cumprod'], t, x_0.shape)
    print("Extracted sqrt alphas cumprod at t:", sqrt_alphas_cumprod_t)
# %% Debugging
if __name__ == "__main__":
    from F_schedulers import *
    from PIL import Image
    import requests
    import numpy as np
    from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
    import matplotlib.pyplot as plt

    timesteps = 500
    betas = cosine_beta_schedule(timesteps)
    diffusion_params = get_diffusion_parameters(betas)

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
    image = image.resize((64, 64))  # Resize to 64x64   

    image_size = 128
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
        Lambda(lambda t: (t * 2) - 1),
        
    ])

    x_start = transform(image).unsqueeze(0)
    x_start.shape

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(diffusion_params['sqrt_alphas_cumprod'], t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(x_start, t):
        # add noise
        x_noisy = q_sample(x_start, t=t)

        # turn back into PIL image
        noisy_image = reverse_transform(x_noisy.squeeze())

        return noisy_image

    torch.manual_seed(0)

    # source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [image] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title='Original image')
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()
        plt.show()

    plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])
