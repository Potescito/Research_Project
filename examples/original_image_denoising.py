"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
# %%
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from src.VideoProcessor import VideoProcessor
from src.anisotropic import anisotropic_diffusion
from src.bilateral import bilateral_filter
from src.GSA import GSA
from src.psnlm import psnlm
from src.utils import imshow
from src.metrics import NRMSE, PSNR, SSIM
# from joint_bilateral_filter_layer import JointBilateralFilter3d
# import time

if __name__ == "__main__":
    dataset_path = r"../data/dataset_2drt_video_only"
    nSubs = [f"sub{str(i).zfill(3)}" for i in range(1, 2)]
    vp = VideoProcessor(dataset_path, nSubs=nSubs, norm=True)
    
    dataset = vp.extract_frames(target="vcv")
    n = list(dataset.keys())
    
    denoised_ds = {}
