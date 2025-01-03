"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import numpy as np

def save_ds(path: str, dataset: dict):
    np.savez(path, **dataset)

def load_ds(path: str):
    ds = np.load(path, allow_pickle=True)
    ds = {key: ds[key].item() if ds[key].ndim==0 else ds[key] for key in ds}
    return ds
