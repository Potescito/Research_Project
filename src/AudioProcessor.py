"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import os
from torch.utils.data import Dataset
import torchaudio

class AudioProcessor(Dataset):
    def __init__(self, root_dir, filter_keyword=None, target_sampling_rate=16000):
        """
        Args:
            root_dir (str): Path to the root directory containing subdirectories (e.g., sub001, sub002, ...).
            filter_keyword (str, optional): If provided, only load audio files that include this keyword in the filename.
            target_sampling_rate (int): The sampling rate to which audio files will be resampled.
        """
        self.root_dir = root_dir
        self.filter_keyword = filter_keyword
        self.target_sampling_rate = target_sampling_rate
        
        self.audio_files = [] # List to hold paths to all chosen audio files.
        
        for subject in sorted(os.listdir(root_dir)):
            subject_path = os.path.join(root_dir, subject)
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith('.wav') and (self.filter_keyword is None or self.filter_keyword in file):
                        self.audio_files.append(os.path.join(subject_path, file))
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(file_path)
        print(sr)
        # Resample if needed
        if sr != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sampling_rate)
            waveform = resampler(waveform)
        
        return waveform, file_path

# %% Debugging
if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    from AudioProcessor import AudioProcessor

    dataset = AudioProcessor(root_dir = r"../data/audios_denoised_16khz", filter_keyword="vcv")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for waveform, file_path in dataloader:
        print(waveform.shape, file_path)
        break
# %%
