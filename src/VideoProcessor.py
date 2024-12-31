"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoProcessor(object):
    def __init__(self, dataset_path: str, nSubs: list=["sub001"], norm=True):
        """
        Class to extract the frames from a dataset of videos.

        Args:
            dataset_path (str)      : Path to the parent folder of the dataset.
            nSubs (list, optional) : List of subjects to extract the frames. Defaults to ["sub001"].
            norm (bool, optional)   : Normalize the frames. Defaults to True.
        """
        self.norm = norm
        self.dataset_path = dataset_path
        self.video_files = self._list_video_files(nSubs)

    def _list_video_files(self, nSubs):
        video_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.mp4') and any([sub in file for sub in nSubs]):
                    video_files.append(os.path.join(root, file))
        return video_files

    def extract_frames(self, target: str="vcv"):
        """
        Args:
            target (str, optional): Target video alias. Defaults to "vcv".
                                    ["vcv", "btv", "topic", "grandfather", 
                                     "picture", "shibboleth", "northwind",
                                     "postures", "rainbow"]
        Returns:
            dict: {video_name: frames}: Set of frames extracted from the videos.
        """
        dataset = {} # {video_name: frames}
        video_files = [file for file in self.video_files if target in file]
        def _process_video(video_file):
            frames = []
            name = video_file.split("\\")[-1].split(".")[0]
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                    if self.norm:
                        frame = frame / frame.max()
                    frames.append(frame)
                else:
                    break
            cap.release()
            frames = np.stack(frames, axis=0)
            return name, frames
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_video, video_file) for video_file in video_files]
            for future in as_completed(futures):
                name, frames = future.result()
                dataset[name] = frames

        return dataset

    @staticmethod
    def noise(dataset: dict, type: str='speckle', mean: float=0.0, std: float=1.0):
        """
        Add noise to the frames. Static method.
        
        Args:
            dataset (dict): Set of frames to add noise. Expected shape of the keys: (Frames, Height, Width)
            type (str, optional): Noise distribution type. Defaults to 'speckle'.
                                  ["gaussian", "speckle", "rayleigh"]
            mean (float, optional): Mean of the noise. Defaults to 0.
            std (float, optional): Standard deviation of the noise. Defaults to 1.
        
        Returns:
            dict: {video_name: frames}: Set of frames with noise added.
        """
        n_dataset = {}
        match type:
            case 'gaussian':
                print("Note: Additive gaussian noise as limit case of the Rician Distribution, since complex-valued image/k-space is not used.")
                for name, frames in dataset.items():
                    noise = np.random.normal(mean, std, frames.shape)
                    n_frames = frames + noise
                    n_dataset[name] = np.clip(n_frames, 0, 1) # should I normalize instead? wouldnt that change the overall intensity of the frames?
            case 'speckle':
                for name, frames in dataset.items():
                    noise = np.random.normal(mean, std, frames.shape)
                    n_frames = frames + frames * noise
                    n_dataset[name] = np.clip(n_frames, 0, 1)
            
            case 'rayleigh':
                print("Note: Adding Rayleigh-distributed noise as limit case of Rician Distribution.")
                for name, frames in dataset.items():
                    noise = np.random.rayleigh(std, frames.shape)
                    n_frames = frames + noise
                    n_dataset[name] = np.clip(n_frames, 0, 1)  

            case _:
                raise ValueError("Invalid noise type. Choose between 'gaussian' or 'speckle'.")

        return n_dataset

# %% ==================================================================================
if __name__ == "__main__":
    from pathlib import Path
    from VideoProcessor import VideoProcessor
    dataset_path = Path("../data/dataset_2drt_video_only")
    processor = VideoProcessor(dataset_path, ["sub001", "sub005"])

    a = processor.extract_frames(target="vcv")
    # print(processor.video_files)