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
from torch.utils.data import Dataset

class VideoProcessor(Dataset):
    def __init__(self, root_dir, filter_keyword=None, norm=False, max_frames=None):
        """
        Args:
            root_dir (str): Path to the root directory containing subdirectories (e.g., sub001, sub002, ..., sub075).
            filter_keyword (str, optional): If provided, only load video files that include this keyword in the filename.
            norm (bool, optional): Normalize the frames. Defaults to False.
            max_frames (int, optional): Maximum number of frames to load per video.
        """
        self.root_dir = root_dir
        self.filter_keyword = filter_keyword
        self.norm = norm
        self.max_frames = max_frames

        self.video_files = []

        for subject in sorted(os.listdir(root_dir)):
            subject_path = os.path.join(root_dir, subject, "2drt", "video")
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith('.mp4') and (self.filter_keyword is None or self.filter_keyword in file):
                        self.video_files.append(os.path.join(subject_path, file))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
            if self.norm:
                frame = frame / frame.max() # Normalize each frame
            frames.append(frame)
            frame_count += 1
            if self.max_frames is not None and frame_count >= self.max_frames:
                break

        cap.release()
        frames = np.array(frames)  # Shape: (num_frames, height, width)
        return frames, video_path

# %% Debugging
if __name__ == "__main__":
    from VideoProcessor2 import VideoProcessor
    from torch.utils.data import DataLoader

    dataset = VideoProcessor(root_dir = r"../data/dataset_2drt_video_only", filter_keyword="vcv", max_frames=2500)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for frames, video_path in dataloader:
        print(frames.shape, video_path)
        break