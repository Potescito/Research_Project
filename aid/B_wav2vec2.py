"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import os
import torch
import numpy as np

def extract_audio_features(dataloader, processor, model, device, output_dir):
    """
    Iterates over the dataloader to extract wav2vec2 features offline.
    The extracted features for each audio sample are saved as .npy files.
    
    Args:
        dataloader: DataLoader providing (waveform, audio_path) pairs.
        processor: Pretrained Wav2Vec2Processor.
        model: Pretrained Wav2Vec2Model.
        device: torch.device (e.g., "cuda" or "cpu").
        output_dir (str): Directory to save the extracted features.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    for waveforms, _, audio_paths, _ in dataloader:
        for i in range(waveforms.size(0)): # batch (this is a one time thing so lets use this as it is but pad!)
            with torch.no_grad():
                inputs = processor(waveforms[i].numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                features = model(**inputs).last_hidden_state  # shape: (1, seq_len, feature_dim)

                base_name = os.path.splitext(os.path.basename(audio_paths[i]))[0]
                feature_path = os.path.join(output_dir, f"{base_name}_features.npy")
                np.save(feature_path, features.cpu().numpy())
                print(f"Saved features for {audio_paths[i]} to {feature_path}")


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch
    from aid.B_wav2vec2 import extract_audio_features
    from torch.utils.data import DataLoader
    from src.AVDataset import AVDataset
    from src.transforms import TemporalWindowTransform, ContextualSamplingTransform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    audio_root = r"../data/audios_denoised_16khz"
    video_root = r"../data/dataset_2drt_video_only"
    keyword = "vcv"

    nSubst = [f"sub{str(i).zfill(3)}" for i in range(1, 51)]
    nSubsv = [f"sub{str(i).zfill(3)}" for i in range(51, 75)]
    
    temporal_transform = TemporalWindowTransform(window_size_sec=1, audio_sample_rate=16000, video_fps=83)
    contextual_transform = ContextualSamplingTransform(context_size=1, audio_sample_rate=16000, video_fps=83)

    train_dataset = AVDataset(audio_root, video_root, subs=nSubst, filter_keyword=keyword, transform=temporal_transform)
    val_dataset = AVDataset(audio_root, video_root, subs=nSubsv, filter_keyword=keyword, transform=temporal_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=AVDataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=AVDataset.collate)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    
    extract_audio_features(train_loader, processor, model, device, output_dir="features_w2v/train_1")