"""
Research Project WiSe 2024/25
- Author:   Julian Hernandez
- Email:    julian.hernandez.potes@fau.de
- Tutor:    Tomas Arias
- Email:    tomas.arias@fau.de
"""
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor  #, HubertModel, WavLMModel

class PretrainedAudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", freeze_encoder=True, output_dim=None, process=False):
        """
        Audio encoder using a pre-trained model from Hugging Face Transformers.

        Args:
            model_name (str): Name of the pre-trained model (e.g., "facebook/wav2vec2-base-960h").
            freeze_encoder (bool): If True, freezes the weights of the pre-trained model.
            output_dim (int, optional): If specified, adds a linear layer to project features to this dimension.
                                         Otherwise, uses the native output dimension of the pre-trained model.
            process (bool): If True, uses the processor to handle input normalization and memory transfers.
        """
        super().__init__()
        self.process = process
        self.model_name = model_name
        self.pretrained_model = Wav2Vec2Model.from_pretrained(model_name)
        if process:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        self.native_feature_dim = self.pretrained_model.config.hidden_size

        self.projection = None
        if output_dim is not None and output_dim != self.native_feature_dim:
            self.projection = nn.Linear(self.native_feature_dim, output_dim)
            self.final_output_dim = output_dim
            print(f"Initialized PretrainedAudioEncoder with {model_name}. Native dim: {self.native_feature_dim}, Projected to: {output_dim}, {'Pre-process' if process else 'No Pre-process'}")
        else:
            self.final_output_dim = self.native_feature_dim
            print(f"Initialized PretrainedAudioEncoder with {model_name}. Native dim: {self.native_feature_dim}, {'Pre-process' if process else 'No Pre-process'}")

    def forward(self, audio_waveforms):
        """
        Args:
            audio_waveforms (torch.Tensor): Batch of raw audio waveforms.
                                           Expected shape: [Batch, NumSamples (e.g., 16000)].
                                           The model handles feature normalization internally if using from_pretrained.
        Returns:
            torch.Tensor: Audio embeddings (sequence of hidden states).
                          Shape: [Batch, NumAudioFrames, FeatureDim (self.final_output_dim)].
        """
        # Wav2Vec2Model expects input_values (raw waveform)
        attention_mask = None
        if self.process:
            processed = self.processor(audio_waveforms.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            audio_waveforms = processed.input_values.to(audio_waveforms.device)
            attention_mask = processed.attention_mask.to(audio_waveforms.device) if 'attention_mask' in processed else None
        
        outputs = self.pretrained_model(input_values=audio_waveforms, attention_mask=attention_mask)
        sequence_features = outputs.last_hidden_state # Shape: [Batch, NumAudioFrames, NativeFeatureDim]

        if self.projection:
            sequence_features = self.projection(sequence_features)
            
        return sequence_features

    def get_output_dim(self):
        return self.final_output_dim


# %% Debugging
if __name__ == '__main__':
    import torch
    from F_encoders import *
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)  # For reproducibility
    dummy_audio = torch.randn(4, 16000).to(DEVICE) # Batch of 4, 1-second audio at 16kHz
    # dummy_audio = torch.nn.functional.normalize(dummy_audio, dim=0)  # Normalize to [-1, 1] range (results are not the same)
    print(dummy_audio.min(), dummy_audio.max())
    print("")

    # Option 1: Use native output dimension (e.g., 768 for wav2vec2-base)
    audio_enc_pretrained = PretrainedAudioEncoder(model_name="facebook/wav2vec2-base-960h", freeze_encoder=True, process=True).to(DEVICE)
    audio_features_seq = audio_enc_pretrained(dummy_audio)
    print("Shape of sequence features (native):", audio_features_seq.shape) # e.g., [4, 49, 768]
    print("Output dim (native):", audio_enc_pretrained.get_output_dim())
    print(audio_features_seq.min(), audio_features_seq.max())
    print("")

    # Option 2: Project to a custom dimension (e.g., 512 to match previous SimpleAudioEncoder if needed,
    # but it's often better to adapt the U-Net to the richer native dimension if possible)
    custom_dim = 512
    audio_enc_projected = PretrainedAudioEncoder(model_name="facebook/wav2vec2-base-960h", freeze_encoder=True, output_dim=custom_dim, process=True).to(DEVICE)
    audio_features_projected_seq = audio_enc_projected(dummy_audio)
    print("Shape of sequence features (projected):", audio_features_projected_seq.shape) # e.g., [4, 49, 512]
    print("Output dim (projected):", audio_enc_projected.get_output_dim())
    print(audio_features_projected_seq.min(), audio_features_projected_seq.max())
