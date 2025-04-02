The training scheme processes each sliding‐window separately and uses the conditional U‑Net (with FiLM) together with an audio feature extractor. In this setup, for each batch the collate function of AVDataset returns:

-waveforms: of shape (B, num_windows, window_audio)
-video_windows: of shape (B, num_windows, window_video, 1, H, W)

Then, for each window in the batch:

Extract the i‑th audio window -> maybe ch dim is necessary.

Pass that window through the audio feature extractor (pre - wav2vec2) to produce a feature sequence, then avg over the temporal dimension to get a global condition vector of shape (B, cond_dim).

Extract the corresponding video window, then rearrange its dimensions so that its shape is (B, 1, window_video, H, W) because the conditional U‑Net expects video input with a channel dimension at pos 1.

Forward the video window together with the condition vector through the conditional U‑Net.

Compute the loss (I use an L1 loss against the input video window as target).
Then I accumulate the loss over windows and update the optimizer once per batch.

FP16 ? 