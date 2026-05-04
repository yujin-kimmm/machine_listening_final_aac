import torch
import torchaudio
from audioset_convnext_inf.pytorch.convnext import ConvNeXt

class ConvNeXtEncoder:
    
    def __init__(self, checkpoint_path):
        self.model = ConvNeXt(in_chans=1, num_classes=527, use_torchaudio=False)
        # patch the hardcoded Conv2d to accept 1 channel instead of 3
        self.model.downsample_layers[0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4,4), stride=(4,4))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()
        self.sample_rate = 32000

    def load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

    def get_embeddings(self, audio_path):
        waveform = self.load_audio(audio_path)
        with torch.no_grad():
            scene_emb = self.model.forward_scene_embeddings(waveform)
            frame_emb = self.model.forward_frame_embeddings(waveform)
            B, C, T, F = frame_emb.shape
            frame_emb = frame_emb.permute(0, 2, 3, 1).reshape(B, T*F, C)
        return scene_emb, frame_emb


if __name__ == "__main__":
    encoder = ConvNeXtEncoder("checkpoints/convnext_tiny_465mAP_BL_AC_70kit.pth")
    scene, frames = encoder.get_embeddings("audio_samples/1-26222-A-10.wav")
    print("Scene embedding shape:", scene.shape)
    print("Frame embeddings shape:", frames.shape)