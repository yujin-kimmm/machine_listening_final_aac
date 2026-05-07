"""
Fusion Encoder modules for DCASE2024 CMU2 architecture.

Implements Early Feature Fusion of BEATs + ConvNeXt with Multi-Layer
Aggregation, followed by a Conformer post-encoder.

Architecture (Early Feature Fusion):
    BEATs (13 layers) → MultiLayerAggregation → [B, T_b, 768]
    ConvNeXt (frames)  → interpolate to T_b    → [B, T_b, 768]
    concat along feature dim                   → [B, T_b, 1536]
    Linear projection                          → [B, T_b, 768]
    Conformer post-encoder                     → [B, T_b, 768]
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)

# Add Model/ dir to path for BEATs/ConvNeXt imports
_model_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

from BEATs import BEATs, BEATsConfig
from convnext_encoder import ConvNeXtEncoder


def load_audio_batch(audio_paths, target_sr=16000):
    """Load a list of audio file paths and pad into a batch tensor.
    
    Use this in the training loop to convert the data loader's audio_path
    strings into the waveform tensor that AudioToConformer expects.
    
    Args:
        audio_paths: list of B file path strings
        target_sr: target sample rate (default 16kHz for BEATs)
    Returns:
        waveforms: [B, max_samples] — zero-padded batch tensor
        wav_lengths: [B] — actual length of each waveform in samples
    """
    waveforms = []
    for path in audio_paths:
        wav, sr = torchaudio.load(path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:  # stereo → mono
            wav = wav.mean(dim=0, keepdim=True)
        waveforms.append(wav.squeeze(0))  # [num_samples]
    
    # Pad to longest and stack
    lengths = torch.tensor([w.size(0) for w in waveforms])
    max_len = lengths.max().item()
    padded = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.size(0)] = w
    
    return padded, lengths


class MultiLayerAggregation(nn.Module):
    """Aggregate all BEATs encoder layer outputs via concatenate-then-compress.
    
    From the paper: concatenate all layer outputs along the feature dimension,
    then compress via LayerNorm → FC → GELU → FC.
    
    Args:
        num_layers: Number of layer outputs (13 for BEATs: 1 input + 12 layers)
        hidden_dim: Hidden dimension of each layer output (768 for BEATs)
        output_dim: Output dimension after compression (768)
    """
    def __init__(self, num_layers=13, hidden_dim=768, output_dim=768):
        super().__init__()
        concat_dim = num_layers * hidden_dim  # 13 * 768 = 9984
        self.layer_norm = nn.LayerNorm(concat_dim)
        self.fc1 = nn.Linear(concat_dim, output_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: list of tensors, each [B, T, hidden_dim]
                           (13 outputs from BEATs: input embed + 12 layers)
        Returns:
            aggregated: [B, T, output_dim]
        """
        # Concatenate along feature dimension: [B, T, num_layers * hidden_dim]
        x = torch.cat(layer_outputs, dim=-1)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class EarlyFeatureFusion(nn.Module):
    """Early fusion by concatenation along the feature dimension.
    
    From the paper: interpolate ConvNeXt features to match BEATs sequence 
    length, then concatenate along the feature dimension, then project
    back to the target dimension.
    
    Args:
        beats_dim: Feature dimension of BEATs output (768)
        convnext_dim: Feature dimension of ConvNeXt output (768)
        output_dim: Output dimension after projection (768)
    """
    def __init__(self, beats_dim=768, convnext_dim=768, output_dim=768):
        super().__init__()
        self.projection = nn.Linear(beats_dim + convnext_dim, output_dim)

    def forward(self, beats_features, convnext_features):
        """
        Args:
            beats_features: [B, T_b, beats_dim] — aggregated BEATs output
            convnext_features: [B, T_c, convnext_dim] — ConvNeXt frame output
        Returns:
            fused: [B, T_b, output_dim]
        """
        T_b = beats_features.size(1)
        T_c = convnext_features.size(1)

        if T_c != T_b:
            # Interpolate ConvNeXt features to match BEATs sequence length
            # [B, T_c, D] → [B, D, T_c] → interpolate → [B, D, T_b] → [B, T_b, D]
            convnext_features = convnext_features.transpose(1, 2)
            convnext_features = F.interpolate(
                convnext_features, size=T_b, mode='linear', align_corners=False
            )
            convnext_features = convnext_features.transpose(1, 2)

        # Concatenate along feature dimension: [B, T_b, beats_dim + convnext_dim]
        fused = torch.cat([beats_features, convnext_features], dim=-1)
        # Project back: [B, T_b, output_dim]
        fused = self.projection(fused)
        return fused


class FusionEncoderConformer(nn.Module):
    """Full fusion encoder pipeline: MultiLayerAgg + EarlyFeatureFusion + Conformer.
    
    This module takes frozen encoder outputs (BEATs layer outputs + ConvNeXt 
    frame embeddings) and processes them through trainable fusion and conformer 
    layers. It handles padding and masking internally.
    
    Usage (from data loader returning variable-length lists):
        model = FusionEncoderConformer()
        # beats_layers_list: list of B items, each is list of 13 tensors [T_i, 768]
        # convnext_list: list of B tensors, each [T_c_i, 768]
        output, mask = model(beats_layers_list, convnext_list)
        # output: [B, T_max, 768], mask: [B, T_max]
    
    Usage (from pre-padded batch tensors):
        # beats_layers: [B, 13, T_max, 768] (already padded)
        # convnext_frames: [B, T_c_max, 768] (already padded)
        # attention_mask: [B, T_max] (True=real, False=padding)
        output, mask = model(beats_layers, convnext_frames, attention_mask=mask)
    
    Args:
        conformer_config: Wav2Vec2ConformerConfig for the post-encoder
        num_beats_layers: Number of BEATs layer outputs (default: 13)
        beats_dim: BEATs hidden dimension (default: 768)
        convnext_dim: ConvNeXt hidden dimension (default: 768)
        output_dim: Output dimension (default: 768)
    """
    def __init__(
        self,
        conformer_config=None,
        conformer_config_path=None,
        num_beats_layers=13,
        beats_dim=768,
        convnext_dim=768,
        output_dim=768,
    ):
        super().__init__()
        self.num_beats_layers = num_beats_layers

        # Multi-layer aggregation for BEATs
        self.multi_layer_agg = MultiLayerAggregation(
            num_layers=num_beats_layers,
            hidden_dim=beats_dim,
            output_dim=output_dim,
        )

        # Early feature fusion (BEATs + ConvNeXt along feature dim)
        self.fusion = EarlyFeatureFusion(
            beats_dim=output_dim,
            convnext_dim=convnext_dim,
            output_dim=output_dim,
        )

        # Conformer post-encoder: load from config file, direct config, or defaults
        if conformer_config is None and conformer_config_path is not None:
            with open(conformer_config_path, "r") as f:
                conformer_config = Wav2Vec2ConformerConfig(**json.load(f))
        elif conformer_config is None:
            # Default: look for config/conformer_config.json relative to project root
            default_path = os.path.join(_model_dir, "..", "config", "conformer_config.json")
            if os.path.exists(default_path):
                with open(default_path, "r") as f:
                    conformer_config = Wav2Vec2ConformerConfig(**json.load(f))
            else:
                conformer_config = Wav2Vec2ConformerConfig(
                    hidden_size=output_dim,
                    num_hidden_layers=2,
                    num_attention_heads=12,
                    intermediate_size=1536,
                    hidden_act="swish",
                    conformer_conv_dropout=0.1,
                    attention_dropout=0.1,
                    hidden_dropout=0.1,
                    position_embeddings_type="relative",
                )

        self.conformer = Wav2Vec2ConformerEncoder(conformer_config)
        self.output_dim = output_dim

    @staticmethod
    def pad_and_create_mask(tensor_list):
        """Pad a list of variable-length tensors and create an attention mask.
        
        Args:
            tensor_list: list of B tensors with shape [..., T_i, D]
                         where T_i varies per sample
        Returns:
            padded: tensor [..., T_max, D] (zero-padded)
            mask: [B, T_max] boolean mask (True = real, False = padding)
        """
        lengths = [t.size(-2) for t in tensor_list]
        T_max = max(lengths)
        B = len(tensor_list)
        D = tensor_list[0].size(-1)
        
        # Determine full shape: could be [T, D] or [13, T, D]
        prefix_shape = tensor_list[0].shape[:-2]
        padded = torch.zeros(B, *prefix_shape, T_max, D,
                            dtype=tensor_list[0].dtype,
                            device=tensor_list[0].device)
        mask = torch.zeros(B, T_max, dtype=torch.bool,
                          device=tensor_list[0].device)
        
        for i, (t, length) in enumerate(zip(tensor_list, lengths)):
            padded[i, ..., :length, :] = t
            mask[i, :length] = True
        
        return padded, mask

    def forward(self, beats_layers, convnext_frames, attention_mask=None):
        """
        Accepts EITHER:
          (a) Pre-padded batch tensors (when collator handles padding):
              beats_layers: [B, 13, T_b, 768] or list of 13 tensors each [B, T_b, 768]
              convnext_frames: [B, T_c, 768]
              attention_mask: [B, T_b] (True = real, False = padding)
              
          (b) Variable-length lists (when this module handles padding):
              beats_layers: list of B items, each [13, T_i, 768]
              convnext_frames: list of B tensors, each [T_c_i, 768]
              attention_mask: None (will be created internally)
              
        Returns:
            output: [B, T_b, 768] — fused encoder output
            attention_mask: [B, T_b] — boolean mask (True = real token)
        """
        # --- Detect input format and normalize to list of 13 × [B, T, 768] ---
        if isinstance(beats_layers, torch.Tensor) and beats_layers.dim() == 4:
            # (c) Stacked tensor: [B, 13, T_b, 768]
            beats_layers_split = [beats_layers[:, i, :, :] for i in range(self.num_beats_layers)]
        elif isinstance(beats_layers, list) and len(beats_layers) == self.num_beats_layers and beats_layers[0].dim() == 3:
            # (a) List of 13 tensors, each [B, T_b, 768] — already batched by collator
            beats_layers_split = beats_layers
        elif isinstance(beats_layers, list):
            # (b) List of B per-sample tensors, each [13, T_i, 768] — variable length
            lengths = [t.size(1) for t in beats_layers]
            T_max = max(lengths)
            B = len(beats_layers)
            D = beats_layers[0].size(-1)
            
            beats_padded = torch.zeros(B, self.num_beats_layers, T_max, D,
                                       dtype=beats_layers[0].dtype,
                                       device=beats_layers[0].device)
            attention_mask = torch.zeros(B, T_max, dtype=torch.bool,
                                        device=beats_layers[0].device)
            for i, (t, length) in enumerate(zip(beats_layers, lengths)):
                beats_padded[i, :, :length, :] = t
                attention_mask[i, :length] = True
            
            beats_layers_split = [beats_padded[:, i, :, :] for i in range(self.num_beats_layers)]

        # Handle ConvNeXt variable-length list
        if isinstance(convnext_frames, list):
            convnext_frames, _ = self.pad_and_create_mask(convnext_frames)
            # convnext_frames: [B, T_c_max, 768]

        # Step 1: Aggregate BEATs layers → [B, T_b, 768]
        beats_agg = self.multi_layer_agg(beats_layers_split)

        # Step 2: Fuse with ConvNeXt → [B, T_b, 768]
        fused = self.fusion(beats_agg, convnext_frames)

        # Step 3: Conformer post-encoder → [B, T_b, 768]
        conformer_out = self.conformer(
            fused,
            attention_mask=attention_mask,
        ).last_hidden_state

        return conformer_out, attention_mask

class AudioToConformer(nn.Module):
    """End-to-end module: raw audio → conformer output.
    
    Loads frozen BEATs + ConvNeXt encoders and wraps FusionEncoderConformer.
    The data loader just passes raw waveforms — this module handles everything:
    frozen encoder inference, padding, masking, fusion, and conformer.
    
    Usage:
        model = AudioToConformer(
            beats_ckpt="pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
            convnext_ckpt="pretrained_weights/convnext_tiny_465mAP_BL_AC_70kit.pth",
        )
        output, mask = model(waveforms_16k, wav_lengths)
        # output: [B, T, 768], mask: [B, T]
    
    Args:
        beats_ckpt: path to BEATs checkpoint
        convnext_ckpt: path to ConvNeXt checkpoint
        conformer_config: optional Wav2Vec2ConformerConfig
    """
    def __init__(self, beats_ckpt, convnext_ckpt, conformer_config=None):
        super().__init__()
        
        # --- Frozen BEATs ---
        checkpoint = torch.load(beats_ckpt, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(checkpoint["model"])
        self.beats.eval()
        for p in self.beats.parameters():
            p.requires_grad_(False)
        
        # --- Frozen ConvNeXt ---
        self.convnext = ConvNeXtEncoder(convnext_ckpt)
        self.convnext.model.eval()
        for p in self.convnext.model.parameters():
            p.requires_grad_(False)
        
        # --- Trainable fusion + conformer ---
        self.fusion_conformer = FusionEncoderConformer(
            conformer_config=conformer_config,
        )
    
    @property
    def output_dim(self):
        return self.fusion_conformer.output_dim
    
    def _run_beats(self, waveform_16k, padding_mask=None):
        """Run frozen BEATs, return all 13 layer outputs.
        
        Args:
            waveform_16k: [B, num_samples] at 16kHz
            padding_mask: [B, num_samples] True = padding (optional)
        Returns:
            list of 13 tensors, each [B, T_b, 768]
        """
        with torch.no_grad():
            _, layer_results, _ = self.beats.extract_features(
                waveform_16k, padding_mask=padding_mask
            )
        return layer_results  # list of 13 × [B, T_b, 768]
    
    def _run_convnext(self, waveform_16k):
        """Run frozen ConvNeXt, return frame embeddings.
        
        Args:
            waveform_16k: [B, num_samples] at 16kHz
        Returns:
            [B, T_c, 768] frame embeddings
        """
        # Resample 16kHz → 32kHz
        waveform_32k = torchaudio.functional.resample(waveform_16k, 16000, 32000)
        
        with torch.no_grad():
            frame_emb = self.convnext.model.forward_frame_embeddings(waveform_32k)
            B, C, T, F_dim = frame_emb.shape
            # [B, C, T, F] → [B, T*F, C]
            frame_emb = frame_emb.permute(0, 2, 3, 1).reshape(B, T * F_dim, C)
        
        return frame_emb
    
    def forward(self, waveforms_16k, wav_lengths=None):
        """
        Args:
            waveforms_16k: [B, max_samples] — batch of 16kHz waveforms (zero-padded)
            wav_lengths: [B] — actual length of each waveform in samples (optional).
                         If None, assumes no padding (all samples same length).
        Returns:
            output: [B, T, 768] — conformer encoder output
            attention_mask: [B, T] — boolean mask (True = real, False = padding)
        """
        device = waveforms_16k.device
        B = waveforms_16k.size(0)
        
        # Create waveform-level padding mask for BEATs (True = padding)
        if wav_lengths is not None:
            max_len = waveforms_16k.size(1)
            wav_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= wav_lengths.unsqueeze(1)
        else:
            wav_padding_mask = None
        
        # Run frozen encoders
        beats_layers = self._run_beats(waveforms_16k, padding_mask=wav_padding_mask)
        convnext_frames = self._run_convnext(waveforms_16k)
        
        # Create attention mask at the frame level (True = real)
        # BEATs T_b is determined by the patch embedding
        T_b = beats_layers[0].size(1)
        if wav_lengths is not None:
            # Approximate: frame length is proportional to waveform length
            max_wav = waveforms_16k.size(1)
            frame_lengths = (wav_lengths.float() / max_wav * T_b).long().clamp(min=1, max=T_b)
            attention_mask = torch.arange(T_b, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        else:
            attention_mask = torch.ones(B, T_b, dtype=torch.bool, device=device)
        
        # Run trainable fusion + conformer
        output, attention_mask = self.fusion_conformer(
            beats_layers, convnext_frames, attention_mask=attention_mask
        )
        
        return output, attention_mask


if __name__ == "__main__":
    import argparse
    
    print("=" * 60)
    print("Test 1: FusionEncoderConformer with pre-padded batch")
    print("=" * 60)
    batch_size = 2
    T_beats = 62
    T_conv = 217
    hidden_dim = 768
    num_layers = 13

    beats_layers = [torch.randn(batch_size, T_beats, hidden_dim) for _ in range(num_layers)]
    convnext_frames = torch.randn(batch_size, T_conv, hidden_dim)
    mask = torch.ones(batch_size, T_beats, dtype=torch.bool)

    model = FusionEncoderConformer()
    output, out_mask = model(beats_layers, convnext_frames, attention_mask=mask)
    print(f"Output: {output.shape}, Mask: {out_mask.shape}")
    assert output.shape == (batch_size, T_beats, hidden_dim)
    print("✅ Passed!\n")

    print("=" * 60)
    print("Test 2: FusionEncoderConformer with variable-length list")
    print("=" * 60)
    T1, T2 = 50, 70
    Tc1, Tc2 = 180, 250
    beats_list = [
        torch.randn(num_layers, T1, hidden_dim),
        torch.randn(num_layers, T2, hidden_dim),
    ]
    conv_list = [
        torch.randn(Tc1, hidden_dim),
        torch.randn(Tc2, hidden_dim),
    ]

    output, out_mask = model(beats_list, conv_list)
    print(f"Output: {output.shape}, Mask: {out_mask.shape}")
    assert output.shape == (2, T2, hidden_dim)
    assert out_mask[0, T1-1] == True and out_mask[0, T1] == False
    print("✅ Passed!\n")

    # Test 3: AudioToConformer (only if checkpoints exist)
    beats_path = os.path.join(_model_dir, "..", "pretrained_weights",
                              "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt")
    convnext_path = os.path.join(_model_dir, "..", "pretrained_weights",
                                 "convnext_tiny_465mAP_BL_AC_70kit.pth")
    
    if os.path.exists(beats_path) and os.path.exists(convnext_path):
        print("=" * 60)
        print("Test 3: AudioToConformer (raw waveform → conformer output)")
        print("=" * 60)
        
        full_model = AudioToConformer(beats_path, convnext_path)
        
        # Count parameters
        frozen = sum(p.numel() for p in full_model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
        print(f"Frozen params: {frozen:,}")
        print(f"Trainable params: {trainable:,}")
        
        # Fake 10s audio batch
        dummy_audio = torch.randn(2, 160000)  # 2 samples, 10s at 16kHz
        wav_lengths = torch.tensor([160000, 120000])  # second sample is shorter
        
        output, mask = full_model(dummy_audio, wav_lengths)
        print(f"Input: waveforms [2, 160000] at 16kHz")
        print(f"Output: {output.shape}, Mask: {mask.shape}")
        print(f"Mask sample 1: {mask[0].sum()} real frames")
        print(f"Mask sample 2: {mask[1].sum()} real frames")
        print("✅ Passed!\n")
    else:
        print("(Skipping AudioToConformer test — checkpoints not found)")

    print("🎉 All tests passed!")

