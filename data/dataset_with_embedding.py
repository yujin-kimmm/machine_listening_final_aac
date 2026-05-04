import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from data.clotho_captioning_dataset import ClothoCaptioningDataset
from encoders.convnext_encoder import ConvNeXtEncoder


class ClothoWithEmbeddings(Dataset):
    def __init__(self, clotho_dataset, convnext_encoder, beats_encoder=None):
        self.dataset = clotho_dataset
        self.convnext = convnext_encoder
        self.beats = beats_encoder  # optional until teammate adds BEATs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        audio = sample["encoder_input"]

        # resample from 16kHz (data loader) to 32kHz (ConvNeXt)
        audio_tensor = torch.tensor(audio).float().unsqueeze(0)
        audio_32k = torchaudio.functional.resample(audio_tensor, 16000, 32000)

        # get ConvNeXt frame embeddings [1, T, 768]
        convnext_emb = self.convnext.get_embeddings(audio_32k)

        if self.beats is not None:
            beats_emb = self.beats.get_embeddings(audio_32k)
            # concatenate along sequence dimension for fusion
            embeddings = torch.cat([convnext_emb, beats_emb], dim=1)
        else:
            embeddings = convnext_emb

        return {
            "embeddings": embeddings,
            "labels": sample["labels"],
            "sample_name": sample["sample_name"]
        }


if __name__ == "__main__":
    from data.clotho_captioning_dataset import ClothoCaptioningDataset
    from encoders.convnext_encoder import ConvNeXtEncoder

    clotho = ClothoCaptioningDataset(
        "/Users/ivanarasch/Documents/GitHub/clotho/validation",
        "gpt2",
        "/Users/ivanarasch/Documents/GitHub/clotho/clotho_captions_validation.csv",
    )

    encoder = ConvNeXtEncoder("checkpoints/convnext_tiny_465mAP_BL_AC_70kit.pth")

    dataset = ClothoWithEmbeddings(clotho, encoder)

    sample = dataset[0]
    print("Sample name:", sample["sample_name"])
    print("Embeddings shape:", sample["embeddings"].shape)
    print("Caption tokens:", sample["labels"])