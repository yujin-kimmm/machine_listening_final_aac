import os
import glob
from typing import List, Dict, Any, Iterable, Optional

import torch
import pandas as pd
from torch.utils.data import Dataset, Subset


CLOTHO_SPLITS = ("development", "validation", "evaluation")
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def normalize_clotho_splits(
    clotho_splits: Optional[Iterable[str] | str],
) -> List[str]:
    if clotho_splits is None:
        return list(CLOTHO_SPLITS)

    if isinstance(clotho_splits, str):
        split_list = [clotho_splits]
    else:
        split_list = list(clotho_splits)

    normalized_splits = []
    for split_name in split_list:
        normalized_name = str(split_name).strip().lower()
        if normalized_name not in CLOTHO_SPLITS:
            raise ValueError(
                f"Unsupported Clotho split: {split_name}. "
                f"Expected one of {CLOTHO_SPLITS}."
            )
        normalized_splits.append(normalized_name)

    if len(normalized_splits) == 0:
        raise ValueError("At least one Clotho split must be provided.")

    return normalized_splits


def get_clotho_audio_dir(audio_root_dir: str, split: str) -> str:
    normalized_split = normalize_clotho_splits(split)[0]
    return os.path.join(audio_root_dir, normalized_split)


def list_audio_files_in_dir(audio_dir: str) -> List[str]:
    if not os.path.isdir(audio_dir):
        raise NotADirectoryError(f"Audio directory not found: {audio_dir}")

    audio_paths = []
    for file_name in sorted(os.listdir(audio_dir)):
        file_path = os.path.join(audio_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
            continue
        audio_paths.append(file_path)

    if len(audio_paths) == 0:
        raise RuntimeError(f"No audio files found in: {audio_dir}")

    return audio_paths


def list_clotho_audio_files(audio_root_dir: str, split: str) -> List[str]:
    return list_audio_files_in_dir(get_clotho_audio_dir(audio_root_dir, split))


class ClothoAudioCaptionDataset(Dataset):
    def __init__(
        self,
        caption_dir: str,
        audio_root_dir: str,
        tokenizer,
        prompt: str = "Describe this audio:",
        max_length: int = 128,
        caption_pattern: str = "clotho_captions_*.csv",
        check_files: bool = True,
        clotho_splits: Optional[Iterable[str] | str] = None,
    ):
        """
        Training dataset for Clotho audio captioning.

        This dataset keeps the original Clotho split information and can
        load one split or multiple splits explicitly.

        Expected caption files:
            caption_dir/clotho_captions_development.csv
            caption_dir/clotho_captions_validation.csv
            caption_dir/clotho_captions_evaluation.csv

        Expected audio folders:
            audio_root_dir/development/<file_name>
            audio_root_dir/validation/<file_name>
            audio_root_dir/evaluation/<file_name>

        Expected CSV format:
            file_name, caption_1, caption_2, caption_3, caption_4, caption_5

        Each audio has 5 captions.
        This dataset expands each audio row into 5 training samples.
        """

        self.caption_dir = caption_dir
        self.audio_root_dir = audio_root_dir
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.caption_pattern = caption_pattern
        self.check_files = check_files
        self.clotho_splits = normalize_clotho_splits(clotho_splits)

        self.caption_columns = [
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
        ]

        self.caption_csvs = sorted(
            glob.glob(os.path.join(caption_dir, caption_pattern))
        )

        if len(self.caption_csvs) == 0:
            raise FileNotFoundError(
                f"No caption CSV files found in: {caption_dir} "
                f"with pattern: {caption_pattern}"
            )

        self.caption_csvs = [
            caption_csv
            for caption_csv in self.caption_csvs
            if self._infer_split_from_csv(caption_csv) in self.clotho_splits
        ]

        if len(self.caption_csvs) == 0:
            raise FileNotFoundError(
                f"No caption CSV files matched requested Clotho splits "
                f"{self.clotho_splits} in: {caption_dir}"
            )

        encoded_prompt = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        self.prompt_length = encoded_prompt["input_ids"].shape[1]

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples were created. Check caption CSV files and column names."
            )

    def _infer_split_from_csv(self, caption_csv: str) -> str:
        """
        Infer Clotho split from caption CSV filename.

        Examples:
            clotho_captions_development.csv -> development
            clotho_captions_validation.csv  -> validation
            clotho_captions_evaluation.csv  -> evaluation
        """

        csv_name = os.path.basename(caption_csv).lower()

        if "development" in csv_name:
            return "development"

        if "validation" in csv_name:
            return "validation"

        if "evaluation" in csv_name:
            return "evaluation"

        raise ValueError(
            f"Cannot infer split from caption CSV filename: {caption_csv}"
        )

    def _audio_id_from_file_name(self, file_name: str) -> str:
        """
        Example:
            'Distorted AM Radio noise.wav'
            -> 'Distorted AM Radio noise'
        """
        return os.path.splitext(file_name)[0]

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples = []

        for caption_csv in self.caption_csvs:
            split = self._infer_split_from_csv(caption_csv)

            df = pd.read_csv(caption_csv)

            required_columns = ["file_name"] + self.caption_columns
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"Missing column '{col}' in {caption_csv}"
                    )

            for _, row in df.iterrows():
                file_name = row["file_name"]
                audio_id = self._audio_id_from_file_name(file_name)

                audio_path = os.path.join(
                    self.audio_root_dir,
                    split,
                    file_name,
                )

                if self.check_files and not os.path.exists(audio_path):
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path}"
                    )

                for caption_col in self.caption_columns:
                    caption = row[caption_col]

                    if pd.isna(caption):
                        continue

                    samples.append(
                        {
                            "clotho_split_source": split,
                            "audio_id": audio_id,
                            "file_name": file_name,
                            "audio_path": audio_path,
                            "caption": str(caption),
                        }
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        caption = sample["caption"]
        full_text = self.prompt + " " + caption + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        # Ignore text padding positions in loss.
        labels[attention_mask == 0] = -100

        # Ignore prompt positions in loss.
        labels[: self.prompt_length] = -100

        return {
            "clotho_split_source": sample["clotho_split_source"],
            "audio_id": sample["audio_id"],
            "file_name": sample["file_name"],
            "audio_path": sample["audio_path"],
            "caption": caption,

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": self.prompt_length,
        }



def count_unique_audio_ids(dataset):
    if isinstance(dataset, Subset):
        audio_ids = {
            dataset.dataset.samples[idx]["audio_id"]
            for idx in dataset.indices
        }
        return len(audio_ids)

    audio_ids = {sample["audio_id"] for sample in dataset.samples}
    return len(audio_ids)


def count_clotho_split_sources(dataset):
    if isinstance(dataset, Subset):
        split_sources = [
            dataset.dataset.samples[idx]["clotho_split_source"]
            for idx in dataset.indices
        ]
    else:
        split_sources = [sample["clotho_split_source"] for sample in dataset.samples]

    split_counts = {}
    for split_source in split_sources:
        split_counts[split_source] = split_counts.get(split_source, 0) + 1
    return split_counts
