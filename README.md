# 2026 Spring NYU Machine Listening Final Project

Ivana Rasch Chinchilla, Mehmet Atilay Kucukoglu,  Yujin Kim

Task: DCASE Challenge 2023 task 6a - Automated Audio Captioning


## Overview

The current pipeline uses:

- BEATs and ConvNeXt audio backbones
- a Conformer-based fusion encoder
- GPT-2 as the language decoder
- LoRA adapters for efficient decoder fine-tuning


## Data and Checkpoints

Update `main/config.yaml` before running:

- `caption_dir`: path to Clotho caption files
- `audio_root_dir`: path to Clotho audio folders
- `save_dir`: checkpoint output directory
- `best_checkpoint_path`: checkpoint used for inference
- `output_csv_path`: inference result CSV path
- `use_wandb`: whether to log training runs with Weights & Biases
- Wandb settings: `project`, `entity`, `name`, `wandb_dir`

Pretrained audio encoder weights are expected under:

```text
main/pretrained_weights/
```

Pretrained audio encoder weights can be downloaded from [here](https://drive.google.com/drive/folders/1VI8NQi34Mxp4euX9N98g58RlJlEdORHA?usp=sharing)

## Training

Training should be run from the `main` directory.

```bash
cd main
python train.py --config-path config.yaml
```

To resume from `save_dir/last.pt`:

```bash
python train.py --config-path config.yaml --resume
```


## Inference

After training, generate captions with:

```bash
cd main
python inference.py --config-path config.yaml
```

The script loads `best_checkpoint_path` and writes predictions to
`output_csv_path`.

## Evaluation

The evaluation is done by the [evaluation repository](https://github.com/audio-captioning/caption-evaluation-tools) from DCASE Challenge.
