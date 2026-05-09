import argparse
import csv
import os
import sys

import torch
import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.decoder import AudioPrefixGPT2
from Model.fusion_encoder import AudioToConformer, load_audio_batch


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    audio_path = config["test_audio_dir"]
    audio_dim = config["audio_dim"]
    prompt_text = config["prompt_text"]
    lora_r = config["lora_r"]
    lora_alpha = config["lora_alpha"]
    lora_dropout = config["lora_dropout"]
    do_sample = config["do_sample"]
    checkpoint_path = config["best_checkpoint_path"]
    output_csv_path = config["output_csv_path"]
    audio_paths = []

    if not os.path.isdir(audio_path):
        raise NotADirectoryError(f"Audio directory not found: {audio_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    for file_name in sorted(os.listdir(audio_path)):
        file_path = os.path.join(audio_path, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue
        audio_paths.append(file_path)

    if len(audio_paths) == 0:
        raise RuntimeError(f"No audio files found in: {audio_path}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AudioPrefixGPT2(
        audio_dim=audio_dim,
        freeze_gpt2=True,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    encoder = AudioToConformer(
        "pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
        "pretrained_weights/convnext_tiny_465mAP_BL_AC_70kit.pth",
    )

    model.gpt2.config.pad_token_id = tokenizer.eos_token_id

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    model = model.to(device)
    encoder = encoder.to(device)
    model.eval()
    encoder.eval()

    encoded_prompt = tokenizer(
        [prompt_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    print(f"Audio directory: {audio_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Prompt: {prompt_text}")
    print(f"Output csv: {output_csv_path}")

    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "caption_predicted"])
        csv_file.flush()

        for audio_path in audio_paths:
            waveforms, lengths = load_audio_batch([audio_path])
            waveforms = waveforms.to(device)
            lengths = lengths.to(device)

            with torch.no_grad():
                encoder_outputs, audio_attention_mask = encoder(waveforms, lengths)
                generated_ids = model.generate_caption(
                    audio_embeddings=encoder_outputs,
                    input_ids=encoded_prompt["input_ids"].to(device),
                    attention_mask=encoded_prompt["attention_mask"].to(device),
                    audio_attention_mask=audio_attention_mask,
                    max_new_tokens=30,
                    do_sample=do_sample,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            ).strip()

            writer.writerow([os.path.basename(audio_path), generated_text])
            csv_file.flush()
            print(f"{os.path.basename(audio_path)} -> {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="./config.yaml", help="Path to config yaml")
    main(parser.parse_args())
