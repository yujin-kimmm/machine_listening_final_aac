import argparse
import hashlib
import math
import os
import sys

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import (
    ClothoAudioCaptionDataset,
    count_clotho_split_sources,
    count_unique_audio_ids,
)

from Model.decoder import AudioPrefixGPT2
from Model.fusion_encoder import AudioToConformer, load_audio_batch

parser = argparse.ArgumentParser()
parser.add_argument("--config-path", default="./config.yaml", help="Path to config yaml")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

caption_dir = config["caption_dir"]
audio_root_dir = config["audio_root_dir"]

batch_size = config["batch_size"]
num_workers = config["num_workers"]
epochs = config["epochs"]
early_stopping_patience = config["early_stopping_patience"]
lr = config["lr"]
weight_decay = config["weight_decay"]
random_seed = 42 

audio_dim = config["audio_dim"]
max_length = config["max_length"]
preview_num_samples = config["preview_num_samples"]
audio_sample_rate = config["audio_sample_rate"]
do_sample = config["do_sample"]
lora_r = config["lora_r"]
lora_alpha = config["lora_alpha"]
lora_dropout = config["lora_dropout"]

save_dir = config["save_dir"]
best_checkpoint_path = config["best_checkpoint_path"]
prompt_text = config["prompt_text"]
train_split = config.get("train_split", "development")
validation_split = config.get("validation_split", "validation")
use_wandb = config.get("use_wandb", True)
_embedding_length_cache = {}



def run_batch(model, batch, encoder, device, inspect_batch=False):
    # Step A. Text side inputs from dataset
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    prompt_length = batch["prompt_length"][0].item()

    # Step B. Placeholder for future fusion encoder output
    waveforms, lengths = load_audio_batch(batch["audio_path"])
    waveforms = waveforms.to(device)
    lengths = lengths.to(device)
    encoder_out, mask = encoder(waveforms, lengths)
    # encoder_out is [B, T, 768] and mask is [B, T]

    # Step C. Model handles projection to GPT-2 dim and prefix concatenation
    outputs = model(
        audio_embeddings=encoder_out,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_length=prompt_length,
        audio_attention_mask=mask,
    )

    if inspect_batch:
        caption_token_count = (labels != -100).sum().item()
        print("[inspect] batch structure")
        print(f"[inspect] encoder_outputs shape: {tuple(encoder_out.shape)}")
        print(f"[inspect] input_ids shape: {tuple(input_ids.shape)}")
        print(f"[inspect] attention_mask shape: {tuple(attention_mask.shape)}")
        print(f"[inspect] labels shape: {tuple(labels.shape)}")
        print(f"[inspect] prefix_length: {encoder_out.shape[1]}")
        print(f"[inspect] prompt_length: {prompt_length}")
        print(f"[inspect] caption tokens contributing to loss: {caption_token_count}")

    return outputs.loss


def collect_unique_preview_samples(val_loader, max_samples):
    unique_audio_ids = set()
    preview_samples = []

    for batch in val_loader:
        batch_size = len(batch["audio_id"])

        for idx in range(batch_size):
            audio_id = batch["audio_id"][idx]
            if audio_id in unique_audio_ids:
                continue

            unique_audio_ids.add(audio_id)
            preview_samples.append(
                {
                    "audio_id": audio_id,
                    "file_name": batch["file_name"][idx],
                    "audio_path": batch["audio_path"][idx],
                    "caption": batch["caption"][idx],
                }
            )

            if len(preview_samples) >= max_samples:
                return preview_samples

    return preview_samples


def save_checkpoint(model, encoder, optimizer, epoch_idx, best_val_loss, save_path):
    checkpoint = {
        "epoch": epoch_idx + 1,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint_if_available(model, encoder, optimizer, device, resume=False):
    last_ckpt_path = os.path.join(save_dir, "last.pt")

    if not resume:
        print("Resume mode: off")
        return 0, float("inf")

    if not os.path.exists(last_ckpt_path):
        print(f"Resume mode: on, but no checkpoint found at {last_ckpt_path}")
        return 0, float("inf")

    checkpoint = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"Resume mode: loaded {last_ckpt_path}")
    print(f"Resume start epoch: {start_epoch + 1}")
    print(f"Resume best val loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


def main():
    print("Main function start")
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_dir = os.path.dirname(best_checkpoint_path)
    if best_checkpoint_dir:
        os.makedirs(best_checkpoint_dir, exist_ok=True)
    resume = "--resume" in sys.argv

    wandb = None
    if use_wandb:
        import wandb

        wandb.init(
            project=config["project"],
            entity=config["entity"],
            name=config["name"],
            config=config,
            dir=config["wandb_dir"],
        )
        print("W&B logging: on")
    else:
        print("W&B logging: off")

    # ----------------
    # Build 
    # ----------------

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Build Dataset using the original Clotho split definitions.
    train_dataset = ClothoAudioCaptionDataset(
        caption_dir=caption_dir,
        audio_root_dir=audio_root_dir,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_length=max_length,
        check_files=True,
        clotho_splits=train_split,
    )
    val_dataset = ClothoAudioCaptionDataset(
        caption_dir=caption_dir,
        audio_root_dir=audio_root_dir,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_length=max_length,
        check_files=True,
        clotho_splits=validation_split,
    )

    print(
        f"Train split: {train_split} | "
        f"samples: {len(train_dataset)} | "
        f"unique audios: {count_unique_audio_ids(train_dataset)} | "
        f"source counts: {count_clotho_split_sources(train_dataset)}"
    )
    print(
        f"Validation split: {validation_split} | "
        f"samples: {len(val_dataset)} | "
        f"unique audios: {count_unique_audio_ids(val_dataset)} | "
        f"source counts: {count_clotho_split_sources(val_dataset)}"
    )

    # Build Dataloadesr
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Build Model
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
    model = model.to(device)
    encoder = encoder.to(device)

    optimizer = AdamW(
        (
            param
            for module in (model, encoder)
            for param in module.parameters()
            if param.requires_grad
        ),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    start_epoch, best_val_loss = load_checkpoint_if_available(
        model,
        encoder,
        optimizer,
        device,
        resume=resume,
    )
    epochs_without_improvement = 0
    # ----------------- 
    # Train 
    # ------------------
    for epoch_idx in range(start_epoch, epochs):
        print(f"\n========== Epoch {epoch_idx + 1}/{epochs} ==========")
        # print(" Train decoder with fake encoder outputs")
        model.train()
        encoder.train()
        train_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inspect_batch = (epoch_idx == 0 and batch_idx == 0)
            loss = run_batch(model, batch, encoder, device, inspect_batch=inspect_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = train_loss_sum / (batch_idx + 1)
                print(
                    f"[train] epoch {epoch_idx + 1} | "
                    f"batch {batch_idx + 1}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | "
                    f"avg_loss {avg_loss:.4f}"
                )

        train_loss = train_loss_sum / len(train_loader)
        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")

        #--------------- 
        # validation
        # -------------
        model.eval()
        encoder.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                loss = run_batch(model, batch, encoder, device)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": train_loss,
                    # "train_ppl": train_ppl,
                    "val_loss": val_loss,
                    # "val_ppl": val_ppl,
                }
            )

        print(
            f"[val] epoch {epoch_idx + 1} | "
            f"loss {val_loss:.4f} | "
            f"ppl {val_ppl:.4f}"
        )

        with torch.no_grad():
            preview_samples = collect_unique_preview_samples(
                val_loader,
                preview_num_samples,
            )
            sample_count = len(preview_samples)

            preview_audio_paths = [sample["audio_path"] for sample in preview_samples]
            waveforms, lengths = load_audio_batch(preview_audio_paths)
            waveforms = waveforms.to(device)
            lengths = lengths.to(device)
            test_encoder_outputs, audio_attention_mask = encoder(waveforms, lengths)

            encoded_prompt = tokenizer(
                [prompt_text] * sample_count,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            generated_ids = model.generate_caption(
                audio_embeddings=test_encoder_outputs,
                input_ids=encoded_prompt["input_ids"].to(device),
                attention_mask=encoded_prompt["attention_mask"].to(device),
                audio_attention_mask=audio_attention_mask,
                max_new_tokens=30,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        if use_wandb:
            preview_table = wandb.Table(
                columns=[
                    "file_name",
                    "audio",
                    "ground_truth_caption",
                    "predicted_caption",
                ]
            )

            for idx in range(sample_count):
                generated_text = tokenizer.decode(
                    generated_ids[idx],
                    skip_special_tokens=True,
                )

                preview_table.add_data(
                    preview_samples[idx]["file_name"],
                    wandb.Audio(
                        preview_samples[idx]["audio_path"],
                        sample_rate=audio_sample_rate,
                    ),
                    preview_samples[idx]["caption"],
                    generated_text,
                )

            wandb.log(
                {
                    "val_preview": preview_table,
                    f"val_preview_epoch_{epoch_idx + 1}": preview_table,
                }
            )

        print(
            f"[epoch {epoch_idx + 1}] "
            f"train_loss {train_loss:.4f} | "
            f"train_ppl {train_ppl:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_ppl {val_ppl:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                encoder,
                optimizer,
                epoch_idx,
                best_val_loss,
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[early_stopping] no improvement for "
                f"{epochs_without_improvement}/{early_stopping_patience} epoch(s)"
            )

        last_ckpt_path = os.path.join(save_dir, "last.pt")
        save_checkpoint(
            model,
            encoder,
            optimizer,
            epoch_idx,
            best_val_loss,
            last_ckpt_path,
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"[early_stopping] stop training at epoch {epoch_idx + 1} "
                f"because val_loss did not improve for "
                f"{early_stopping_patience} consecutive epochs"
            )
            break

    print("\nTraining finished.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
