import hashlib
import math
import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ClothoAudioCaptionDataset, count_clotho_split_sources, count_unique_audio_ids, split_clotho_dataset

from model import AudioPrefixGPT2


caption_dir = "/scratch/yk3281/dataset/clotho"
audio_root_dir = "/scratch/yk3281/dataset/clotho"

batch_size = 8
num_workers = 0
epochs = 5
lr = 1e-4
weight_decay = 0.01
val_ratio = 0.1
random_seed = 42

prefix_length = 10
audio_dim = 512
max_length = 128

save_dir = "outputs/simple_train"
prompt_text = "Describe this audio:"


def get_fake_encoder_outputs(audio_paths, device):
    batch_embeddings = []

    for audio_path in audio_paths:
        path_hash = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
        seed = int(path_hash[:8], 16)

        generator = torch.Generator()
        generator.manual_seed(seed)

        audio_embedding = torch.randn(
            prefix_length,
            audio_dim,
            generator=generator,
        )
        batch_embeddings.append(audio_embedding)

    fake_encoder_outputs = torch.stack(batch_embeddings, dim=0)
    return fake_encoder_outputs.to(device)


def run_batch(model, batch, device, inspect_batch=False):
    # Step A. Text side inputs from dataset
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    prompt_length = batch["prompt_length"][0].item()

    # Step B. Placeholder for future fusion encoder output
    fake_encoder_outputs = get_fake_encoder_outputs(
        batch["audio_path"],
        device,
    )

    # Step C. Model handles projection to GPT-2 dim and prefix concatenation
    outputs = model(
        audio_embeddings=fake_encoder_outputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_length=prompt_length,
    )

    if inspect_batch:
        caption_token_count = (labels != -100).sum().item()
        print("[inspect] batch structure")
        print(f"[inspect] fake_encoder_outputs shape: {tuple(fake_encoder_outputs.shape)}")
        print(f"[inspect] input_ids shape: {tuple(input_ids.shape)}")
        print(f"[inspect] attention_mask shape: {tuple(attention_mask.shape)}")
        print(f"[inspect] labels shape: {tuple(labels.shape)}")
        print(f"[inspect] prefix_length: {fake_encoder_outputs.shape[1]}")
        print(f"[inspect] prompt_length: {prompt_length}")
        print(f"[inspect] caption tokens contributing to loss: {caption_token_count}")

    return outputs.loss


def build_prompt_batch(tokenizer, batch, device):
    batch_size = len(batch["audio_path"])
    encoded_prompt = tokenizer(
        [prompt_text] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    return {
        "input_ids": encoded_prompt["input_ids"].to(device),
        "attention_mask": encoded_prompt["attention_mask"].to(device),
    }


def preview_generation(model, val_loader, tokenizer, device, num_samples=2):
    model.eval()

    with torch.no_grad():
        batch = next(iter(val_loader))
        sample_count = min(num_samples, len(batch["audio_path"]))

        fake_encoder_outputs = get_fake_encoder_outputs(
            batch["audio_path"][:sample_count],
            device,
        )
        prompt_batch = build_prompt_batch(tokenizer, batch, device)

        generated_ids = model.generate_caption(
            audio_embeddings=fake_encoder_outputs,
            input_ids=prompt_batch["input_ids"][:sample_count],
            attention_mask=prompt_batch["attention_mask"][:sample_count],
            max_new_tokens=30,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("[val] sample generations")
    for idx in range(sample_count):
        generated_text = tokenizer.decode(
            generated_ids[idx],
            skip_special_tokens=True,
        )
        print(f"sample {idx + 1} file: {batch['file_name'][idx]}")
        print(f"sample {idx + 1} target: {batch['caption'][idx]}")
        print(f"sample {idx + 1} generated: {generated_text}")


def save_checkpoint(model, optimizer, epoch_idx, best_val_loss, save_path):
    checkpoint = {
        "epoch": epoch_idx + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint_if_available(model, optimizer, device, resume=False):
    last_ckpt_path = os.path.join(save_dir, "last.pt")

    if not resume:
        print("Resume mode: off")
        return 0, float("inf")

    if not os.path.exists(last_ckpt_path):
        print(f"Resume mode: on, but no checkpoint found at {last_ckpt_path}")
        return 0, float("inf")

    checkpoint = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"Resume mode: loaded {last_ckpt_path}")
    print(f"Resume start epoch: {start_epoch + 1}")
    print(f"Resume best val loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)
    resume = "--resume" in sys.argv

    print(f"Using device: {device}")
    # ----------------
    # Build 
    # ----------------

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Build Dataset
    full_dataset = ClothoAudioCaptionDataset(
        caption_dir=caption_dir,
        audio_root_dir=audio_root_dir,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_length=max_length,
        check_files=True,
    )

    # Split dataset
    train_dataset, val_dataset = split_clotho_dataset(
        full_dataset,
        val_ratio=val_ratio,
        random_seed=random_seed,
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
    )
    model.gpt2.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(device)

    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    # print(f"Full dataset samples: {len(full_dataset)}")
    # print(f"Train samples: {len(train_dataset)}")
    # print(f"Val samples: {len(val_dataset)}")
    # print(f"Combined pool audio files: {count_unique_audio_ids(full_dataset)}")
    # print(f"Train audio files: {count_unique_audio_ids(train_dataset)}")
    # print(f"Val audio files: {count_unique_audio_ids(val_dataset)}")
    # print(f"Original Clotho split counts in combined pool: {count_clotho_split_sources(full_dataset)}")
    # print(f"Original Clotho split counts in train split: {count_clotho_split_sources(train_dataset)}")
    # print(f"Original Clotho split counts in val split: {count_clotho_split_sources(val_dataset)}")
    # print(f"Split rule: combine development/validation/evaluation, then split by audio_id with val_ratio={val_ratio}")
    # print("\nTrainable parameters:")
    # model.print_trainable_parameters()

    start_epoch, best_val_loss = load_checkpoint_if_available(
        model,
        optimizer,
        device,
        resume=resume,
    )

    for epoch_idx in range(start_epoch, epochs):
        print(f"\n========== Epoch {epoch_idx + 1}/{epochs} ==========")
        # print("Step 6. Train decoder with fake encoder outputs")
        model.train()
        train_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inspect_batch = (epoch_idx == 0 and batch_idx == 0)
            loss = run_batch(model, batch, device, inspect_batch=inspect_batch)

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
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                loss = run_batch(model, batch, device)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

        print(
            f"[val] epoch {epoch_idx + 1} | "
            f"loss {val_loss:.4f} | "
            f"ppl {val_ppl:.4f}"
        )

        print("Step 8. Preview validation generations")
        preview_generation(
            model,
            val_loader,
            tokenizer,
            device,
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
            best_ckpt_path = os.path.join(save_dir, "best.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch_idx,
                best_val_loss,
                best_ckpt_path,
            )

        last_ckpt_path = os.path.join(save_dir, "last.pt")
        save_checkpoint(
            model,
            optimizer,
            epoch_idx,
            best_val_loss,
            last_ckpt_path,
        )

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
