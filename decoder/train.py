import hashlib
import math
import os
import sys

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ClothoAudioCaptionDataset, count_clotho_split_sources, count_unique_audio_ids, split_clotho_dataset

from model import AudioPrefixGPT2


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

caption_dir = config["caption_dir"]
audio_root_dir = config["audio_root_dir"]

batch_size = config["batch_size"]
num_workers = config["num_workers"]
epochs = config["epochs"]
lr = config["lr"]
weight_decay = config["weight_decay"]
val_ratio = config["val_ratio"]
random_seed = 42 

target_prefix_length = config["target_prefix_length"]
prefix_length = config["prefix_length"] # for fake encoder
audio_dim = config["audio_dim"]
max_length = config["max_length"]
preview_num_samples = config["preview_num_samples"]
audio_sample_rate = config["audio_sample_rate"]

save_dir = config["save_dir"]
prompt_text = config["prompt_text"]
_embedding_length_cache = {}


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


def get_global_max_embedding_length(embedding_root_dir):
    if embedding_root_dir in _embedding_length_cache:
        return _embedding_length_cache[embedding_root_dir]

    max_sequence_length = 0

    for split_name in ["development", "validation", "evaluation"]:
        split_dir = os.path.join(embedding_root_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        for file_name in sorted(os.listdir(split_dir)):
            if not file_name.endswith(".pt"):
                continue

            embedding_path = os.path.join(split_dir, file_name)
            embedding = torch.load(embedding_path, map_location="cpu")

            if not isinstance(embedding, torch.Tensor):
                raise TypeError(f"Expected tensor embedding at {embedding_path}")

            if embedding.dim() == 3 and embedding.size(0) == 1:
                embedding = embedding.squeeze(0)

            if embedding.dim() == 1:
                current_length = 1
            elif embedding.dim() == 2:
                current_length = embedding.size(0)
            else:
                raise ValueError(
                    f"Expected 1D or 2D tensor at {embedding_path}, got shape {tuple(embedding.shape)}"
                )

            max_sequence_length = max(max_sequence_length, current_length)

    if max_sequence_length == 0:
        raise RuntimeError(
            f"No embedding files found under {embedding_root_dir}"
        )

    _embedding_length_cache[embedding_root_dir] = max_sequence_length
    return max_sequence_length


def get_test_encoder_outputs(audio_paths, device, target_prefix_length=target_prefix_length):
    embedding_root_dir = "/scratch/yk3281/repo/machine_listening_final_aac/decoder/test_emb"
    batch_embeddings = []
    audio_attention_masks = []
    max_sequence_length = 0

    for audio_path in audio_paths:
        split_name = os.path.basename(os.path.dirname(audio_path))
        audio_file_name = os.path.basename(audio_path)
        embedding_path = os.path.join(
            embedding_root_dir,
            split_name,
            f"{audio_file_name}.pt",
        )

        embedding = torch.load(embedding_path, map_location="cpu")

        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"Expected tensor embedding at {embedding_path}")

        if embedding.dim() == 3 and embedding.size(0) == 1:
            embedding = embedding.squeeze(0)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        if embedding.dim() != 2:
            raise ValueError(
                f"Expected 1D or 2D tensor at {embedding_path}, got shape {tuple(embedding.shape)}"
            )

        if target_prefix_length is not None:
            embedding = torch.nn.functional.adaptive_avg_pool1d(
                embedding.transpose(0, 1).unsqueeze(0),
                target_prefix_length,
            ).squeeze(0).transpose(0, 1)

        batch_embeddings.append(embedding)
        max_sequence_length = max(max_sequence_length, embedding.size(0))

    padded_embeddings = []

    for embedding in batch_embeddings:
        sequence_length = embedding.size(0)
        pad_length = max_sequence_length - sequence_length

        if pad_length > 0:
            pad_tensor = torch.zeros(
                pad_length,
                embedding.size(1),
                dtype=embedding.dtype,
            )
            embedding = torch.cat([embedding, pad_tensor], dim=0)

        padded_embeddings.append(embedding)
        audio_attention_masks.append(
            torch.cat(
                [
                    torch.ones(sequence_length, dtype=torch.long),
                    torch.zeros(pad_length, dtype=torch.long),
                ],
                dim=0,
            )
        )

    test_encoder_outputs = torch.stack(padded_embeddings, dim=0)
    audio_attention_mask = torch.stack(audio_attention_masks, dim=0)
    return test_encoder_outputs.to(device), audio_attention_mask.to(device)


def run_batch(model, batch, device, inspect_batch=False):
    # Step A. Text side inputs from dataset
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    prompt_length = batch["prompt_length"][0].item()

    # Step B. Placeholder for future fusion encoder output
    fake_encoder_outputs, audio_attention_mask = get_test_encoder_outputs(
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
        audio_attention_mask=audio_attention_mask,
    )

    if inspect_batch:
        caption_token_count = (labels != -100).sum().item()
        print("[inspect] batch structure")
        print(f"[inspect] test_encoder_outputs shape: {tuple(fake_encoder_outputs.shape)}")
        print(f"[inspect] input_ids shape: {tuple(input_ids.shape)}")
        print(f"[inspect] attention_mask shape: {tuple(attention_mask.shape)}")
        print(f"[inspect] labels shape: {tuple(labels.shape)}")
        print(f"[inspect] prefix_length: {fake_encoder_outputs.shape[1]}")
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

    wandb.init(
        project=config["project"],
        entity=config["entity"],
        name=config["name"],
        config=config,
        dir=config["wandb_dir"],
    )

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
    # ----------------- 
    # Train 
    # ------------------
    for epoch_idx in range(start_epoch, epochs):
        print(f"\n========== Epoch {epoch_idx + 1}/{epochs} ==========")
        # print(" Train decoder with fake encoder outputs")
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

            test_encoder_outputs, audio_attention_mask = get_test_encoder_outputs(
                [sample["audio_path"] for sample in preview_samples],
                device,
            )

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
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

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
    wandb.finish()


if __name__ == "__main__":
    main()
