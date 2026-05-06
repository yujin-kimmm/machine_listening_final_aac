from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pprint import pprint
from dataset import ClothoAudioCaptionDataset


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ClothoAudioCaptionDataset(
        caption_dir="/scratch/yk3281/dataset/clotho",
        audio_root_dir="/scratch/yk3281/dataset/clotho",
        tokenizer=tokenizer,
        prompt="Describe this audio:",
        max_length=128,
        caption_pattern="clotho_captions_*.csv",
        check_files=True,
    )
    print("Dataset size:", len(dataset))


    sample = dataset[0]

    print("\nSingle sample")
    print("source split:", sample["train_split_source"])
    print("audio_id:", sample["audio_id"])
    print("file_name:", sample["file_name"])
    print("audio_path:", sample["audio_path"])
    print("caption:", sample["caption"])
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("labels shape:", sample["labels"].shape)
    print("prompt_length:", sample["prompt_length"])

    print("\nDecoded text:")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=True))

    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    print("\nBatch")
    print("source splits:")
    pprint(batch["train_split_source"])
    print()
    print("audio paths:")
    pprint(batch["audio_path"])
    print()
    print("captions:")
    pprint(batch["caption"])
    print()
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)


if __name__ == "__main__":
    main()
