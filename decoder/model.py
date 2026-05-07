import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model

class AudioPrefixGPT2(nn.Module):
    def __init__(
        self,
        model_name="gpt2",
        audio_dim=512, #FIXME: CHANGE TO AUDIO EMB SIZE
        freeze_gpt2=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()

        # Load GPT-2
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name)

        # GPT-2 hidden size, usually 768 for "gpt2"
        self.gpt_dim = self.gpt2.config.n_embd

        # Project audio embeddings into GPT-2 embedding dimension
        self.audio_projection = nn.Linear(audio_dim, self.gpt_dim)

        # Optionally freeze GPT-2 original weights
        if freeze_gpt2:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        # Add LoRA adapters to GPT-2

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.gpt2 = get_peft_model(self.gpt2, lora_config)

    def forward(
        self,
        audio_embeddings,
        input_ids,
        attention_mask,
        labels=None,
        prompt_length=None,
    ):
        """
        audio_embeddings:
            Shape: (batch, prefix_length, audio_dim)

        input_ids:
            Token IDs for prompt + caption.
            Shape: (batch, text_length)

        attention_mask:
            Attention mask for prompt + caption.
            Shape: (batch, text_length)

        labels:
            Usually same as input_ids before masking.
            Shape: (batch, text_length)

        prompt_length:
            Number of prompt tokens. These tokens will be ignored in the loss.
        """

        batch_size = input_ids.size(0)
        prefix_length = audio_embeddings.size(1)

        # Convert audio embeddings to GPT-2 embedding size
        audio_prefix_embeds = self.audio_projection(audio_embeddings)
        # Shape: (batch, prefix_length, gpt_dim)

        # Convert text token IDs to GPT-2 token embeddings
        # text_embeds = self.gpt2.transformer.wte(input_ids)
        text_embeds = self.gpt2.get_input_embeddings()(input_ids)
        # Shape: (batch, text_length, gpt_dim)

        # Concatenate audio prefix + text embeddings
        inputs_embeds = torch.cat(
            [audio_prefix_embeds, text_embeds],
            dim=1,
        )
        # Shape: (batch, prefix_length + text_length, gpt_dim)

        # Extend attention mask for audio prefix
        prefix_attention_mask = torch.ones(
            batch_size,
            prefix_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        new_attention_mask = torch.cat(
            [prefix_attention_mask, attention_mask],
            dim=1,
        )
        # Shape: (batch, prefix_length + text_length)

        new_labels = None

        if labels is not None:
            # Copy labels so input_ids are not modified
            labels = labels.clone()

            # Ignore padding tokens in loss
            labels[attention_mask == 0] = -100

            # Ignore prompt tokens in loss
            if prompt_length is not None:
                labels[:, :prompt_length] = -100

            # Ignore audio prefix positions in loss
            prefix_labels = torch.full(
                (batch_size, prefix_length),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )

            new_labels = torch.cat(
                [prefix_labels, labels],
                dim=1,
            )
            # Shape: (batch, prefix_length + text_length)

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
        )

        return outputs

    def generate_caption(
        self,
        audio_embeddings,
        input_ids,
        attention_mask,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
        Generate caption from audio embeddings + prompt.
        """

        batch_size = input_ids.size(0)
        prefix_length = audio_embeddings.size(1)

        # Audio embedding -> GPT-2 embedding dimension
        audio_prefix_embeds = self.audio_projection(audio_embeddings)

        # Prompt token IDs -> GPT-2 text embeddings
        # text_embeds = self.gpt2.transformer.wte(input_ids)
        text_embeds = self.gpt2.get_input_embeddings()(input_ids)

        # Audio prefix + prompt embeddings
        inputs_embeds = torch.cat(
            [audio_prefix_embeds, text_embeds],
            dim=1,
        )

        # Attention mask for audio prefix + prompt
        prefix_attention_mask = torch.ones(
            batch_size,
            prefix_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        new_attention_mask = torch.cat(
            [prefix_attention_mask, attention_mask],
            dim=1,
        )

        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": new_attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }

        # temperature and top_p are only meaningful when sampling is enabled
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        generated_ids = self.gpt2.generate(**generation_kwargs)

        return generated_ids

    def print_trainable_parameters(self):
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name} | shape: {tuple(param.shape)}")

        print("=" * 60)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")
