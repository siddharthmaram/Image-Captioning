import os, sys, math, random, json, gc
from dataclasses import dataclass
from typing import List, Dict, Any
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PIL import Image
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


# Base/model/output
MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
OUTPUT_DIR = "llava-llama3-next-8b-caption-17k-inter"

# Dataset root (Point this to where BioKosh17k-Cleaned is located)
DATASET_ROOT = "BioKosh17k-Cleaned"

# We define the specific paths for Train and Validation
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

TRAIN_CSV = os.path.join(TRAIN_DIR, "metadata.csv") 
VAL_CSV = os.path.join(VAL_DIR, "metadata.csv")
# TRAIN_CSV = os.path.join(DATASET_ROOT, "train_captions.csv")
# VAL_CSV = os.path.join(DATASET_ROOT, "val_captions.csv")

INSTRUCTION = "Describe the image in a paragraph"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_ds(csv_path, base_dir):
    """Loads CSV and prepends the directory path to image filenames."""
    df = pd.read_csv(csv_path)
    
    # Rename columns to match the training pipeline's expectations
    df = df.rename(columns={"file_name": "image", "text": "caption"})
    
    # Prepend the directory path (e.g., '10004.png' -> '.../train/10004.png')
    df["image"] = df["image"].apply(lambda x: os.path.join(base_dir, x))
    
    # Filter for safety
    initial_count = len(df)
    df = df[df["image"].apply(os.path.exists)].reset_index(drop=True)
    
    print(f"Loaded {len(df)}/{initial_count} samples from {csv_path}")
    return Dataset.from_pandas(df)



def build_texts(caption: str):
    template_user = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}
    ]
    prompt_only_text = processor.apply_chat_template(
        template_user,
        add_generation_prompt=True,
        tokenize=False
    )

    template_full = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]},
    ]
    full_text = processor.apply_chat_template(
        template_full,
        tokenize=False
    )
    return prompt_only_text, full_text

def map_build_texts(ex):
    p, f = build_texts(ex["caption"])
    return {"prompt": p, "full_text": f}


def collate_fn(batch):
    # 1. Load images and handle RGB conversion
    images = [Image.open(item["image"]).convert("RGB") for item in batch]
    full_texts = [item["full_text"] for item in batch]
    prompt_texts = [item["prompt"] for item in batch]

    # 2. Processor handles AnyRes patching (creates 5D tensors for LLaVA 1.6)
    # DO NOT use .to(device) here; let the Trainer handle it
    inputs = processor(
        images=images,
        text=full_texts,
        padding=True,
        return_tensors="pt"
    )

    # 3. Compute prompt lengths to mask them in labels
    prompt_lens = [
        len(processor.tokenizer(p, add_special_tokens=False).input_ids)
        for p in prompt_texts
    ]

    # 4. Build labels: mask everything up to the assistant's response
    labels = inputs["input_ids"].clone()
    labels[:] = -100  # Start fully masked

    for i in range(len(batch)):
        ids = inputs["input_ids"][i]
        nonpad = ids != pad_id
        total_nonpad = int(nonpad.sum().item())
        assist_len = total_nonpad - prompt_lens[i]
        
        if assist_len > 0:
            target_tokens = ids[nonpad][-assist_len:]
            labels[i, -assist_len:] = target_tokens

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    train_ds = load_and_prepare_ds(TRAIN_CSV, TRAIN_DIR)
    val_ds = load_and_prepare_ds(VAL_CSV, VAL_DIR)

    # Wrap into DatasetDict
    raw = DatasetDict({
        "train": train_ds,
        "validation": val_ds
    })

    print(raw)
    
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    
    train_proc = raw["train"].map(map_build_texts)
    val_proc = raw["validation"].map(map_build_texts)

    pad_id = processor.tokenizer.pad_token_id
    
    model.config.use_cache = False

    # Prepare for k-bit training and wrap with LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "mm_projector"]
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mm_projector"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    
    wandb.init(project="llava-bio-captioning", name="llama3-llava-next-8b")


    # 5) Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        logging_strategy="steps",
        eval_steps=400,
        save_steps=400,
        save_total_limit=2,
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=True,
        report_to="wandb",
        remove_unused_columns=False,  # <-- important
    )
    

    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_proc,
        eval_dataset=val_proc,
        data_collator=collate_fn,
    )

    trainer.train()

    # 7) Save LoRA adapter and processor
    trainer.save_model(OUTPUT_DIR)  # saves PEFT adapter
    processor.save_pretrained(OUTPUT_DIR)
    print("Saved LoRA adapter + processor to:", OUTPUT_DIR)