#!/usr/bin/env python3
import os
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

random.seed(42)
torch.manual_seed(42)

BASE_FOLDER = "path/to/dataset"
TRAIN_SPLIT = "train" 
VAL_SPLIT = "validation"


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "./qwen3.5-captioning-10k"
BEST_OUTPUT_DIR = "./qwen3.5-captioning-10k-best"

BATCH_SIZE = 4           
GRAD_ACCUM_STEPS = 4         
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1

MAX_LEN = 4096
IMAGE_SIZE = {"longest_edge": 384}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

processor.tokenizer.padding_side = "right"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

print("Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


lora_cfg = LoraConfig(
    r=32,                    
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head" 
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.train()


class ImageCaptionDataset(Dataset):
    def __init__(self, processor, split, image_size=IMAGE_SIZE):
        self.data = load_dataset("imagefolder", data_dir=BASE_FOLDER, split=split)
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample["image"].convert("RGB")
        caption = sample["text"]

        # Resize image to keep token count manageable
        max_edge = self.image_size.get("longest_edge", 384)
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in a paragraph."},
                ],
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": caption}]
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            tokenize=False
        )
        
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        return {
            "full_text": full_text,
            "prompt_text": prompt_text,
            "image": img
        }


def collate_fn(batch):
    full_texts = [item["full_text"] for item in batch]
    prompt_texts = [item["prompt_text"] for item in batch]
    images = [item["image"] for item in batch]
    
    batch_inputs = processor(
        text=full_texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    
    prompt_inputs = processor(
        text=prompt_texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    
    labels = batch_inputs["input_ids"].clone()
    
    for i in range(len(batch)):
        prompt_len = prompt_inputs["attention_mask"][i].sum().item()
        
        labels[i, :prompt_len] = -100
        
        labels[i, batch_inputs["attention_mask"][i] == 0] = -100
        
    batch_inputs["labels"] = labels
    return batch_inputs


print("Loading datasets...")
train_ds = ImageCaptionDataset(processor, TRAIN_SPLIT)
val_ds = ImageCaptionDataset(processor, VAL_SPLIT)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
num_warmup_steps = int(WARMUP_RATIO * max_train_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)


run = wandb.init(
    project="Captioning-10k",
    config={
        "model": MODEL_NAME,
        "lr": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "effective_bs": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "max_len": MAX_LEN,
    },
)

@torch.no_grad()
def run_eval(model, dataloader):
    model.eval()
    losses = []
    
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        outputs = model(**batch)
        losses.append(outputs.loss.item())

    model.train()
    return sum(losses) / len(losses)


model.train()
global_step = 0
best_val_loss = float("inf")
accumulated_loss = 0.0
logging_loss = 0.0
log_steps_count = 0

print(f"\nStarting training for {max_train_steps} steps...")

try:
    for epoch in range(NUM_EPOCHS):
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(epoch_bar):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            accumulated_loss += outputs.loss.item()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                log_steps_count += 1
                
                if global_step % 10 == 0:
                    avg_loss = accumulated_loss / (log_steps_count * GRAD_ACCUM_STEPS)
                    run.log({
                        "train_loss": avg_loss,
                        "lr": optimizer.param_groups[0]['lr'],
                        "step": global_step
                    })
                    epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    accumulated_loss = 0.0
                    log_steps_count = 0
            
            if global_step >= max_train_steps:
                break
        
        val_loss = run_eval(model, val_loader)
        print(f"\n[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")
        run.log({"val_loss": val_loss, "epoch": epoch+1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best! Saving to {BEST_OUTPUT_DIR}")
            model.save_pretrained(BEST_OUTPUT_DIR)
            processor.save_pretrained(BEST_OUTPUT_DIR)

except KeyboardInterrupt:
    print("\nTraining interrupted...")

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
run.finish()