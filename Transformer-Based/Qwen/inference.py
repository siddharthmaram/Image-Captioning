#!/usr/bin/env python3
import os
import torch
import pandas as pd
import csv
import time
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


BASE_FOLDER = "path/to/dataset"
TEST_ROOT = os.path.join(BASE_FOLDER, "test")
TEST_CSV = os.path.join(TEST_ROOT, "metadata.csv") 

ADAPTER_PATH = "./qwen3.5-captioning-best" 
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BATCH_SIZE = 4
IMAGE_SIZE = {"longest_edge": 384}

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

print("Loading model in native bfloat16 for fast inference...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()


class QwenEvalDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size=IMAGE_SIZE):
        full_df = pd.read_csv(csv_file)
        self.df = full_df.drop_duplicates(subset=["file_name"]).reset_index(drop=True)
        self.root = root_dir
        self.image_size = image_size

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["file_name"]
        image_path = os.path.join(self.root, img_name)
        
        image = Image.open(image_path).convert("RGB")
        
        max_edge = self.image_size.get("longest_edge", 384)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image in a paragraph."}
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        return {
            "image": image, 
            "text": text_prompt, 
            "image_name": img_name
        }

def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    names = [item["image_name"] for item in batch]

    inputs = processor(
        text=texts, 
        images=images, 
        return_tensors="pt", 
        padding=True
    )
    return inputs, names

dataset = QwenEvalDataset(TEST_CSV, TEST_ROOT)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn, 
    num_workers=4,
    pin_memory=True
)


NUM_WARMUP_BATCHES = 1
NUM_TEST_BATCHES = 5

print(f"\nStarting warmup for {NUM_WARMUP_BATCHES} batch(es)...")
with torch.inference_mode():
    dataloader_iter = iter(dataloader)
    
    for _ in range(NUM_WARMUP_BATCHES):
        try:
            inputs, names = next(dataloader_iter)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dtype in [torch.float32, torch.float16]:
                    inputs[k] = v.to(torch.bfloat16)
            
            _ = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        except StopIteration:
            break

    print(f"Starting timed inference for {NUM_TEST_BATCHES} batch(es)...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    images_processed = 0
    results = []
    
    for _ in tqdm(range(NUM_TEST_BATCHES)):
        try:
            inputs, names = next(dataloader_iter)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dtype in [torch.float32, torch.float16]:
                    inputs[k] = v.to(torch.bfloat16)

            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
            prompt_lengths = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, prompt_lengths:]
            generated_texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
            
            for name, text in zip(names, generated_texts):
                results.append([name, text.strip()])
                
            images_processed += len(names)
            
        except StopIteration:
            break 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()

total_time = end_time - start_time
avg_time_per_image = total_time / images_processed if images_processed > 0 else 0

print(f"\n--- Timing Results ---")
print(f"Total Images Processed: {images_processed}")
print(f"Total Time Taken: {total_time:.4f} seconds")
print(f"Average Time Per Image: {avg_time_per_image:.4f} seconds")
print(f"Throughput: {1/avg_time_per_image:.2f} images/second")

output_file = "predictions_test_run.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(results)
print(f"Saved test run to {output_file}")