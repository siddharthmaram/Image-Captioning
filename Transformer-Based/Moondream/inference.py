from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import math
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from dataset import datasets
import csv
import os
import time

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MD_REVISION = "2024-05-20"

NUM_WARMUP_SAMPLES = 2
NUM_TEST_SAMPLES = 20


print("Loading Moondream2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
base_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

print("Loading fine-tuned weights...")
state_dict = load_file("checkpoints/moondream-best/model.safetensors", device="cpu")
missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

moondream2 = base_model.eval().to(DEVICE)


dataset_iterator = iter(datasets['test'])

print(f"\nStarting warmup for {NUM_WARMUP_SAMPLES} sample(s)...")
with torch.inference_mode():
    for _ in range(NUM_WARMUP_SAMPLES):
        try:
            sample = next(dataset_iterator)
            img = sample["image"]
            img = img.convert("RGB") if hasattr(img, "mode") and img.mode != "RGB" else img
            
            _ = moondream2.answer_question(
                moondream2.encode_image(img),
                sample['qa'][0]['question'],
                tokenizer=tokenizer,
                num_beams=4,
                no_repeat_ngram_size=5,
                early_stopping=True
            )
        except StopIteration:
            break

    print(f"Starting timed inference for {NUM_TEST_SAMPLES} sample(s)...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    samples_processed = 0
    results = []

    for _ in tqdm(range(NUM_TEST_SAMPLES)):
        try:
            sample = next(dataset_iterator)
            img = sample["image"]
            imgname = os.path.basename(img.filename)
            img = img.convert("RGB") if hasattr(img, "mode") and img.mode != "RGB" else img
            
            md_answer = moondream2.answer_question(
                moondream2.encode_image(img),
                sample['qa'][0]['question'],
                tokenizer=tokenizer,
                num_beams=4,
                no_repeat_ngram_size=5,
                early_stopping=True
            )

            results.append([imgname, md_answer])
            samples_processed += 1
            
        except StopIteration:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()

total_time = end_time - start_time
avg_time_per_image = total_time / samples_processed if samples_processed > 0 else 0

print(f"\n--- Timing Results ---")
print(f"Total Images Processed: {samples_processed}")
print(f"Total Time Taken: {total_time:.4f} seconds")
print(f"Average Time Per Image: {avg_time_per_image:.4f} seconds")
print(f"Throughput: {1/avg_time_per_image:.2f} images/second")

output_file = "predictions_moondream_test.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(results)

print(f"\nSaved predictions to {output_file}")