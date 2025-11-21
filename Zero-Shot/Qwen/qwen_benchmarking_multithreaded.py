import os
import csv
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Paths
DATASET_DIR = "/home/maramreddy/dataset/Biology-Dataset/test"
CSV_FILE = os.path.join(DATASET_DIR, "metadata.csv")
OUTPUT_FILE = "captions.csv"


# Load model & processor
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "float32",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

PROMPT = (
    "Strictly compose a single, small, concise paragraph of 3 to 4 lines "
    "describing this labeled biological diagram. Identify all key biological "
    "structures, organelles, molecules, or entities indicated by the labels "
    "or annotations. Explicitly reference these labels and explain their "
    "spatial relationships, interactions, and functions with precise biological "
    "terminology. Ensure the description is succinct, informative, and scientifically "
    "accurate, without extra elaboration or multiple paragraphs."
)


# Read image paths
images = []
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    _ = next(reader, None)
    for row in reader:
        if len(row) >= 1:
            img_path = os.path.join(DATASET_DIR, row[0])
            if os.path.exists(img_path):
                images.append(img_path)

print(f"Found {len(images)} images for captioning.")

# Threaded preprocessing (I/O parallel)
def prepare_inputs(img_path):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return img_path, inputs
    except Exception as e:
        return img_path, e

# Preprocess in threads (I/O and CPU)
print("🧩 Preprocessing images in parallel...")
prepared = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(prepare_inputs, img) for img in images]
    for future in tqdm(as_completed(futures), total=len(futures)):
        prepared.append(future.result())


# Generate captions (GPU sequential)
results = []
for img_path, inputs in tqdm(prepared, desc="Generating captions"):
    if isinstance(inputs, Exception):
        results.append((os.path.basename(img_path), f"Error: {inputs}"))
        continue
    try:
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        results.append((os.path.basename(img_path), output_text))
    except Exception as e:
        results.append((os.path.basename(img_path), f"Error: {e}"))


# Save results
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(results)

print(f"Captions saved to {OUTPUT_FILE}")

