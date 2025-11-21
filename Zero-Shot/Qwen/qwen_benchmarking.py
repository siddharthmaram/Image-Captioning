import os
import csv
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DATASET_DIR = "/home/maramreddy/dataset/Biology-Dataset/test"
CSV_FILE = os.path.join(DATASET_DIR, "metadata.csv")
OUTPUT_FILE = "captions.csv"

# ----------------------------------------------------------------------
# Load model & processor
# ----------------------------------------------------------------------
print("🔹 Loading Qwen2.5-VL-7B-Instruct model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# ----------------------------------------------------------------------
# Define the caption generation prompt
# ----------------------------------------------------------------------
PROMPT = (
    "Strictly compose a single, small, concise paragraph of 3 to 4 lines "
    "describing this labeled biological diagram. Identify all key biological "
    "structures, organelles, molecules, or entities indicated by the labels "
    "or annotations. Explicitly reference these labels and explain their "
    "spatial relationships, interactions, and functions with precise biological "
    "terminology. Ensure the description is succinct, informative, and scientifically "
    "accurate, without extra elaboration or multiple paragraphs."
)

# ----------------------------------------------------------------------
# Read image list
# ----------------------------------------------------------------------
images = []
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header if present
    for row in reader:
        if len(row) >= 1:
            img_path = os.path.join(DATASET_DIR, row[0])
            if os.path.exists(img_path):
                images.append(img_path)

print(f"📸 Found {len(images)} images for captioning.")

# ----------------------------------------------------------------------
# Generate captions
# ----------------------------------------------------------------------
results = []

for img_path in tqdm(images, desc="Generating captions"):
    try:
        # Prepare input message
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
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        results.append((os.path.basename(img_path), output_text))

    except Exception as e:
        print(f"[ERROR] {os.path.basename(img_path)}: {e}")
        results.append((os.path.basename(img_path), f"Error: {e}"))

# ----------------------------------------------------------------------
# Save results to CSV
# ----------------------------------------------------------------------
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(results)

print(f"✅ Captions saved to: {OUTPUT_FILE}")

