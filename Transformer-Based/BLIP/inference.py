import os
import time
import torch
import pandas as pd
import csv
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, image_folder, processor):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_folder, row["file_name"])
        with Image.open(img_path) as im:
            if im.mode == "P":
                im = im.convert("RGBA")
            image = im.convert("RGB")
        caption = row["text"]
        return {
            "image_name": row["file_name"], 
            "image": image, 
            "caption": caption, 
            "id": idx
        }

def collate_fn(batch, processor):
    image_names = [item["image_name"] for item in batch]
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    ids = [item["id"] for item in batch]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    return image_names, pixel_values, captions, ids

BASE_FOLDER = "path/to/dataset"
test_folder = os.path.join(BASE_FOLDER, "test")
test_csv = os.path.join(test_folder, "metadata.csv")

BATCH_SIZE = 4
NUM_WARMUP_BATCHES = 1
NUM_TEST_BATCHES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading processor and model...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "best_model_epoch",
    device_map="auto"
)
model.eval()

test_dataset = ImageCaptionDataset(test_csv, test_folder, processor)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, processor),
    num_workers=4,
    pin_memory=True
)


print(f"\nStarting warmup for {NUM_WARMUP_BATCHES} batch(es)...")
with torch.inference_mode():
    dataloader_iter = iter(test_loader)
    
    for _ in range(NUM_WARMUP_BATCHES):
        try:
            image_names, pixel_values, captions, ids = next(dataloader_iter)
            pixel_values = pixel_values.to(device)
            
            _ = model.generate(
                pixel_values=pixel_values,
                max_length=512,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        except StopIteration:
            break

    print(f"Starting timed inference for {NUM_TEST_BATCHES} batch(es)...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    images_processed = 0
    predictions = []

    for _ in tqdm(range(NUM_TEST_BATCHES), desc="Evaluating"):
        try:
            image_names, pixel_values, captions, ids = next(dataloader_iter)
            pixel_values = pixel_values.to(device)

            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=512,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for i, idx in enumerate(ids):
                img = image_names[i]
                prediction = generated_text[i].strip()
                predictions.append([img, prediction])  
                
            images_processed += len(image_names)
            
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

output_file = "predictions_blip_test.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(predictions)

print(f"\nDone! Saved test run to {output_file}")