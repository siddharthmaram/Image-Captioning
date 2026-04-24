import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from transformers import AutoProcessor, BlipForConditionalGeneration

import csv

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "best_model_epoch",
    device_map="auto"
)

model.eval()

test_dataset = ImageCaptionDataset(test_csv, test_folder, processor)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, processor)
)

predictions = []

print("Generating captions...")
with torch.inference_mode():
    for image_names, pixel_values, captions, ids in tqdm(test_loader, desc="Evaluating"):
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
            prediction = generated_text[i]
            predictions.append([img, prediction])  
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"Generated {len(predictions)} captions")

with open("predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(predictions)
