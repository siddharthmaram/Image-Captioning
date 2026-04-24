import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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
        
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            return_tensors="pt"
        )
        
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding