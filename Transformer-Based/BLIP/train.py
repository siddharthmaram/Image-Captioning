import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset import ImageCaptionDataset

from transformers import AutoProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

import wandb

BASE_FOLDER = "path/to/dataset"

train_folder = os.path.join(BASE_FOLDER, "train")
train_csv = os.path.join(train_folder, "metadata.csv")

val_folder = os.path.join(BASE_FOLDER, "validation")
val_csv = os.path.join(val_folder, "metadata.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    device_map="auto"
)

# for param in model.parameters():
#     param.requires_grad = False

# for i in [-1, -4]:
#     for param in model.text_decoder.bert.encoder.layer[i].parameters():
#         param.requires_grad = True

# for i in [-1, -4]:
#     layer = model.text_decoder.bert.encoder.layer[i]
#     if hasattr(layer, "crossattention"):
#         for param in layer.crossattention.parameters():
#             param.requires_grad = True

train_dataset = ImageCaptionDataset(train_csv, train_folder, processor)
val_dataset = ImageCaptionDataset(val_csv, val_folder, processor)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
)

num_epochs = 30
total_steps = len(train_loader) * num_epochs

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=1e-5)
print(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

wandb.init(project="Captioning-17k")

best_val_loss = float("inf")
model.train()


try:
    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}")

        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for step, batch in enumerate(train_bar):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask, 
                labels=input_ids
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)
                attention_mask = batch.pop("attention_mask").to(device)

                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                val_loss += outputs.loss.item()
                val_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} | Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(f"best_model_epoch")
            print(f"best_model_epoch saved")

        model.train()
except KeyboardInterrupt:
    print("\nInterrupted! Saving current state...")
    save_path = f"interrupted_checkpoint_epoch_{epoch}"
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    
    processor.save_pretrained(save_path)
    
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, os.path.join(save_path, "training_state.pt"))
    
    print(f"State saved to {save_path}")

wandb.finish()

