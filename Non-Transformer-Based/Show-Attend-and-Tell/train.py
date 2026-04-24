import os, json, random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from dataset import CaptionCSV
from config import cfg

from vocabulary import Vocab, SPECIALS
from models import Captioner

import csv

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from tqdm import tqdm
import wandb

os.makedirs(cfg.OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

base_biology_dir = "path/to/dataset"
train_images = os.path.join(base_biology_dir, "train")
test_images = os.path.join(base_biology_dir, "test")
val_images = os.path.join(base_biology_dir, "validation")

def get_biology_splits():
    train_split = os.path.join(train_images, "metadata.csv")
    test_split = os.path.join(test_images, "metadata.csv")
    val_split = os.path.join(val_images, "metadata.csv")

    # Load splits
    with open(train_split, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        train_data = list(reader)
    with open(test_split, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        test_data = list(reader)
    with open(val_split, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        val_data = list(reader)

    # DataFrames
    train_df = pd.DataFrame(train_data, columns=["image", "caption"])
    val_df = pd.DataFrame(val_data, columns=["image", "caption"])
    test_df = pd.DataFrame(test_data, columns=["image", "caption"])

    return train_df, val_df, test_df


train_df, val_df, test_df = get_biology_splits()

def to_coco(df: pd.DataFrame):
    images = []
    annotations = []
    for i, row in enumerate(df.itertuples(index=False)):
        images.append({"id": i, "image": row.image})
        annotations.append({"image_id": i, "caption": row.caption})
    return {"images": images, "annotations": annotations}

ann_dir = os.path.join(cfg.OUT_DIR, 'annotations')
os.makedirs(ann_dir, exist_ok=True)

with open(os.path.join(ann_dir, 'train.json'), 'w') as f: 
    json.dump(to_coco(train_df), f)

with open(os.path.join(ann_dir, 'val.json'),   'w') as f: 
    json.dump(to_coco(val_df), f)

with open(os.path.join(ann_dir, 'test.json'),  'w') as f: 
    json.dump(to_coco(test_df), f)

vocab = Vocab(cfg.MIN_WORD_FREQ).build(list(train_df['caption']))
with open(os.path.join(cfg.OUT_DIR,'vocab.json'),'w') as f: 
    json.dump(vocab.w2i, f)
print('Vocab size:', len(vocab.w2i))


train_ds = CaptionCSV(train_df, train_images, vocab, cfg.MAX_LEN, train_split=True)
val_ds   = CaptionCSV(val_df,   val_images, vocab, cfg.MAX_LEN, train_split=False)
test_ds  = CaptionCSV(test_df,  test_images,  vocab, cfg.MAX_LEN, train_split=False)

PAD_ID = SPECIALS['<pad>']; BOS_ID = SPECIALS['<bos>']; EOS_ID = SPECIALS['<eos>']

def collate(batch):
    imgs = torch.stack([b[1] for b in batch])
    caps = torch.stack([b[2] for b in batch])
    return imgs, caps

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=collate)

model = Captioner(len(vocab.w2i)).to(DEVICE)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR)

def ce_loss(logits, caps):
    # logits: (B,T-1,V), targets next tokens
    gold = caps[:,1:1+logits.size(1)]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=PAD_ID)

wandb.init(project="Captioning-17k")


try:
    best_vloss = float("inf")
    for epoch in range(1, cfg.EPOCHS+1):
        model.train()
        total=0.0

        for imgs, caps in tqdm(train_loader):
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs, caps)
            loss = ce_loss(logits, caps)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
            opt.step()
            total += float(loss.item())

        print(f"Epoch {epoch:02d} | train CE: {total/len(train_loader):.4f}")

        model.eval()
        vloss=0.0

        with torch.no_grad():
            for imgs, caps in tqdm(val_loader):
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
                logits = model(imgs, caps)
                vloss += float(ce_loss(logits, caps).item())
        print(f"           | val CE:   {vloss/len(val_loader):.4f}")

        # Save checkpoint
        if vloss < best_vloss:
            best_vloss = vloss
            ckpt = os.path.join(cfg.OUT_DIR, f'sat_model_best.pt')
            torch.save({
                'epoch': epoch,
                'val_loss': vloss,
                'model': model.state_dict(), 
                'vocab': vocab.w2i, 
                'cfg': cfg.__dict__}, 
                ckpt
            )
            print('Best model saved to', ckpt)
except KeyboardInterrupt:
    ckpt = os.path.join(cfg.OUT_DIR, f'sat_model_last.pt')
    torch.save({
        'epoch': epoch,
        'best_val_loss': best_vloss,
        'optimizer_state': opt.state_dict(),
        'model': model.state_dict(), 
        'vocab': vocab.w2i, 
        'cfg': cfg.__dict__}, 
        ckpt
    )
    print('Last checkpoint saved to', ckpt)
