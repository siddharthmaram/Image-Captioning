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
nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)

from tqdm import tqdm


os.makedirs(cfg.OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# def read_split_csv(split_dir: str, csv_name: str):
#     csv_path = os.path.join(split_dir, csv_name)
#     df = pd.read_csv(csv_path)
#     # normalize column names
#     cols = {c.lower().strip(): c for c in df.columns}
#     assert 'file_name' in cols and 'text' in cols, "CSV must have 'file_name' and 'text' columns"
#     file_col = cols['file_name']
#     text_col = cols['text']
#     # ensure strings
#     df[file_col] = df[file_col].astype(str)
#     df[text_col] = df[text_col].astype(str)
#     return df[[file_col, text_col]].rename(columns={file_col:'file_name', text_col:'caption'})

# base = os.path.join(cfg.DATASET_ROOT, cfg.USE_FOLDER_WITH_SPACE)
# train_dir = os.path.join(base, 'train')
# test_dir  = os.path.join(base, 'test')

# train_df = read_split_csv(train_dir, cfg.USE_DESCRIPTION)
# test_df  = read_split_csv(test_dir,  cfg.USE_DESCRIPTION)

# # Build val as 10% of *train* deterministically (you asked to use the given split; we keep test intact)
# perm = list(range(len(train_df)))
# random.Random(SEED).shuffle(perm)
# val_count = max(1, int(0.1 * len(train_df)))
# val_idx = set(perm[:val_count])
# val_df = train_df.iloc[list(val_idx)].reset_index(drop=True)
# train_df2 = train_df.iloc[list(set(perm[val_count:]))].reset_index(drop=True)

# print('Train/Val/Test sizes:', len(train_df2), len(val_df), len(test_df))

# Flickr-30k
base_dir = "/home/maramreddy/dataset/Show-Attend-and-Tell/data/flickr30k"
image_dir = os.path.join(base_dir, "Images")

def get_flickr_splits():
    captions_file = os.path.join(base_dir, "captions.txt")
    train_split = os.path.join(base_dir, "train.txt")
    test_split = os.path.join(base_dir, "test.txt")
    val_split = os.path.join(base_dir, "val.txt")

    # Load splits
    with open(train_split, "r") as f:
        train_imgs = set(line.strip() for line in f)
    with open(val_split, "r") as f:
        val_imgs = set(line.strip() for line in f)
    with open(test_split, "r") as f:
        test_imgs = set(line.strip() for line in f)

    # Storage
    train_files, train_texts = [], []
    val_files, val_texts = [], []
    test_files, test_texts = [], []

    with open(captions_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip if there is a header
        for file_name, caption in reader:
            image = file_name.split(".")[0]  # check this based on train.txt contents
            if image in train_imgs:
                train_files.append(file_name)
                train_texts.append(caption)
            elif image in test_imgs:
                test_files.append(file_name)
                test_texts.append(caption)
            elif image in val_imgs:
                val_files.append(file_name)
                val_texts.append(caption)

    # DataFrames
    train_df = pd.DataFrame({"file_name": train_files, "caption": train_texts})
    val_df = pd.DataFrame({"file_name": val_files, "caption": val_texts})
    test_df = pd.DataFrame({"file_name": test_files, "caption": test_texts})

    return train_df, val_df, test_df

base_biology_dir = "/home/maramreddy/dataset/Biology-Dataset"
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
    train_df = pd.DataFrame(train_data, columns=["file_name", "caption"])
    val_df = pd.DataFrame(val_data, columns=["file_name", "caption"])
    test_df = pd.DataFrame(test_data, columns=["file_name", "caption"])

    return train_df, val_df, test_df


train_df2, val_df, test_df = get_biology_splits()

# Make COCO-like dicts

def to_coco(df: pd.DataFrame):
    images = []
    annotations = []
    for i, row in enumerate(df.itertuples(index=False)):
        images.append({"id": i, "file_name": row.file_name})
        annotations.append({"image_id": i, "caption": row.caption})
    return {"images": images, "annotations": annotations}

ann_dir = os.path.join(cfg.OUT_DIR, 'annotations')
os.makedirs(ann_dir, exist_ok=True)
with open(os.path.join(ann_dir, 'train.json'), 'w') as f: json.dump(to_coco(train_df2), f)
with open(os.path.join(ann_dir, 'val.json'),   'w') as f: json.dump(to_coco(val_df), f)
with open(os.path.join(ann_dir, 'test.json'),  'w') as f: json.dump(to_coco(test_df), f)

vocab = Vocab(cfg.MIN_WORD_FREQ).build(list(train_df2['caption']))
with open(os.path.join(cfg.OUT_DIR,'vocab.json'),'w') as f: json.dump(vocab.w2i, f)
print('Vocab size:', len(vocab.w2i))


# train_ds = CaptionCSV(train_df2, os.path.join(train_dir,'Images'), vocab, cfg.MAX_LEN, train_split=True)
# val_ds   = CaptionCSV(val_df,   os.path.join(train_dir,'Images'), vocab, cfg.MAX_LEN, train_split=False)
# test_ds  = CaptionCSV(test_df,  os.path.join(test_dir,'Images'),  vocab, cfg.MAX_LEN, train_split=False)

train_ds = CaptionCSV(train_df2, train_images, vocab, cfg.MAX_LEN, train_split=True)
val_ds   = CaptionCSV(val_df,   val_images, vocab, cfg.MAX_LEN, train_split=False)
test_ds  = CaptionCSV(test_df,  test_images,  vocab, cfg.MAX_LEN, train_split=False)

PAD_ID = SPECIALS['<pad>']; BOS_ID = SPECIALS['<bos>']; EOS_ID = SPECIALS['<eos>']

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    caps = torch.stack([b[1] for b in batch])
    return imgs, caps

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True, collate_fn=collate)

checkpoint = torch.load('sat_csv_biology/sat_model_20.pt', map_location='cuda')
state_dict = checkpoint["model"]
# state_dict.pop('dec.embed.weight', None)
# state_dict.pop('dec.out.weight', None)
# state_dict.pop('dec.out.bias', None)


model = Captioner(len(vocab.w2i)).to(DEVICE)
model.load_state_dict(state_dict, strict=False)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR)

def ce_loss(logits, caps):
    # logits: (B,T-1,V), targets next tokens
    gold = caps[:,1:1+logits.size(1)]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=PAD_ID)

for epoch in range(21, cfg.EPOCHS+21):
    model.train(); total=0.0
    for imgs, caps in tqdm(train_loader):
        imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
        opt.zero_grad()
        logits = model(imgs, caps)
        loss = ce_loss(logits, caps)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
        opt.step(); total += float(loss.item())
    print(f"Epoch {epoch:02d} | train CE: {total/len(train_loader):.4f}")

    # quick val perplexity proxy
    model.eval(); vloss=0.0
    with torch.no_grad():
        for imgs, caps in tqdm(val_loader):
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            logits = model(imgs, caps)
            vloss += float(ce_loss(logits, caps).item())
    print(f"           | val CE:   {vloss/len(val_loader):.4f}")

    # Save checkpoint
    ckpt = os.path.join(cfg.OUT_DIR, f'sat_model_{epoch}.pt')
    torch.save({'model': model.state_dict(), 'vocab': vocab.w2i, 'cfg': cfg.__dict__}, ckpt)
    print('Saved model to', ckpt)
