import os
import json
import csv
import time
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import CaptionCSV
from vocabulary import Vocab, SPECIALS
from config import cfg
from models import Captioner

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_ID = SPECIALS['<pad>']; BOS_ID = SPECIALS['<bos>']; EOS_ID = SPECIALS['<eos>']


NUM_WARMUP_SAMPLES = 2
NUM_TEST_SAMPLES = 20 


checkpoint = torch.load('sat_csv_biology/sat_model_best.pt', map_location='cuda')

base_biology_dir = "/home/maramreddy/dataset/Captioning-17k/BioKosh17k-Cleaned"
train_images = os.path.join(base_biology_dir, "train")
test_images = os.path.join(base_biology_dir, "test")
val_images = os.path.join(base_biology_dir, "validation")

def get_biology_splits():
    train_split = os.path.join(train_images, "metadata.csv")
    test_split = os.path.join(test_images, "metadata.csv")
    val_split = os.path.join(val_images, "metadata.csv")

    with open(train_split, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        train_data = list(reader)
    with open(test_split, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        test_data = list(reader)
        unique_images = set()
        filtered_test_data = []

        for img, caption in test_data:
            if img not in unique_images:
                unique_images.add(img)
                filtered_test_data.append([img, caption])

    with open(val_split, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        val_data = list(reader)

    train_df = pd.DataFrame(train_data, columns=["image", "caption"])
    val_df = pd.DataFrame(val_data, columns=["image", "caption"])
    test_df = pd.DataFrame(filtered_test_data, columns=["image", "caption"])

    return train_df, val_df, test_df

train_df, val_df, test_df = get_biology_splits()

word_map = checkpoint['vocab'] 
vocab = Vocab(cfg.MIN_WORD_FREQ)
vocab.w2i = word_map
vocab.i2w = {i: w for w, i in word_map.items()}

train_ds = CaptionCSV(train_df, train_images, vocab, cfg.MAX_LEN, train_split=True)
val_ds   = CaptionCSV(val_df,   val_images, vocab, cfg.MAX_LEN, train_split=False)
test_ds  = CaptionCSV(test_df,  test_images,  vocab, cfg.MAX_LEN, train_split=False)


model = Captioner(len(vocab.w2i)).to(DEVICE)
model.load_state_dict(checkpoint['model'])
model.eval()


print(f"\nStarting warmup for {NUM_WARMUP_SAMPLES} sample(s)...")


with torch.inference_mode():
    
    for i in range(NUM_WARMUP_SAMPLES):
        if i >= len(test_ds): break
        _, img, _ = test_ds[i]
        img = img.unsqueeze(0).to(DEVICE)
        
        V = model.enc(img)
        _ = model.dec.beam_search(V, BOS_ID, EOS_ID, beam=cfg.BEAM_SIZE, max_len=cfg.MAX_LEN)

    print(f"Starting timed inference for {NUM_TEST_SAMPLES} sample(s)...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    samples_processed = 0
    predictions = []

    for i in tqdm(range(NUM_TEST_SAMPLES)):
        if i >= len(test_ds): break
        img_name, img, _ = test_ds[i]
        img = img.unsqueeze(0).to(DEVICE)
        
        V = model.enc(img)
        seq = model.dec.beam_search(V, BOS_ID, EOS_ID, beam=cfg.BEAM_SIZE, max_len=cfg.MAX_LEN)
        pred = vocab.decode(seq[1:])
        
        predictions.append([img_name, pred])
        samples_processed += 1

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

out_dir = getattr(cfg, 'OUT_DIR', '.') 
output_file = os.path.join(out_dir, 'predictions_sat_test.csv')

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(predictions)

print(f"\nDone! Saved test run to {output_file}")