from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import torch
from torch.utils.data import DataLoader

from dataset import CaptionCSV
from vocabulary import Vocab, SPECIALS
from config import cfg
from models import Captioner

import os, json
import csv
import pandas as pd

from tqdm import tqdm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_ID = SPECIALS['<pad>']; BOS_ID = SPECIALS['<bos>']; EOS_ID = SPECIALS['<eos>']

checkpoint = torch.load('sat_csv_biology/sat_model_21.pt', map_location='cuda')

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

# train_df2, val_df, test_df = get_flickr_splits()

vocab = Vocab(cfg.MIN_WORD_FREQ).build(list(train_df2['caption']))

# train_ds = CaptionCSV(train_df2, image_dir, vocab, cfg.MAX_LEN, train_split=True)
# val_ds   = CaptionCSV(val_df,   image_dir, vocab, cfg.MAX_LEN, train_split=False)
# test_ds  = CaptionCSV(test_df,  image_dir,  vocab, cfg.MAX_LEN, train_split=False)
train_ds = CaptionCSV(train_df2, train_images, vocab, cfg.MAX_LEN, train_split=True)
val_ds   = CaptionCSV(val_df,   val_images, vocab, cfg.MAX_LEN, train_split=False)
test_ds  = CaptionCSV(test_df,  test_images,  vocab, cfg.MAX_LEN, train_split=False)

model = Captioner(len(vocab.w2i)).to(DEVICE)
model.load_state_dict(checkpoint['model'])

def generate_captions(model, dataset: CaptionCSV):
    model.eval()
    preds = {}
    refs = {}
    for i, (img, cap_ids) in enumerate(tqdm(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))):
        # if (i+1) % 100 == 0:
        #     print(i+1)
        img = img.to(DEVICE)
        V = model.enc(img)
        seq = model.dec.beam_search(V, BOS_ID, EOS_ID, beam=cfg.BEAM_SIZE, max_len=cfg.MAX_LEN)
        hyp = vocab.decode(seq[1:])
        preds[str(i)] = [hyp]
        refs[str(i)] = [vocab.decode(cap_ids[0].tolist()[1:])]
    return preds, refs

preds, refs = generate_captions(model, test_ds)
with open(os.path.join(cfg.OUT_DIR,'preds.json'),'w') as f: json.dump(preds, f)
with open(os.path.join(cfg.OUT_DIR,'refs.json'),'w') as f: json.dump(refs, f)

# Compute metrics
bleu = Bleu(4); meteor = Meteor(); rouge = Rouge(); cider = Cider()
bleu_scores, _  = bleu.compute_score(refs, preds)
meteor_score, _ = meteor.compute_score(refs, preds)
rouge_score, _  = rouge.compute_score(refs, preds)
cider_score, _  = cider.compute_score(refs, preds)
spice_score = None

summary = {
    'BLEU-1': bleu_scores[0],
    'BLEU-2': bleu_scores[1],
    'BLEU-3': bleu_scores[2],
    'BLEU-4': bleu_scores[3],
    'METEOR': meteor_score,
    'ROUGE-L': rouge_score,
    'CIDEr': cider_score,
}

print('\nEvaluation summary:')
for k,v in summary.items():
    print(f"{k:>7}: {v:.4f}")

# Save summary
with open(os.path.join(cfg.OUT_DIR,'metrics.json'),'w') as f: json.dump({k: float(v) for k,v in summary.items()}, f)
print('\nArtifacts saved in', cfg.OUT_DIR)