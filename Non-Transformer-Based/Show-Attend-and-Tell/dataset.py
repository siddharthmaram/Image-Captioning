import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import Dataset
from vocabulary import Vocab
from config import cfg

class CaptionCSV(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, vocab: Vocab, max_len: int, train_split=True):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.vocab = vocab
        self.max_len = max_len
        if train_split:
            self.tfms = transforms.Compose([
                transforms.Resize((cfg.CROP_SIZE, cfg.CROP_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((cfg.CROP_SIZE, cfg.CROP_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, os.path.basename(str(row.file_name))) if os.path.dirname(str(row.file_name)) else os.path.join(self.img_root, str(row.file_name))
        # If CSV paths are like 'Images/123.jpg', respect that subdir under split dir
        if os.path.dirname(str(row.file_name)):
            img_path = os.path.join(os.path.dirname(self.img_root), str(row.file_name))
        # img = Image.open(img_path).convert('RGB')
        with Image.open(img_path) as im:
            if im.mode == "P":
                im = im.convert("RGBA")
            img = im.convert("RGB")
        img = self.tfms(img)
        cap_ids = self.vocab.encode(row.caption, cfg.MAX_LEN)
        return img, cap_ids


