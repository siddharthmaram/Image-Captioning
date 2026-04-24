from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from dataset import datasets
import csv
import os


DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-05-20"

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
base_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

state_dict = load_file("checkpoints/moondream-best/model.safetensors", device="cpu")


missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
print("Loaded fine-tuned weights.")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

moondream2 = base_model.eval().to(DEVICE)

results = []

for i, sample in tqdm(enumerate(datasets['test']), total=len(datasets['test'])):
    img = sample["image"]
    imgname = os.path.basename(img.filename)
    img = img.convert("RGB") if hasattr(img, "mode") and img.mode != "RGB" else img
    md_answer = moondream2.answer_question(
        moondream2.encode_image(img),
        sample['qa'][0]['question'],
        tokenizer=tokenizer,
        num_beams=4,
        no_repeat_ngram_size=5,
        early_stopping=True
    )

    results.append([imgname, md_answer])

# save to disk
with open("predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(results)

print("Saved predictions to predictions.csv")