from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from dataset import datasets
import os

DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MD_REVISION = "2024-05-20"
EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
LR = 1e-5
USE_WANDB = True
IMG_TOKENS = 729
ANSWER_EOS = "<|endoftext|>"
VAL_STEPS = 2000


tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

def collate_fn(batch):
    images = [sample['image'] for sample in batch]
    images = [img.convert("RGB") if hasattr(img, "mode") and img.mode != "RGB" else img for img in images]
    images = [moondream.vision_encoder.preprocess(image) for image in images]

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for qa in sample['qa']:
            q_t = tokenizer(
                f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = tokenizer(
                f" {qa['answer']}{ANSWER_EOS}",
                add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = -1
    for labels in labels_acc:
        max_len = max(max_len, len(labels))

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


def compute_loss(batch):
    images, tokens, labels, attn_mask = batch

    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder(images)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss

def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            loss = compute_loss(batch)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count

# --- Setup ---
dataloaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    ),
    "validation": DataLoader(
        datasets["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
}

moondream.text_model.train()
moondream.text_model.transformer.gradient_checkpointing_enable()

total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
optimizer = Adam8bit(
    [{"params": moondream.text_model.parameters()}],
    lr=LR * 0.1,
    betas=(0.9, 0.95),
    eps=1e-6
)

if USE_WANDB:
    import wandb
    wandb.init(
        project="Captioning-17k",
        config={
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        }
    )


best_val_loss = float('inf')
i = 0

try:
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1

            loss = compute_loss(batch)
            loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if USE_WANDB:
                wandb.log({
                    "loss/train": loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                })

            
            if i % VAL_STEPS == 0:
                val_loss = evaluate(moondream.text_model, dataloaders["validation"])
                print(f"Validation Loss (step {i}): {val_loss:.4f}")

                if USE_WANDB:
                    wandb.log({"loss/val": val_loss})
                
                
                if val_loss < best_val_loss:
                    print(f"New best validation loss: {val_loss:.4f}. Saving checkpoint...")
                    best_val_loss = val_loss
                    moondream.save_pretrained("checkpoints/moondream-best")

    
    print("Training finished. Saving final model...")
    moondream.save_pretrained("checkpoints/moondream-final")

except KeyboardInterrupt:
    print("\nTraining interrupted by user!")
    print("Saving current state to 'checkpoints/moondream-interrupted'...")
    moondream.save_pretrained("checkpoints/moondream-interrupted")
    if USE_WANDB:
        wandb.finish()
    print("Exited gracefully.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    if USE_WANDB:
        wandb.finish()
    raise e

if USE_WANDB:
    wandb.finish()