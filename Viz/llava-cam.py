import os
import sys
import math
import re
import textwrap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter


# ============================================================
# 0) Environment & GPU Check
# ============================================================
print("--- System Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CRITICAL ERROR: PyTorch cannot find a CUDA-enabled GPU.")
    print("Install the CUDA-enabled PyTorch build and retry.")
    sys.exit(1)
else:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("--------------------\n")

device = torch.device("cuda")


# ============================================================
# 1) Paths and Config
# ============================================================
MODEL_ID     = "llava-hf/llama3-llava-next-8b-hf"
OUTPUT_DIR   = "llava-llama3-next-8b-caption-inter"
DATASET_ROOT = "BioKosh17k-Cleaned"
TEST_DIR     = os.path.join(DATASET_ROOT, "test")
TEST_CSV     = os.path.join(DATASET_ROOT, "beginner_csv", "test.csv")

INSTRUCTION = "Describe the image in a paragraph"

RESULTS_CSV = "test_results_cam_specific.csv"
CAM_DIR     = "llava_cam_outputs_inter"

MAX_NEW_TOKENS = 200
MAX_CAM_TARGET_TOKENS = None  # Use ALL generated tokens for saliency score

SAVE_RAW_HEATMAP = True

# ── Target images ─────────────────────────────────────────────
TARGET_IMAGES = ["4096.png", "17482.png", "8160.png"]
# ─────────────────────────────────────────────────────────────

# ── SmoothGrad config ─────────────────────────────────────────
SMOOTHGRAD_N   = 20
SMOOTHGRAD_STD = 0.015

# ── Post-processing config─────────────────────────
BLUR_SIGMA      = 3     
CLIP_PERCENTILE = 99    
OVERLAY_ALPHA   = 0.35 
MASK_BACKGROUND = True  
BG_THRESHOLD    = 0.8 
# ─────────────────────────────────────────────────────────────

os.makedirs(CAM_DIR, exist_ok=True)


# ============================================================
# 2) Load Test Data — filter by specific filenames
# ============================================================
print("Loading test dataset...")
df_test = pd.read_csv(TEST_CSV)
df_test = df_test.rename(columns={"file_name": "image", "text": "caption"})
df_test["image"] = df_test["image"].apply(lambda x: os.path.join(TEST_DIR, x))

df_test = df_test[
    df_test["image"].apply(os.path.basename).isin(TARGET_IMAGES)
].reset_index(drop=True)

found_names = set(df_test["image"].apply(os.path.basename))
for name in TARGET_IMAGES:
    if name not in found_names:
        print(f"  WARNING: '{name}' not found in CSV or does not exist on disk.")

df_test   = df_test[df_test["image"].apply(os.path.exists)].reset_index(drop=True)
test_rows = df_test[["image", "caption"]].to_dict(orient="records")

print(f"Running SmoothGrad saliency on {len(test_rows)} images:")
for r in test_rows:
    print(f"  {r['image']}")
print()


# ============================================================
# 3) Load Base Model + LoRA Adapter (fp16 for clean backprop)
# ============================================================
print("Loading model and adapters (this may take a moment)...")

base = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": 0},
)

model = PeftModel.from_pretrained(base, OUTPUT_DIR)
model = model.eval()
model.config.use_cache = False

if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
    print("Gradient checkpointing disabled for saliency compatibility.")

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"


# ============================================================
# 4) Prompt Helper
# ============================================================
def build_prompt(instruction: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


# ============================================================
# 5) Caption Generation (no_grad)
# ============================================================
@torch.no_grad()
def generate_caption(image: Image.Image, instruction: str, max_new_tokens=200):
    prompt = build_prompt(instruction)
    inputs = processor(
        images=[image], text=[prompt],
        return_tensors="pt", padding=True
    ).to(device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )

    input_len = inputs.input_ids.shape[1]
    decoded = processor.batch_decode(
        gen_ids[:, input_len:], skip_special_tokens=True
    )[0].strip()

    return decoded, prompt


# ============================================================
# 6) Single-pass gradient helper
# ============================================================
def _single_grad_pass(image, prompt, generated_text, pv_base, noise):
    full_inputs = processor(
        images=[image], text=[prompt + generated_text],
        return_tensors="pt", padding=True
    ).to(device)

    pv_noisy = (pv_base + noise).detach().requires_grad_(True)
    full_inputs["pixel_values"] = pv_noisy

    prompt_inputs = processor(
        images=[image], text=[prompt],
        return_tensors="pt", padding=True
    ).to(device)
    prompt_len = prompt_inputs.input_ids.shape[1]

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = model(**full_inputs, return_dict=True)

        target_ids = full_inputs.input_ids[:, prompt_len:]
        if target_ids.shape[1] == 0:
            return None

        if MAX_CAM_TARGET_TOKENS is not None:
            target_ids = target_ids[:, :MAX_CAM_TARGET_TOKENS]

        pred_logits = outputs.logits[
            :, prompt_len - 1: prompt_len - 1 + target_ids.shape[1], :
        ]
        score = torch.gather(pred_logits, 2, target_ids.unsqueeze(-1)).squeeze(-1).sum()

        if score.grad_fn is None:
            return None

        score.backward()

    return pv_noisy.grad   # [1, num_crops, C, H, W]  or  [1, C, H, W]


# ============================================================
# 7) SmoothGrad Saliency (crop 0 only — spatially aligned)
# ============================================================
def compute_smoothgrad(image: Image.Image, prompt: str, generated_text: str,
                        n_samples: int = SMOOTHGRAD_N,
                        noise_std: float = SMOOTHGRAD_STD):

    if not generated_text.strip():
        raise ValueError("Generated text is empty; cannot compute saliency.")

    base_inputs = processor(
        images=[image], text=[prompt + generated_text],
        return_tensors="pt", padding=True
    ).to(device)
    pv_base  = base_inputs["pixel_values"].detach()
    pv_range = pv_base.max() - pv_base.min()
    sigma    = noise_std * pv_range.item()

    is_anyres = (pv_base.ndim == 5)
    num_crops  = pv_base.shape[1] if is_anyres else 1
    print(f"    Pixel values shape: {list(pv_base.shape)} "
          f"({'AnyRes, using crop 0 only' if is_anyres else 'standard'})")

    accumulated = None
    successful  = 0

    print(f"    Running SmoothGrad ({n_samples} passes, "
          f"full caption tokens)...", end="", flush=True)

    for _ in range(n_samples):
        noise = torch.randn_like(pv_base) * sigma
        grad  = _single_grad_pass(image, prompt, generated_text, pv_base, noise)

        if grad is None:
            continue

        g = grad[0].abs().float()

        if is_anyres and g.ndim == 4:
            g = g[0]                    # crop 0 only — full-image resize
            g = g.max(dim=0)[0]         # max over RGB channels
        elif g.ndim == 3:
            g = g.max(dim=0)[0]
        else:
            continue

        accumulated = g if accumulated is None else accumulated + g
        successful += 1
        print(".", end="", flush=True)

    print(f" done ({successful}/{n_samples} passes succeeded)")

    if accumulated is None or successful == 0:
        raise RuntimeError("SmoothGrad: no gradient pass succeeded.")

    saliency = accumulated / successful
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency.cpu().numpy()


# ============================================================
# 8) Post-processing: blur + percentile clip + background mask
#    IMPROVED: sharper blur, higher clip percentile, optional
#              foreground masking to suppress white background
# ============================================================
def postprocess_saliency(saliency,
                          image_np=None,
                          blur_sigma: float = BLUR_SIGMA,
                          percentile: float = CLIP_PERCENTILE):
    """
    Args:
        saliency  : raw saliency array [H, W]
        image_np  : original image as np.uint8 [H, W, 3] (optional).
                    If provided and MASK_BACKGROUND=True, near-white
                    background pixels are suppressed.
        blur_sigma: Gaussian blur sigma (default 3, was 8).
        percentile: clip percentile (default 99, was 95).
    """
    saliency = gaussian_filter(saliency.astype(np.float32), sigma=blur_sigma)

    # ── Background masking (IMPROVEMENT) ──────────────────────
    if MASK_BACKGROUND and image_np is not None:
        # Resize image_np to match saliency if needed
        if image_np.shape[:2] != saliency.shape:
            img_pil  = Image.fromarray(image_np).resize(
                (saliency.shape[1], saliency.shape[0]), Image.LANCZOS
            )
            image_np_resized = np.array(img_pil)
        else:
            image_np_resized = image_np

        gray     = image_np_resized.mean(axis=-1) / 255.0          # [H, W]
        fg_mask  = (gray < BG_THRESHOLD).astype(np.float32)        # 1=foreground
        fg_mask  = gaussian_filter(fg_mask, sigma=2)               # soft edges
        saliency = saliency * fg_mask
    # ──────────────────────────────────────────────────────────

    vmax = np.percentile(saliency, percentile)
    if vmax > 0:
        saliency = np.clip(saliency, 0, vmax) / vmax
    return saliency.astype(np.float32)


# ============================================================
# 9) Visualization Helpers
#    IMPROVED: LANCZOS resampling for sharper upscaling,
#              reduced alpha for cleaner overlay
# ============================================================
def saliency_to_heatmap(saliency_array):
    heat = (cm.jet(saliency_array)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(heat)


def blend_image_and_heatmap(image, heatmap, alpha=OVERLAY_ALPHA):
    """
    IMPROVED: uses Image.LANCZOS (was BILINEAR) for sharper upscaling,
              and lower alpha=0.35 (was 0.45) so the diagram stays visible.
    """
    return Image.blend(
        image.convert("RGB"),
        heatmap.resize(image.size, Image.LANCZOS).convert("RGB"),  # LANCZOS
        alpha=alpha
    )


def make_triptych(image, heatmap, overlay, pred_text, ref_text):
    image   = image.convert("RGB")
    heatmap = heatmap.convert("RGB")
    overlay = overlay.convert("RGB")

    w, h     = image.size
    footer_h = 180
    canvas   = Image.new("RGB", (w * 3, h + footer_h), (255, 255, 255))
    canvas.paste(image,                                    (0,   0))
    canvas.paste(heatmap.resize((w, h), Image.LANCZOS),   (w,   0))  # LANCZOS
    canvas.paste(overlay,                                  (2*w, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((20,       15), "Original",                          fill=(0, 0, 0))
    draw.text((w + 20,   15), f"SmoothGrad (n={SMOOTHGRAD_N}, full caption)",
                                                                    fill=(0, 0, 0))
    draw.text((2*w + 20, 15), "Overlay",                           fill=(0, 0, 0))

    line_y = h + 10
    for line in textwrap.wrap(f"Prediction: {pred_text}", width=120)[:4]:
        draw.text((20, line_y), line, fill=(0, 0, 0));    line_y += 24
    line_y += 8
    for line in textwrap.wrap(f"Reference:  {ref_text}", width=120)[:4]:
        draw.text((20, line_y), line, fill=(60, 60, 60)); line_y += 24

    return canvas


# ============================================================
# 10) Main Loop
# ============================================================
results = []
print(f"\nStarting caption generation + SmoothGrad saliency "
      f"for {len(test_rows)} images...\n")

for idx, row in enumerate(tqdm(test_rows, total=len(test_rows))):
    image_path  = row["image"]
    ref_caption = row["caption"]

    print(f"\n[{idx+1}/{len(test_rows)}] {os.path.basename(image_path)}")

    try:
        image    = Image.open(image_path).convert("RGB")
        image_np = np.array(image)   # kept for background masking

        pred_caption, prompt = generate_caption(image, INSTRUCTION, MAX_NEW_TOKENS)
        print(f"  Prediction ({len(pred_caption.split())} words): {pred_caption[:80]}...")

        saliency = compute_smoothgrad(image, prompt, pred_caption)

        # IMPROVED: pass image_np so background can be masked
        saliency = postprocess_saliency(saliency, image_np=image_np)

        print(f"  Saliency map: {saliency.shape}, "
              f"range [{saliency.min():.3f}, {saliency.max():.3f}]")

        heatmap = saliency_to_heatmap(saliency)
        overlay = blend_image_and_heatmap(image, heatmap)
        panel   = make_triptych(image, heatmap, overlay, pred_caption, ref_caption)

        stem         = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(CAM_DIR, f"{stem}_overlay.png")
        panel_path   = os.path.join(CAM_DIR, f"{stem}_panel.png")
        heatmap_path = os.path.join(CAM_DIR, f"{stem}_heatmap.png")

        overlay.save(overlay_path)
        panel.save(panel_path)
        if SAVE_RAW_HEATMAP:
            heatmap.save(heatmap_path)

        print(f"  Saved → {panel_path}")

        results.append({
            "image":       image_path,
            "reference":   ref_caption,
            "prediction":  pred_caption,
            "cam_overlay": overlay_path,
            "cam_panel":   panel_path,
            "cam_heatmap": heatmap_path if SAVE_RAW_HEATMAP else "",
        })

    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({
            "image": image_path, "reference": ref_caption,
            "prediction": "", "cam_overlay": "",
            "cam_panel": "", "cam_heatmap": "", "error": str(e),
        })

pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print(f"\nResults saved   → {RESULTS_CSV}")
print(f"Saliency panels → {CAM_DIR}/")