import os
import sys
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from peft import PeftModel

# --- Environment & GPU Check ---
print("--- System Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CRITICAL ERROR: PyTorch cannot find a CUDA-enabled GPU.")
    print("Please ensure you have installed the CUDA-compiled version of PyTorch.")
    sys.exit(1) # Stops the script immediately so it doesn't hang on the CPU
else:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("--------------------\n")

# --- Paths and Config ---
MODEL_ID = "llava-hf/llama3-llava-next-8b-hf" 
OUTPUT_DIR = "llava-llama3-next-8b-caption-beginner" # MUST MATCH YOUR FINETUNE
DATASET_ROOT = "BioKosh17k-Cleaned"
TEST_DIR = os.path.join(DATASET_ROOT, "test")
TEST_CSV = os.path.join(DATASET_ROOT, "beginner_csv", "test.csv")
INSTRUCTION = "Describe the image in a paragraph"

# Explicitly define our device
device = "cuda"

# --- Load Test Data ---
print("Loading test dataset...")
df_test = pd.read_csv(TEST_CSV)
df_test = df_test.rename(columns={"file_name": "image", "text": "caption"})
df_test["image"] = df_test["image"].apply(lambda x: os.path.join(TEST_DIR, x))
df_test = df_test[df_test["image"].apply(os.path.exists)].reset_index(drop=True)
test_rows = df_test[["image", "caption"]].to_dict(orient="records")
print(f"Found {len(test_rows)} images for testing.\n")

# --- Load Base Model + LoRA Adapter ---
print("Loading model and adapters (this may take a moment)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# Force device map to GPU 0 to prevent silent CPU offloading
base = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map={"": "cuda"} 
)

model = PeftModel.from_pretrained(base, OUTPUT_DIR).eval()

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left" 

# --- Helper functions ---
def build_prompt(instruction: str):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def generate_captions(rows, batch_size=2, max_new_tokens=200): 
    preds, refs = [], []
    for i in tqdm(range(0, len(rows), batch_size)):
        batch = rows[i:i+batch_size]
        images = [Image.open(x["image"]).convert("RGB") for x in batch]
        prompts = [build_prompt(INSTRUCTION) for _ in batch]
        
        # Explicitly move processed inputs to the CUDA device
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # Decode only the generated part
        input_len = inputs.input_ids.shape[1]
        decoded_texts = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)

        for text, row in zip(decoded_texts, batch):
            preds.append(text.strip())
            refs.append(row["caption"])
            
    return preds, refs

# --- Run Inference ---
print("\nStarting inference...")
preds, refs = generate_captions(test_rows, batch_size=2)

# --- Save Results ---
results_df = pd.DataFrame({"image": df_test["image"], "reference": refs, "prediction": preds})
output_csv = "test_results_beginner.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nInference complete. Results saved to {output_csv}")