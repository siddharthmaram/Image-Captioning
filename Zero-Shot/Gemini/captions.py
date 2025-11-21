""" 
First setup Vertex API account.
Run these commands to login:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform
"""

from google import genai
from google.genai.types import GenerateContentConfig, Part
from google.api_core.exceptions import TooManyRequests, InternalServerError

import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
INPUT_CSV = "test/metadata.csv"      # CSV file with column: image
OUTPUT_CSV = "image_captions2.csv"

MODEL_ID = (
    "projects/gemini-benchmarking-476114"
    "/locations/us-central1"
    "/publishers/google"
    "/models/gemini-2.5-flash"
)

PROMPT = "Strictly compose a single, small, concise paragraph of 3 to 4 lines describing this labeled biological diagram. Identify all key biological structures, organelles, molecules, or entities indicated by the labels or annotations. Explicitly reference these labels and explain their spatial relationships, interactions, and functions with precise biological terminology. Ensure the description is succinct, informative, and scientifically accurate, without extra elaboration or multiple paragraphs."


# Vertex AI Call Functions
def call_with_inline(client, image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    image_part = Part.from_bytes(
        data=img_bytes,
        mime_type="image/jpeg" if image_path.endswith("jpg") or image_path.endswith("jpeg") else "image/png"
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[PROMPT, image_part]
    )
    return response.text.strip()

def safe_call(image_path):
    retries = 3
    for attempt in range(retries):
        try:
            client = genai.Client(vertexai=True, project="gemini-benchmarking-476114", location="us-central1")
            return call_with_inline(client, image_path)
        except (TooManyRequests, InternalServerError) as e:
            wait = 2 ** attempt
            print(f"[WARN] API error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Non-retryable error for {image_path}: {e}")
            break
    return None


# Image Processing Function
def process_image(image_path):
    caption = safe_call(os.path.join("test", image_path))
    if caption:
        return (image_path, caption)
    return None


# Main Loop
if __name__ == "__main__":
    max_threads = 8

    # Read image paths from CSV
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)  # skip header if present
        image_paths = [row[0] for row in reader]

    # Process images and write results
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as write_file:
        writer = csv.writer(write_file)
        writer.writerow(["image", "caption"])

        futures = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for image_path in image_paths: 
                futures.append(executor.submit(process_image, image_path))

            for future in tqdm(as_completed(futures), total=len(image_paths)):
                result = future.result()
                if result:
                    writer.writerow([result[0], result[1]])

