import os, time
import csv
from google import genai
from google.genai import types
from tqdm import tqdm

TEST_PATH = "/home/maramreddy/dataset/Biology-Dataset/test"

test_file = os.path.join(TEST_PATH, "metadata.csv")

with open(test_file, "r") as f:
    reader = csv.reader(f)
    _ = next(reader)
    images = [img for img, caption in list(reader)]

client = genai.Client(api_key="AIzaSyAEDbB5c172NSLzRqMuL3rZ5m_kV5dQQU0")

def generate_caption(img):
    """Generate caption for a single image, with retry on 429 errors."""
    while True:
        try:
            with open(os.path.join(TEST_PATH, img), 'rb') as f:
                image_bytes = f.read()

            ext = "png" if img.endswith("png") else "jpeg"

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=f'image/{ext}',
                    ),
                    'Describe this image in one paragraph.'
                ]
            )
            return [img, response.text]

        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                # Extract suggested retry time if available
                wait_time = 30
                if "Please retry in" in msg:
                    try:
                        wait_time = float(msg.split("Please retry in")[1].split("s")[0])
                    except:
                        pass
                print(f"Rate limit hit. Waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time + 1)
            else:
                return [img, f"Error: {e}"]

generated_captions = []

for img in tqdm(images[:10]):
    generated_captions.append(generate_caption(img))

# Sort results by filename to keep order consistent
generated_captions.sort(key=lambda x: x[0])

with open("preds_gemini.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(generated_captions)
