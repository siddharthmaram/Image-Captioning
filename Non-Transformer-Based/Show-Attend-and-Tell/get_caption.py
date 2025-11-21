import json
import csv
from bert_score import score


images = ['391.jpeg', '1.png', '137.png', '235.png', '378.png', '580.png', 
            '715.png', '817.png', '879.png', '2152.png', '2403.jpeg', 
            '2807.jpeg', '4075.jpeg', '4099.png', '4444.png', '4531.png', 
            '4672.jpeg', '6143.jpeg', '6308.jpeg', '6896.jpeg', 
            '7068.png', '7539.png', '8032.png', '8433.jpeg']

with open("sat_csv_biology/preds.json", "r") as f:
    captions = json.load(f)
    generated = []
    for num, caption in captions.items():
        generated.append(caption[0])

    
with open("/home/maramreddy/dataset/Biology-Dataset/test/metadata.csv", "r") as f:
    reader = csv.reader(f)
    _ = next(reader)

    results = []
    gts = []

    for ind, (img, caption) in enumerate(reader):
        gts.append(caption)
        if img in images:
            results.append((img, captions[str(ind)][0]))

results.sort(key=lambda x: images.index(x[0]))

for img, caption in results:
    print(f"{img}: {caption}")

# Compute BERTScore (precision, recall, F1)
P, R, F1 = score(generated, gts, lang="en", verbose=True)

print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")


