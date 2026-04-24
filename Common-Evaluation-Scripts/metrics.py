from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from bert_score import score
import csv
import collections

import re

NREFS = 1

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r'\s+([,.!?;:])', r'\1', s)
    s = re.sub(r'([,.!?;:])(?=[^\s])', r'\1 ', s)
    return s

gts = collections.defaultdict(list)
with open("/path/to/dataset/Captioning-17k/BioKosh17k-Cleaned/test/metadata.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for image, caption in reader:
        gts[image].append(normalize_string(caption))

res = {}
with open("path/to/predictions.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for image, caption in reader:
        res[image] = [normalize_string(caption)]

final_gts = {}
final_res = {}
for i, img_id in enumerate(res.keys()):
    if img_id in gts:
        final_gts[str(i)] = gts[img_id]
        final_res[str(i)] = res[img_id]

# ============ Metrics ============ #
print("\n==== Evaluation Results ====")

# BLEU-1 to BLEU-4
try:
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(final_gts, final_res)
    print(f"BLEU-1: {bleu_score[0]:.4f}")
    print(f"BLEU-2: {bleu_score[1]:.4f}")
    print(f"BLEU-3: {bleu_score[2]:.4f}")
    print(f"BLEU-4: {bleu_score[3]:.4f}")
except Exception as e:
    print(f"Error computing BLEU: {e}")

# ROUGE-L
try:
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(final_gts, final_res)
    print(f"ROUGE-L: {rouge_score:.4f}")
except Exception as e:
    print(f"Error computing ROUGE: {e}")

# CIDEr
try:
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(final_gts, final_res)
    print(f"CIDEr: {cider_score:.4f}")
except Exception as e:
    print(f"Error computing CIDEr: {e}")

# SacreBLEU
try:
    sorted_keys = sorted(final_res.keys(), key=int)
    hypotheses = [final_res[k][0] for k in sorted_keys]

    references = []
    for j in range(NREFS):
        ref_stream = [final_gts[k][j] for k in sorted_keys]
        references.append(ref_stream)

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"BLEU (SacreBLEU): {bleu.score:.2f}")
except Exception as e:
    print(f"SacreBLEU not available: {e}")


res_bert = [final_res[str(i)][0] for i in range(len(final_res))]
gts_bert = [final_gts[str(i)] for i in range(len(final_gts))]

P, R, F1 = score(
    res_bert, 
    gts_bert, 
    lang="en", 
    verbose=True
)

print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")
