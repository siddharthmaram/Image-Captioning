from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from bert_score import score
import csv

with open("/home/maramreddy/dataset/Biology-Dataset/test/metadata.csv", "r") as f:
    reader = csv.reader(f)
    _ = next(reader)
    d1 = {}
    
    for image, caption in reader:
        d1[image] = caption.lower()

with open("preds_sat.csv", "r") as f:
    reader = csv.reader(f)
    _ = next(reader)
    d2 = {}
    
    for image, caption in reader:
        d2[image] = caption.lower()

gts = {}
gts1 = []
res = {}
res1 = []

for ind, img in enumerate(d1.keys()):
    gts[str(ind)] = [d1[img]]
    res[str(ind)] = [d2[img]]
    gts1.append(d1[img])
    res1.append(d2[img])

# ============ Metrics ============ #
print("\n==== Evaluation Results ====")

# BLEU-1 to BLEU-4
try:
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    print(f"BLEU-1: {bleu_score[0]:.4f}")
    print(f"BLEU-2: {bleu_score[1]:.4f}")
    print(f"BLEU-3: {bleu_score[2]:.4f}")
    print(f"BLEU-4: {bleu_score[3]:.4f}")
except Exception as e:
    print(f"Error computing BLEU: {e}")

# ROUGE-L
try:
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    print(f"ROUGE-L: {rouge_score:.4f}")
except Exception as e:
    print(f"Error computing ROUGE: {e}")

# CIDEr
try:
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    print(f"CIDEr: {cider_score:.4f}")
except Exception as e:
    print(f"Error computing CIDEr: {e}")

# SacreBLEU
try:
    # Convert dict to lists for sacreBLEU
    hypotheses = [res[i][0] for i in sorted(res.keys())]
    references = [[gts[i][0] for i in sorted(gts.keys())]]  # single reference per example

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"BLEU (SacreBLEU): {bleu.score:.2f}")
except Exception as e:
    print(f"SacreBLEU not available: {e}")

# Compute BERTScore (precision, recall, F1)
P, R, F1 = score(res1, gts1, lang="en", verbose=True)

print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")
