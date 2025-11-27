from bert_score import score as bert_score
from rouge_score import rouge_scorer
import json

# Load JSON files
gts_file_path = 'data/dataset_flickr30k.json'
preds_file_path = '/home/sri/AoANet_final/AoANet/eval_results_flickr/.cache_aoanet_test.json'

with open(gts_file_path, 'r') as f:
    gts = json.load(f)
with open(preds_file_path, 'r') as f:
    preds = json.load(f)

# Extract ground truth captions and image ids
gts_captions = []
gts_imgids = []
for image_data in gts['images']:
    if image_data.get('split') == 'test':
        for sentence in image_data.get('sentences', []):
            gts_captions.append(sentence['raw'])
            gts_imgids.append(image_data['imgid'])

# Ensure both lists are sorted by image id for proper alignment
# Build a map from image_id to predicted caption
preds_map = {pred['image_id']: pred['caption'] for pred in preds}

# Now, for each ground-truth image id, get the prediction (empty if none)
preds_captions = [preds_map.get(imgid, "") for imgid in gts_imgids]
# print(gts_captions[1])
# print(preds_captions[1])
# print(gts_imgids[:10])
# print(list(preds_map.keys())[:10])

# --- METRICS ---

# BERTScore
P, R, F1 = bert_score(preds_captions, gts_captions, lang='en', rescale_with_baseline=True)
print(f"BERTScore - Precision: {P.mean().item():.4f}")
print(f"BERTScore - Recall:    {R.mean().item():.4f}")
print(f"BERTScore - F1 Score:  {F1.mean().item():.4f}")

# ROUGE-L
# scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# rouge_l_scores = [
#     scorer.score(ref, pred)['rougeL'].fmeasure
#     for ref, pred in zip(gts_captions, preds_captions)
# ]
# avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
# print(f"Average ROUGE-L F1 Score: {avg_rouge_l:.4f}")

# (OPTIONAL) CLIPScore -- requires `clip_score` from https://github.com/KaiyangZhou/CLIPScore 
# and image files.
# Example (uncomment and update paths if available):
# from clip_score import clip_score
# image_paths = ['path/to/test/image_{}.png'.format(imgid) for imgid in gts_imgids]
# clip_scores = clip_score(image_paths, preds_captions)
# avg_clip_score = sum(clip_scores) / len(clip_scores)
# print(f"Average CLIPScore: {avg_clip_score:.4f}")
