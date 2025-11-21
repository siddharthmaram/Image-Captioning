from collections import Counter
import torch
from typing import List
import nltk

SPECIALS = {'<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3}

class Vocab:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.w2i = dict(SPECIALS)
        self.i2w = {i:w for w,i in self.w2i.items()}
        self.freq = Counter()

    def tokenize(self, s):
        return [w.lower() for w in nltk.word_tokenize(str(s))]

    def build(self, captions: List[str]):
        for cap in captions:
            self.freq.update(self.tokenize(cap))
        for w,c in self.freq.items():
            if c >= self.min_freq and w not in self.w2i:
                idx = len(self.w2i); self.w2i[w]=idx; self.i2w[idx]=w
        return self

    def encode(self, s, max_len):
        toks = self.tokenize(s)
        ids = [self.w2i['<bos>']]
        for t in toks[:max_len-2]:
            ids.append(self.w2i.get(t, self.w2i['<unk>']))
        ids.append(self.w2i['<eos>'])
        if len(ids) < max_len: ids += [self.w2i['<pad>']]*(max_len-len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        words=[]
        for i in ids:
            w=self.i2w.get(int(i),'<unk>')
            if w=='<eos>': break
            if w not in ('<pad>','<bos>'): words.append(w)
        return ' '.join(words)

# vocab = Vocab(cfg.MIN_WORD_FREQ).build(list(train_df2['caption']))
# with open(os.path.join(cfg.OUT_DIR,'vocab.json'),'w') as f: json.dump(vocab.w2i, f)
# print('Vocab size:', len(vocab.w2i))