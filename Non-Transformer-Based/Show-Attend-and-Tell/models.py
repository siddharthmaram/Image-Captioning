import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import cfg

class EncoderCNN(nn.Module):
    def __init__(self, backbone='resnet50', freeze=True):
        super().__init__()
        if backbone=='resnet101':
            net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(net.children())[:-2]  # keep conv5 features
        self.cnn = nn.Sequential(*modules)
        self.adapt = nn.Conv2d(2048, 2048, kernel_size=1)
        if freeze:
            for p in self.cnn.parameters(): p.requires_grad=False

    def forward(self, x):
        feats = self.cnn(x)               # (B,2048,H=7,W=7)
        feats = self.adapt(feats)
        B,C,H,W = feats.size()
        feats = feats.view(B, C, -1).permute(0,2,1)  # (B,49,2048)
        return feats

class AdditiveAttention(nn.Module):
    def __init__(self, hdim, vdim, attdim):
        super().__init__()
        self.W_h = nn.Linear(hdim, attdim)
        self.W_v = nn.Linear(vdim, attdim)
        self.v = nn.Linear(attdim, 1)
    def forward(self, h, V):
        # h: (B,H), V: (B,L,C)
        q = self.W_h(h).unsqueeze(1)     # (B,1,A)
        k = self.W_v(V)                  # (B,L,A)
        e = self.v(torch.tanh(q + k)).squeeze(-1)  # (B,L)
        a = torch.softmax(e, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), V).squeeze(1)  # (B,C)
        return ctx, a

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, att_dim, vdim=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.att = AdditiveAttention(hidden_dim, vdim, att_dim)
        self.lstm = nn.LSTMCell(embed_dim + vdim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, V, caps):
        B,L,C = V.size()
        T = caps.size(1)
        h = torch.zeros(B, self.lstm.hidden_size, device=V.device)
        c = torch.zeros(B, self.lstm.hidden_size, device=V.device)
        logits = []
        emb = self.embed(caps)
        for t in range(T-1):
            yprev = emb[:,t,:]
            ctx,_ = self.att(h,V)
            h,c = self.lstm(torch.cat([yprev, ctx], dim=-1), (h,c))
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def beam_search(self, V, bos_id, eos_id, beam=3, max_len=30):
        assert V.size(0)==1
        device = V.device
        h = torch.zeros(1, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, self.lstm.hidden_size, device=device)
        beams=[(0.0, [bos_id], h, c)]
        finished=[]
        for _ in range(max_len):
            new=[]
            for logp, toks, h, c in beams:
                if toks[-1]==eos_id: finished.append((logp,toks)); continue
                yprev = self.embed(torch.tensor([toks[-1]], device=device))
                ctx,_ = self.att(h, V)
                h,c = self.lstm(torch.cat([yprev.squeeze(0), ctx.squeeze(0)], dim=-1).unsqueeze(0),(h,c))
                lp = F.log_softmax(self.out(h), dim=-1).squeeze(0)
                topk = torch.topk(lp, beam)
                for k in range(beam):
                    new.append((logp+float(topk.values[k]), toks+[int(topk.indices[k])], h, c))
            new.sort(key=lambda x:x[0], reverse=True)
            beams = new[:beam]
            if len(finished)>=beam: break
        finished += beams
        finished.sort(key=lambda x:x[0], reverse=True)
        return finished[0][1]

class Captioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.enc = EncoderCNN(cfg.BACKBONE, cfg.FREEZE_ENCODER)
        self.dec = DecoderLSTM(vocab_size, cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.ATT_DIM)
    def forward(self, imgs, caps):
        V = self.enc(imgs)
        return self.dec(V, caps)

