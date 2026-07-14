"""Attention-MIL downstream target classifier — replaces mean-pool-over-contexts with LEARNED aggregation.

Each protein is a BAG of its context-specific embeddings {z_g,c} (one per context it appears in, e.g. crohn_macrophage,
crohn_fibroblast, ...). A gated-attention pooler (Ilse et al. 2018) learns weights over the bag -> one vector ->
head -> P(drug target). Trained on all-OT membership, 5-fold OOF over proteins. Reports H@10/H@100/MRR and compares to
(a) MEAN-pool of the same bag -> MLP, and (b) ESM -> MLP (the frozen-sequence ceiling).

Hypothesis: if attention beats mean-pool, the context-specific info is useful and averaging was destroying it.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/mil_attention_target.py --run link_v12 --epochs 40
Out: results/<run>/mil_attention_target.tsv  (per-protein OOF scores + label)
"""
from __future__ import annotations
import argparse, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

SEQ = Path("mlp_mods/seq_context")
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())


class GatedAttnMIL(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.V = nn.Linear(d, h); self.U = nn.Linear(d, h); self.w = nn.Linear(h, 1)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, X, mask):                              # X:(B,T,d) mask:(B,T) bool
        a = self.w(torch.tanh(self.V(X)) * torch.sigmoid(self.U(X))).squeeze(-1)   # (B,T)
        a = a.masked_fill(~mask, -1e9)
        a = torch.softmax(a, 1)
        z = (a.unsqueeze(-1) * X).sum(1)                     # (B,d) attention-weighted bag vector
        return self.head(z).squeeze(-1)


def rank_metrics(score, genes, pos):
    order = genes[np.argsort(-score)]; rk = {g: i + 1 for i, g in enumerate(order)}
    P = [g for g in genes if g in pos]
    if not P:
        return 0, 0, float("nan")
    return (sum(rk[g] <= 10 for g in P), sum(rk[g] <= 100 for g in P), float(np.mean([1 / rk[g] for g in P])))


def train_mil(bags, y, tr, te, d, device, epochs, lr=1e-3):
    m = GatedAttnMIL(d).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    pw = torch.tensor([(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    tr = list(tr); bs = 128
    for ep in range(epochs):
        m.train(); np.random.shuffle(tr)
        for i in range(0, len(tr), bs):
            idx = tr[i:i + bs]
            T = max(bags[j].shape[0] for j in idx)
            X = torch.zeros(len(idx), T, d); mask = torch.zeros(len(idx), T, dtype=torch.bool)
            for r, j in enumerate(idx):
                b = bags[j]; X[r, :b.shape[0]] = torch.from_numpy(b); mask[r, :b.shape[0]] = True
            X, mask = X.to(device), mask.to(device)
            yt = torch.tensor(y[idx], dtype=torch.float32, device=device)
            logit = m(X, mask); loss = lossf(logit, yt)
            opt.zero_grad(); loss.backward(); opt.step()
    # predict te
    m.eval(); out = np.zeros(len(te), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(te), bs):
            idx = te[i:i + bs]
            T = max(bags[j].shape[0] for j in idx)
            X = torch.zeros(len(idx), T, d); mask = torch.zeros(len(idx), T, dtype=torch.bool)
            for r, j in enumerate(idx):
                b = bags[j]; X[r, :b.shape[0]] = torch.from_numpy(b); mask[r, :b.shape[0]] = True
            out[i:i + len(idx)] = torch.sigmoid(m(X.to(device), mask.to(device))).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v12")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = set()
    for f in glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"):
        pos |= set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))

    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    E = d["emb"].astype(np.float32); idx = d["prot_idx"]; dim = E.shape[1]
    # standardize features (global; exploratory)
    mu, sd = E.mean(0), E.std(0) + 1e-6; E = (E - mu) / sd
    perprot = defaultdict(list)
    for j in range(len(idx)):
        perprot[PROT[idx[j]]].append(E[j])
    genes = np.array([g for g in sorted(perprot) if g in ESM_ALL])
    bags = [np.stack(perprot[g]).astype(np.float32) for g in genes]
    y = np.array([1 if g in pos else 0 for g in genes], dtype=np.int64)
    meanX = np.stack([b.mean(0) for b in bags]).astype(np.float64)   # mean-pool baseline features
    esmX = np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
    print(f"{args.run}: {len(genes)} proteins | targets={y.sum()} | dim={dim} | device={device} | "
          f"bag sizes med={int(np.median([b.shape[0] for b in bags]))} max={max(b.shape[0] for b in bags)}", flush=True)

    skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    mil = np.zeros(len(genes), dtype=np.float32)
    for k, (tr, te) in enumerate(skf.split(genes, y)):
        mil[te] = train_mil(bags, y, tr, te, dim, device, args.epochs)
        print(f"  fold {k+1}/5 done", flush=True)

    def mlp():
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    meanP = cross_val_predict(mlp(), meanX, y, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed), method="predict_proba")[:, 1]
    esmP = cross_val_predict(mlp(), esmX, y, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed), method="predict_proba")[:, 1]

    print(f"\n{'method':22s} {'H@10':>5s} {'H@100':>6s} {'MRR':>8s}")
    for name, sc in [("MIL-attention (EMB)", mil), ("mean-pool (EMB)", meanP), ("ESM", esmP)]:
        h10, h100, mrr = rank_metrics(np.asarray(sc), genes, pos)
        print(f"{name:22s} {h10:5d} {h100:6d} {mrr:8.4f}", flush=True)
    pd.DataFrame({"protein": genes, "label": y, "mil_emb": mil, "mean_emb": meanP, "esm": esmP}) \
        .sort_values("mil_emb", ascending=False).to_csv(SEQ / "results" / args.run / "mil_attention_target.tsv", sep="\t", index=False)
    print(f"\nwrote {SEQ/'results'/args.run/'mil_attention_target.tsv'}", flush=True)


if __name__ == "__main__":
    main()
