"""v16 — same input as v15 (learned [protein-ID, disease, tissue, cell-type, state] embeddings, NO ESM), but the
OUTPUT is the protein's EXPRESSION VALUE in that context (MSE), instead of its network neighbors.

Target = scenic/expression_activity.tsv[context, gene] (per-gene z-scored mean log-expression). So the label
embeddings must learn per-context expression; disease/tissue/state carry the differential-expression modulation.
Macrophage only. Then Crohn disease->healthy z-shift -> top genes (expression-flavored drivers/markers, NOT TF-biased).

Out: results/link_v16/embeddings.npz. Run: .venv_scvi/bin/python mlp_mods/seq_context/scripts/train_v16_expression.py
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEQ = Path("mlp_mods/seq_context")
PID_DIM, CTX_DIM, LATENT, HIDDEN, EPOCHS, BS, SEED = 128, 32, 64, 256, 60, 4096, 0
DIS_TISSUES = ("colon", "ileum")


class V16(nn.Module):
    def __init__(self, nP, nD, nT, nC, nS):
        super().__init__()
        self.p = nn.Embedding(nP, PID_DIM); self.d = nn.Embedding(nD, CTX_DIM)
        self.t = nn.Embedding(nT, CTX_DIM); self.c = nn.Embedding(nC, CTX_DIM); self.s = nn.Embedding(nS, CTX_DIM)
        self.enc = nn.Sequential(nn.Linear(PID_DIM + 4 * CTX_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def embed(self, p, d, t, c, s):
        return self.enc(torch.cat([self.p(p), self.d(d), self.t(t), self.c(c), self.s(s)], -1))

    def forward(self, p, d, t, c, s):
        return self.dec(self.embed(p, d, t, c, s)).squeeze(-1)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = pd.read_csv(SEQ / "scenic/expression_activity.tsv", sep="\t", index_col=0)
    A = A.loc[[t for t in A.index if "macrophage" in t]]                    # macrophage contexts only
    genes = list(A.columns); gv = {g: i for i, g in enumerate(genes)}
    def parts(t): a = t.split("_"); return a[0], a[1], a[2], a[3]
    dv = {v: i for i, v in enumerate(sorted({parts(t)[0] for t in A.index}))}
    tv = {v: i for i, v in enumerate(sorted({parts(t)[1] for t in A.index}))}
    cv = {v: i for i, v in enumerate(sorted({parts(t)[2] for t in A.index}))}
    sv = {v: i for i, v in enumerate(sorted({parts(t)[3] for t in A.index}))}
    pid, did, tid, cid, sid, y, ctx, prot = [], [], [], [], [], [], [], []
    for t in A.index:
        dd, tt, cc, ss = parts(t); row = A.loc[t]
        for g in genes:
            v = row[g]
            if np.isfinite(v):
                pid.append(gv[g]); did.append(dv[dd]); tid.append(tv[tt]); cid.append(cv[cc]); sid.append(sv[ss])
                y.append(v); ctx.append(t); prot.append(g)
    pid = np.array(pid); did = np.array(did); tid = np.array(tid); cid = np.array(cid); sid = np.array(sid)
    y = np.array(y, np.float32); ctx = np.array(ctx); prot = np.array(prot); r = len(y)
    print(f"v16: samples={r}, genes={len(genes)}, contexts={len(A.index)}", flush=True)

    m = V16(len(genes), len(dv), len(tv), len(cv), len(sv)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); lossf = nn.MSELoss()
    P = torch.tensor(pid, device=dev); D = torch.tensor(did, device=dev); T = torch.tensor(tid, device=dev)
    C = torch.tensor(cid, device=dev); S = torch.tensor(sid, device=dev); Y = torch.tensor(y, device=dev)
    idx = np.arange(r)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); tot = 0.0
        for s0 in range(0, r, BS):
            b = torch.tensor(idx[s0:s0 + BS], device=dev)
            loss = lossf(m(P[b], D[b], T[b], C[b], S[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 15 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1}/{EPOCHS} mse {tot/(r//BS+1):.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        pred = np.zeros(r, np.float32); Z = np.zeros((r, LATENT), np.float32)
        for s0 in range(0, r, 8192):
            e = min(s0 + 8192, r)
            pred[s0:e] = m(P[s0:e], D[s0:e], T[s0:e], C[s0:e], S[s0:e]).cpu().numpy()
            Z[s0:e] = m.embed(P[s0:e], D[s0:e], T[s0:e], C[s0:e], S[s0:e]).cpu().numpy()
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"expression prediction R^2 (from labels only) = {r2:.3f}", flush=True)
    out = SEQ / "results/link_v16"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz", emb=Z, context=ctx, prot_idx=prot,
             disease=np.array([t.split("_")[0] for t in ctx]), tissue=np.array([t.split("_")[1] for t in ctx]),
             state=np.array([t.split("_")[3] for t in ctx]))
    print(f"saved -> {out/'embeddings.npz'}", flush=True)

    # Crohn disease->healthy shift (expression-flavored)
    Zd = Z.astype(np.float64)
    isc = np.array([t.startswith("crohn_") and any(x in t for x in DIS_TISSUES) for t in ctx])
    ish = np.array([t.startswith("healthy_") and any(x in t for x in DIS_TISSUES) for t in ctx])
    def pool(mask):
        zc = defaultdict(list); ec = defaultdict(list)
        for k in np.where(mask)[0]: zc[prot[k]].append(Zd[k]); ec[prot[k]].append(y[k])
        return {g: np.mean(v, 0) for g, v in zc.items()}, {g: np.mean(v) for g, v in ec.items()}
    zc, ec = pool(isc); zh, eh = pool(ish); both = sorted(set(zc) & set(zh))
    dz = {g: np.linalg.norm(zh[g] - zc[g]) for g in both}
    top = sorted(both, key=lambda g: -dz[g])[:10]
    print("\n=== v16 Crohn macrophage: top-10 by z-shift (disease->healthy) ===")
    print(f"{'gene':10s} {'z_shift':>8s} {'expr_crohn':>10s} {'expr_healthy':>12s} {'direction':>9s}")
    for g in top:
        print(f"{g:10s} {dz[g]:8.3f} {ec[g]:10.2f} {eh[g]:12.2f} {'down_in_dis' if ec[g]<eh[g] else 'up_in_dis':>11s}")


if __name__ == "__main__":
    main()
