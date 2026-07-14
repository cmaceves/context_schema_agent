"""v17 = v16 + a second output head for PATHWAY ACTIVITY (per-protein-per-context, membership-gated).

Input (learned embeddings, no ESM): [protein-ID, disease, tissue, state] -> encoder -> z. Two heads:
  head 1 (expression):  z -> scalar,   MSE vs the protein's expression in the context
  head 2 (pathway):     z -> Np,       for the protein's OWN pathways, predict their activity in the context
                                        (member pathways -> context activity; non-members -> 0, to force protein-specificity)
L = MSE_expr + lam * (mean_member (pred-activity)^2 + mean_nonmember pred^2). Macrophage only.

Out: results/link_v17/embeddings.npz + Crohn driver readout (embedding shift), compared to expression-only.
Run: .venv_scvi/bin/python mlp_mods/seq_context/scripts/train_v17_expr_pathway.py
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEQ = Path("mlp_mods/seq_context")
PID_DIM, CTX_DIM, LATENT, HIDDEN, EPOCHS, BS, LAM, SEED = 128, 32, 64, 256, 60, 4096, 0.5, 0
DIS_TISSUES = ("colon", "ileum")


class V17(nn.Module):
    def __init__(self, nP, nD, nT, nS, Np):
        super().__init__()
        self.p = nn.Embedding(nP, PID_DIM); self.d = nn.Embedding(nD, CTX_DIM)
        self.t = nn.Embedding(nT, CTX_DIM); self.s = nn.Embedding(nS, CTX_DIM)
        self.enc = nn.Sequential(nn.Linear(PID_DIM + 3 * CTX_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT))
        self.eh = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
        self.ph = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, Np))

    def embed(self, p, d, t, s):
        return self.enc(torch.cat([self.p(p), self.d(d), self.t(t), self.s(s)], -1))

    def forward(self, p, d, t, s):
        z = self.embed(p, d, t, s)
        return self.eh(z).squeeze(-1), self.ph(z)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = pd.read_csv(SEQ / "scenic/expression_activity.tsv", sep="\t", index_col=0)
    E = E.loc[[t for t in E.index if "macrophage" in t]]
    A = pd.read_csv(SEQ / "scenic/pathway_activity.tsv", sep="\t", index_col=0).reindex(E.index).fillna(0.0)
    genes = list(E.columns); gv = {g: i for i, g in enumerate(genes)}
    pathways = list(A.columns); ctx_row = {t: i for i, t in enumerate(E.index)}
    A_arr = A.to_numpy(np.float32)                                       # (n_ctx, Np)
    # membership M: genes x pathways (from Reactome, aligned to A's columns)
    gmt = {}
    for ln in open("mlp_mods/reactome/ReactomePathways.gmt"):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 4:
            gmt[f[0]] = set(f[2:])
    M = np.zeros((len(genes), len(pathways)), np.float32)
    for k, pw in enumerate(pathways):
        for g in gmt.get(pw, ()):
            if g in gv:
                M[gv[g], k] = 1.0
    def parts(t): a = t.split("_"); return a[0], a[1], a[3]
    dv = {v: i for i, v in enumerate(sorted({parts(t)[0] for t in E.index}))}
    tv = {v: i for i, v in enumerate(sorted({parts(t)[1] for t in E.index}))}
    sv = {v: i for i, v in enumerate(sorted({parts(t)[2] for t in E.index}))}
    pid, did, tid, sid, cr, y, ctx, prot = [], [], [], [], [], [], [], []
    for t in E.index:
        dd, tt, ss = parts(t); row = E.loc[t]
        for g in genes:
            v = row[g]
            if np.isfinite(v):
                pid.append(gv[g]); did.append(dv[dd]); tid.append(tv[tt]); sid.append(sv[ss])
                cr.append(ctx_row[t]); y.append(v); ctx.append(t); prot.append(g)
    pid, did, tid, sid, cr = map(lambda a: np.array(a), (pid, did, tid, sid, cr))
    y = np.array(y, np.float32); ctx = np.array(ctx); prot = np.array(prot); r = len(y)
    print(f"v17: samples={r}, genes={len(genes)}, pathways={len(pathways)}, mac-contexts={len(E.index)}", flush=True)

    m = V17(len(genes), len(dv), len(tv), len(sv), len(pathways)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); mse = nn.MSELoss()
    P = torch.tensor(pid, device=dev); D = torch.tensor(did, device=dev); T = torch.tensor(tid, device=dev)
    S = torch.tensor(sid, device=dev); CR = torch.tensor(cr, device=dev); Y = torch.tensor(y, device=dev)
    A_t = torch.tensor(A_arr, device=dev); M_t = torch.tensor(M, device=dev)
    idx = np.arange(r)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); te = tp = 0.0
        for s0 in range(0, r, BS):
            b = torch.tensor(idx[s0:s0 + BS], device=dev)
            pe, pp = m(P[b], D[b], T[b], S[b])
            le = mse(pe, Y[b])
            tgt = A_t[CR[b]]; memb = M_t[P[b]]
            lp = (memb * (pp - tgt) ** 2).sum() / memb.sum().clamp(min=1) + \
                 ((1 - memb) * pp ** 2).sum() / (1 - memb).sum().clamp(min=1)
            loss = le + LAM * lp; opt.zero_grad(); loss.backward(); opt.step(); te += le.item(); tp += lp.item()
        if ep % 15 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1}/{EPOCHS} expr-mse {te/(r//BS+1):.4f} pathway-loss {tp/(r//BS+1):.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        pe = np.zeros(r, np.float32); Z = np.zeros((r, LATENT), np.float32)
        for s0 in range(0, r, 8192):
            e = min(s0 + 8192, r)
            o, _ = m(P[s0:e], D[s0:e], T[s0:e], S[s0:e]); pe[s0:e] = o.cpu().numpy()
            Z[s0:e] = m.embed(P[s0:e], D[s0:e], T[s0:e], S[s0:e]).cpu().numpy()
    r2 = 1 - ((y - pe) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"expression R^2 = {r2:.3f}", flush=True)
    out = SEQ / "results/link_v17"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz", emb=Z, context=ctx, prot_idx=prot,
             disease=np.array([c.split("_")[0] for c in ctx]), state=np.array([c.split("_")[3] for c in ctx]))
    print(f"saved -> {out/'embeddings.npz'}", flush=True)
    # Crohn driver readout (embedding shift), same as v16
    Zd = Z.astype(np.float64)
    isc = np.array([t.startswith("crohn_") and any(x in t for x in DIS_TISSUES) for t in ctx])
    ish = np.array([t.startswith("healthy_") and any(x in t for x in DIS_TISSUES) for t in ctx])
    def pool(mask):
        acc = defaultdict(list); ex = defaultdict(list)
        for k in np.where(mask)[0]: acc[prot[k]].append(Zd[k]); ex[prot[k]].append(y[k])
        return {g: np.mean(v, 0) for g, v in acc.items()}, {g: np.mean(v) for g, v in ex.items()}
    zc, ec = pool(isc); zh, eh = pool(ish); both = sorted(set(zc) & set(zh))
    dz = {g: np.linalg.norm(zh[g] - zc[g]) for g in both}
    top = sorted(both, key=lambda g: -dz[g])[:12]
    print("\n=== v17 (expr+pathway) Crohn macrophage: top-12 by embedding shift ===")
    print(f"{'gene':10s} {'z_shift':>8s} {'expr_crohn':>10s} {'expr_healthy':>12s} {'direction':>11s}")
    for g in top:
        print(f"{g:10s} {dz[g]:8.3f} {ec[g]:10.2f} {eh[g]:12.2f} {'up_in_dis' if ec[g]>eh[g] else 'down_in_dis':>11s}")


if __name__ == "__main__":
    main()
