"""What dominates a protein's embedding CHANGES? Variance decomposition of the macrophage connectivity-AE latent.
Trains the AE (saves latent to results/macrophage_ae_latent.npz for reuse), then:
 (1) total variance explained (eta^2) by each factor: protein identity / disease / tissue / state.
 (2) between-protein vs within-protein split (identity vs all context effects).
 (3) WITHIN-protein (residualize each protein's mean out) -> of a protein's across-context ΔZ, how much is
     disease vs tissue vs state.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

SEQ = Path("mlp_mods/seq_context"); NET = SEQ / "scenic/networks"
LATENT, HIDDEN, EPOCHS, BS, SEED = 64, 256, 40, 512, 0


def eta2(Z, labels, gm=None, tot=None):
    gm = Z.mean(0) if gm is None else gm
    tot = ((Z - gm) ** 2).sum() if tot is None else tot
    between = 0.0
    for g in set(labels):
        m = labels == g
        between += m.sum() * ((Z[m].mean(0) - gm) ** 2).sum()
    return between / tot


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tags = sorted(d.name for d in NET.glob("*macrophage*") if (d / "edges_cistarget.tsv").exists())
    conn = {}; nodes = set()
    for t in tags:
        e = pd.read_csv(NET / t / "edges_cistarget.tsv", sep="\t"); c = defaultdict(set)
        for tf, tg in zip(e.tf.astype(str), e.target.astype(str)):
            c[tf].add(tg); c[tg].add(tf); nodes.add(tf); nodes.add(tg)
        conn[t] = c
    nodes = sorted(nodes); nidx = {g: i for i, g in enumerate(nodes)}; G = len(nodes)
    rctx, rprot, ii, jj, r = [], [], [], [], 0
    for t in tags:
        for p, nbrs in conn[t].items():
            rctx.append(t); rprot.append(p)
            for n in nbrs:
                ii.append(r); jj.append(nidx[n])
            r += 1
    X = sp.csr_matrix((np.ones(len(ii), np.float32), (ii, jj)), shape=(r, G))
    rctx = np.array(rctx); rprot = np.array(rprot)

    pos_w = torch.tensor([(G - X.nnz / r) / (X.nnz / r)], device=dev)
    enc = nn.Sequential(nn.Linear(G, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT)).to(dev)
    dec = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, G)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w); idx = np.arange(r)
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, r, BS):
            b = idx[s:s + BS]; xb = torch.tensor(X[b].toarray(), device=dev)
            loss = lossf(dec(enc(xb)), xb); opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); Z = np.zeros((r, LATENT), np.float32)
    with torch.no_grad():
        for s in range(0, r, 4096):
            Z[s:s+4096] = enc(torch.tensor(X[s:s+4096].toarray(), device=dev)).cpu().numpy()
    Z = Z.astype(np.float64)
    disease = np.array([t.split("_")[0] for t in rctx]); tissue = np.array([t.split("_")[1] for t in rctx])
    state = np.array([t.split("_")[3] for t in rctx])
    np.savez(SEQ / "results/macrophage_ae_latent.npz", Z=Z, prot=rprot, ctx=rctx, disease=disease, tissue=tissue, state=state)

    print(f"samples={r}, proteins={len(set(rprot))}, contexts={len(tags)}\n", flush=True)
    print("=== (1) TOTAL variance explained (eta^2) by each factor ===")
    for name, lab in [("protein identity", rprot), ("disease", disease), ("tissue", tissue), ("state", state)]:
        print(f"  {name:18s} {eta2(Z, lab)*100:5.1f}%", flush=True)

    # (2) between- vs within-protein
    Zr = Z.copy()
    for p in set(rprot):
        m = rprot == p; Zr[m] -= Z[m].mean(0)
    tot = ((Z - Z.mean(0)) ** 2).sum(); within = (Zr ** 2).sum()
    print(f"\n=== (2) identity vs context ===")
    print(f"  between-protein (identity): {(1-within/tot)*100:5.1f}%   within-protein (all context): {within/tot*100:5.1f}%")

    # (3) within-protein: of a protein's across-context ΔZ, how much is disease/tissue/state
    print(f"\n=== (3) WITHIN-protein ΔZ decomposition (eta^2 on protein-residualized latent) ===")
    for name, lab in [("disease", disease), ("tissue", tissue), ("state", state)]:
        print(f"  {name:8s} {eta2(Zr, lab, gm=np.zeros(LATENT), tot=within)*100:5.1f}%", flush=True)
    print(f"  (remainder = residual / interactions)")


if __name__ == "__main__":
    main()
