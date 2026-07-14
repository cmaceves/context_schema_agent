"""Connectivity autoencoder on macrophage regulatory networks (no ESM — pure network role).

Each (protein, context) -> its regulatory-neighbor set (who it connects to in that context's cisTarget network)
-> encoder -> latent z -> decoder reconstructs the neighbor set. z therefore encodes a protein's *context-specific
regulatory role*. For Crohn: pool z over crohn-macrophage contexts vs tissue-matched healthy-macrophage contexts,
take the per-protein latent shift dz = z(healthy) - z(crohn), and rank genes by how much their regulatory role
changes (||dz||, weighted by influence) = candidate levers whose normalization would move the disease state toward
healthy. Directionality from degree(crohn) vs degree(healthy): "reduce" (disease-gained) vs "restore" (disease-lost).

CAVEAT: this is a correlational disease-vs-healthy difference (what regulatory roles differ), a hypothesis — not
proven causal reversal.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/macrophage_ae_reversal.py
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
LATENT, HIDDEN, EPOCHS, BS = 64, 256, 40, 512
DIS_TISSUES = ("colon", "ileum")           # Crohn macrophage tissues -> tissue-match healthy to these


def macrophage_contexts():
    return sorted(d.name for d in NET.glob("*macrophage*") if (d / "edges_cistarget.tsv").exists())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tags = macrophage_contexts()
    print(f"macrophage contexts: {len(tags)}", flush=True)
    # global node set + per-context undirected connectivity (in+out neighbors)
    conn = {}                                                    # tag -> {node: set(neighbors)}
    nodes = set()
    for t in tags:
        e = pd.read_csv(NET / t / "edges_cistarget.tsv", sep="\t")
        c = defaultdict(set)
        for tf, tg in zip(e.tf.astype(str), e.target.astype(str)):
            c[tf].add(tg); c[tg].add(tf); nodes.add(tf); nodes.add(tg)
        conn[t] = c
    nodes = sorted(nodes); nidx = {g: i for i, g in enumerate(nodes)}; G = len(nodes)
    # build sample matrix: one row per (context, protein) = that protein's neighbor indicator vector
    rows_ctx, rows_prot, ii, jj = [], [], [], []
    r = 0
    for t in tags:
        for p, nbrs in conn[t].items():
            rows_ctx.append(t); rows_prot.append(p)
            for n in nbrs:
                ii.append(r); jj.append(nidx[n])
            r += 1
    X = sp.csr_matrix((np.ones(len(ii), np.float32), (ii, jj)), shape=(r, G))
    rows_ctx = np.array(rows_ctx); rows_prot = np.array(rows_prot)
    print(f"samples (protein x context) = {r}, gene universe G = {G}, mean neighbors/row = {X.nnz / r:.1f}", flush=True)

    # autoencoder
    pos_w = torch.tensor([(G - X.nnz / r) / (X.nnz / r)], device=device)
    enc = nn.Sequential(nn.Linear(G, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT)).to(device)
    dec = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, G)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    idx = np.arange(r)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); tot = 0.0
        for s in range(0, r, BS):
            b = idx[s:s + BS]
            xb = torch.tensor(X[b].toarray(), device=device)
            z = enc(xb); logit = dec(z); loss = lossf(logit, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1}/{EPOCHS} recon-loss {tot/(r//BS+1):.4f}", flush=True)

    # latent for all rows
    enc.eval()
    Z = np.zeros((r, LATENT), np.float32)
    with torch.no_grad():
        for s in range(0, r, 4096):
            Z[s:s+4096] = enc(torch.tensor(X[s:s+4096].toarray(), device=device)).cpu().numpy()

    # pool per protein: crohn-macrophage (colon/ileum) vs tissue-matched healthy-macrophage
    def pool(mask):
        acc = defaultdict(list)
        for k in np.where(mask)[0]:
            acc[rows_prot[k]].append(Z[k])
        return {g: np.mean(v, 0) for g, v in acc.items()}, \
               {g: sum(len(conn[rows_ctx[k]][g]) for k in np.where(mask & (rows_prot == g))[0]) for g in set(rows_prot[mask])}
    is_crohn = np.array([t.startswith("crohn_") and any(ts in t for ts in DIS_TISSUES) for t in rows_ctx])
    is_healthy = np.array([t.startswith("healthy_") and any(ts in t for ts in DIS_TISSUES) for t in rows_ctx])
    print(f"crohn-mac rows={is_crohn.sum()} ({len(set(rows_prot[is_crohn]))} proteins), "
          f"matched healthy-mac rows={is_healthy.sum()} ({len(set(rows_prot[is_healthy]))} proteins)", flush=True)
    zc, degc = pool(is_crohn); zh, degh = pool(is_healthy)
    both = sorted(set(zc) & set(zh))
    dz = {g: np.linalg.norm(zh[g] - zc[g]) for g in both}
    # rank by role-shift magnitude, weighted by influence (mean degree across the two states)
    infl = {g: 0.5 * (degc.get(g, 0) + degh.get(g, 0)) for g in both}
    score = {g: dz[g] * np.log1p(infl[g]) for g in both}
    top = sorted(both, key=lambda g: -score[g])[:10]
    print("\n=== Crohn macrophage: top-10 genes whose regulatory role shifts most (disease -> healthy) ===")
    print(f"{'gene':10s} {'role_shift':>10s} {'infl':>6s} {'deg_crohn':>9s} {'deg_healthy':>11s} {'direction':>9s}")
    for g in top:
        dc, dh = degc.get(g, 0), degh.get(g, 0)
        direction = "reduce" if dc > dh else "restore"      # disease-gained connectivity -> reduce; lost -> restore
        print(f"{g:10s} {dz[g]:10.3f} {infl[g]:6.0f} {dc:9d} {dh:11d} {direction:>9s}")
    pd.DataFrame([{"gene": g, "role_shift": dz[g], "influence": infl[g],
                   "deg_crohn": degc.get(g, 0), "deg_healthy": degh.get(g, 0),
                   "direction": "reduce" if degc.get(g, 0) > degh.get(g, 0) else "restore"}
                  for g in sorted(both, key=lambda g: -score[g])]).to_csv(
        SEQ / "results/macrophage_crohn_reversal.tsv", sep="\t", index=False)
    print(f"\nwrote {SEQ/'results/macrophage_crohn_reversal.tsv'}", flush=True)


if __name__ == "__main__":
    main()
