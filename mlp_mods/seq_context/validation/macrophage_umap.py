"""UMAP of the macrophage connectivity-autoencoder latent (protein x context points), colored by macrophage state.
Rebuilds the same AE as macrophage_ae_reversal.py, embeds, UMAPs a subsample, colors by state (and a disease panel)."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEQ = Path("mlp_mods/seq_context"); NET = SEQ / "scenic/networks"
LATENT, HIDDEN, EPOCHS, BS, SEED, NPLOT = 64, 256, 40, 512, 0, 25000


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tags = sorted(d.name for d in NET.glob("*macrophage*") if (d / "edges_cistarget.tsv").exists())
    conn = {}; nodes = set()
    for t in tags:
        e = pd.read_csv(NET / t / "edges_cistarget.tsv", sep="\t"); c = defaultdict(set)
        for tf, tg in zip(e.tf.astype(str), e.target.astype(str)):
            c[tf].add(tg); c[tg].add(tf); nodes.add(tf); nodes.add(tg)
        conn[t] = c
    nodes = sorted(nodes); nidx = {g: i for i, g in enumerate(nodes)}; G = len(nodes)
    rctx, ii, jj, r = [], [], [], 0
    for t in tags:
        for p, nbrs in conn[t].items():
            rctx.append(t)
            for n in nbrs:
                ii.append(r); jj.append(nidx[n])
            r += 1
    X = sp.csr_matrix((np.ones(len(ii), np.float32), (ii, jj)), shape=(r, G)); rctx = np.array(rctx)
    print(f"samples={r}, G={G}", flush=True)

    pos_w = torch.tensor([(G - X.nnz / r) / (X.nnz / r)], device=device)
    enc = nn.Sequential(nn.Linear(G, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT)).to(device)
    dec = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, G)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w); idx = np.arange(r)
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, r, BS):
            b = idx[s:s + BS]; xb = torch.tensor(X[b].toarray(), device=device)
            loss = lossf(dec(enc(xb)), xb); opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); Z = np.zeros((r, LATENT), np.float32)
    with torch.no_grad():
        for s in range(0, r, 4096):
            Z[s:s+4096] = enc(torch.tensor(X[s:s+4096].toarray(), device=device)).cpu().numpy()

    state = np.array([t.split("_")[3] for t in rctx])          # <arm>_<tissue>_macrophage_<state>
    disease = np.array([t.split("_")[0] for t in rctx])
    sel = np.random.choice(r, min(NPLOT, r), replace=False)
    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(Z[sel])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, lab, title in [(axes[0], state[sel], "macrophage state"), (axes[1], disease[sel], "disease")]:
        for i, v in enumerate(sorted(set(lab))):
            m = lab == v
            ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, label=f"{v} ({m.sum()})",
                       color=plt.cm.tab10(i % 10))
        ax.set(title=f"Macrophage connectivity-AE latent — colored by {title}", xticks=[], yticks=[])
        ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout()
    out = SEQ / "images/macrophage_umap_state.png"; fig.savefig(out, dpi=130)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
