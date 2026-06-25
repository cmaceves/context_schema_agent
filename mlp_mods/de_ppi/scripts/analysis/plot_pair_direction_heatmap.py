"""DIRECTION-aware network similarity heatmap (no healthy reference).

The pair_shift heatmap uses ‖Z_A[p] - Z_B[p]‖ (magnitude only, sign-blind). This one uses the ANGLE
between each protein's deviation from the cross-network CONSENSUS:

  consensus position  Z̄[p]  = mean of Z[p] over ALL networks in the panel (the reference-free origin)
  deviation vector    r_X[p] = Z_X[p] - Z̄[p]
  cell(A,B)           = mean over shared "mover" proteins of  cos(r_A[p], r_B[p])

So warm (->+1) = A and B deviate from the group consensus the SAME way (aligned reorganization),
~0 = unrelated, cool (->-1) = OPPOSITE directions. Direction-aware, and needs no healthy arm — the
origin is the centroid of the networks shown (add/remove networks => consensus moves).

Every shared protein is scored, but the cosine is magnitude-weighted by min(‖r_A‖,‖r_B‖): a protein that
moved far from consensus in BOTH networks dominates, while one near the consensus in EITHER (whose angle
is unstable noise) gets ~0 weight and contributes nothing. So there is no arbitrary mover cutoff and
moderate movers count proportionally (--mover-q adds an optional hard top-fraction filter, default off).

Layout (cell type -> tissue -> state) and the gene-set / DE restriction options mirror
plot_pair_similarity_heatmap.py.

Output: results/<out_name>/influence_analysis/pair_direction_heatmap[_<sfx>].png  (+ .tsv)

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_pair_direction_heatmap.py \
      --out-name crohn_alzheimer_ild_uc_embedding_expressed
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import order_tags, block_separators, annotate_hierarchy   # shared cell-type -> tissue -> state ordering

MOVER_Q = 0.0     # default: NO rank cutoff. Each protein is weighted by min(‖r_A‖,‖r_B‖), so
                  # near-consensus proteins (unstable angle) contribute ~0 continuously — no arbitrary
                  # threshold, and moderate movers count proportionally. >0 adds an optional hard filter.


def main(out_name, mover_q=MOVER_Q, genes_file=None) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    node_id, node_type = d["node_id"], d["node_type"]
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]          # Z:(T,N,dim)  present:(T,N) bool
    is_prot = node_type == "protein"

    gene_label = None
    if genes_file:                                                   # restrict to a fixed gene list
        gl = set()
        for line in Path(genes_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                gl.add(line.split()[0])
        gl.discard("gene")
        keep_gene = np.array([g in gl for g in node_id])
        is_prot = is_prot & keep_gene
        gene_label = Path(genes_file).stem
        print(f"genes={genes_file}: {len(gl)} listed, {int((is_prot).sum())} present as proteins", flush=True)

    pi = np.where(is_prot)[0]
    Zp, pres = Z[:, pi, :], present[:, pi]                           # restrict to proteins

    # consensus position per protein = mean over networks where it is present (reference-free origin)
    masked = np.where(pres[:, :, None], Zp, np.nan)
    consensus = np.nanmean(masked, axis=0)                           # (n_prot, dim)
    R = Zp - consensus[None, :, :]                                   # deviation vectors (T, n_prot, dim)
    Rn = np.linalg.norm(R, axis=2)                                   # (T, n_prot)

    # per-network mover mask: present AND deviation magnitude above the within-network quantile
    mover = np.zeros_like(pres)
    for t in range(len(tags)):
        m = pres[t]
        if m.sum() == 0:
            continue
        thr = np.quantile(Rn[t, m], mover_q)
        mover[t] = m & (Rn[t] >= thr)

    T = len(tags)
    V = pd.DataFrame(np.nan, index=tags, columns=tags, dtype=float)
    Nm = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
    for i in range(T):
        for j in range(i, T):
            both = mover[i] & mover[j]
            if both.sum() == 0:
                continue
            ra, rb = R[i, both], R[j, both]
            na, nb = Rn[i, both], Rn[j, both]
            cos = (ra * rb).sum(1) / (na * nb + 1e-9)
            w = np.minimum(na, nb)                                   # weight by the confidently-displaced endpoint
            val = float((cos * w).sum() / (w.sum() + 1e-9))
            V.loc[tags[i], tags[j]] = V.loc[tags[j], tags[i]] = val
            Nm.loc[tags[i], tags[j]] = Nm.loc[tags[j], tags[i]] = int(both.sum())

    tags_o = order_tags(tags)
    P = V.loc[tags_o, tags_o]

    sz = max(12, 0.55 * len(tags_o))                       # scale figure with #networks so cells stay legible
    fig, ax = plt.subplots(figsize=(sz, sz))
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("0.85")
    im = ax.imshow(np.ma.masked_invalid(P.values), cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(tags_o))); ax.set_xticklabels(tags_o, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tags_o))); ax.set_yticklabels(tags_o, fontsize=8)
    for i in range(len(tags_o)):
        for j in range(len(tags_o)):
            v = P.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.55 else "black")
    ct_b, ti_b = block_separators(tags_o)
    for k in ct_b:
        ax.axhline(k - 0.5, color="0.2", lw=1.8); ax.axvline(k - 0.5, color="0.2", lw=1.8)
    for k in ti_b:
        ax.axhline(k - 0.5, color="0.2", lw=0.7); ax.axvline(k - 0.5, color="0.2", lw=0.7)
    annotate_hierarchy(ax, tags_o, also_left=True)
    gtag = f"  [gene set: {gene_label}]" if gene_label else ""
    filt = f"; movers only (top {1 - mover_q:.0%} by ‖dev‖/net)" if mover_q > 0 else ""
    ax.set_title(f"Direction-aware network similarity ({out_name})\n"
                 f"min‖dev‖-weighted mean cos of consensus-centered deviation over shared proteins"
                 f"{filt}; +1=same direction, -1=opposite{gtag}", pad=60)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cos(deviation_A, deviation_B)")
    sfx = f"_{gene_label}" if gene_label else ""
    out = res / "images" / f"pair_direction_heatmap{sfx}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    P.to_csv(res / "tables" / f"pair_direction_heatmap{sfx}.tsv", sep="\t")
    print(f"wrote {out}", flush=True)

    tri = V.where(np.triu(np.ones(V.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    print("\nmost direction-ALIGNED pairs:")
    for (a, b), v in tri.head(8).items():
        print(f"  {v:+.2f}  {a}  <->  {b}  (movers={Nm.loc[a,b]})")
    print("most direction-OPPOSED pairs:")
    for (a, b), v in tri.tail(5).items():
        print(f"  {v:+.2f}  {a}  <->  {b}  (movers={Nm.loc[a,b]})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="direction-aware (consensus-centered) network similarity heatmap")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--mover-q", type=float, default=MOVER_Q,
                    help="within-network quantile of ‖deviation‖ above which a protein counts as a mover")
    ap.add_argument("--genes", default=None, help="restrict to a fixed gene-list file (1st token per non-# line)")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.mover_q, a.genes))
