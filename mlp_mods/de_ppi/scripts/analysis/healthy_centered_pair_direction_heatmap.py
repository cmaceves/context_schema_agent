"""Healthy-centered direction heatmap: SECOND version of plot_pair_direction_heatmap.py that centers on
the healthy reference protein embedding (default healthy_pinnacle_macrophage) instead of the
cross-network MEAN of that protein. r_X[p] = Z_X[p] - Z_ref[p]; cell(A,B)=min‖r‖-weighted mean
cos(r_A,r_B). Scoped to MACROPHAGE networks + proteins present in the reference (the origin is
macrophage-specific). The reference tag itself is excluded from the matrix.

Run: .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_healthy_centered_pair_direction_heatmap.py \
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
DE_PPI = Path("mlp_mods/de_ppi")
from _layout import order_tags

def main(out_name, ref_tag="healthy_pinnacle_macrophage", mover_q=0.0):
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    is_prot = d["node_type"] == "protein"
    assert ref_tag in tags, f"{ref_tag} not in embedding"
    ri = tags.index(ref_tag)
    pi = np.where(is_prot)[0]
    Zp, pres = Z[:, pi, :], present[:, pi]
    H = Zp[ri]; presH = pres[ri]                                  # healthy origin per protein
    R = Zp - H[None]                                              # deviation from healthy origin
    Rn = np.linalg.norm(R, axis=2)
    # macrophage networks only (origin is macrophage); drop the reference + non-macrophage + controls
    macs = [t for t in tags if "macrophage" in t or "_mac_" in t]
    macs = [t for t in macs if t != ref_tag and "split" not in t and not t.endswith(("_s1","_s2","_s3")) and "loo" not in t]
    T = macs
    V = pd.DataFrame(np.nan, index=T, columns=T, dtype=float)
    for a in range(len(T)):
        for b in range(a, len(T)):
            ia, ib = tags.index(T[a]), tags.index(T[b])
            both = pres[ia] & pres[ib] & presH                    # present in both AND in the healthy ref
            if both.sum() == 0: continue
            ra, rb = R[ia, both], R[ib, both]; na, nb = Rn[ia, both], Rn[ib, both]
            cos = (ra*rb).sum(1)/(na*nb+1e-9); w = np.minimum(na, nb)
            v = float((cos*w).sum()/(w.sum()+1e-9))
            V.loc[T[a], T[b]] = V.loc[T[b], T[a]] = v
    To = [t for t in order_tags(tags) if t in set(T)]
    P = V.loc[To, To]
    fig, ax = plt.subplots(figsize=(max(8,0.6*len(To)),)*2)
    im = ax.imshow(np.ma.masked_invalid(P.values), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(To))); ax.set_xticklabels(To, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(To))); ax.set_yticklabels(To, fontsize=7)
    for i in range(len(To)):
        for j in range(len(To)):
            v=P.values[i,j]
            if np.isfinite(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,color="white" if abs(v)>.55 else "black")
    ax.set_title(f"Healthy-centered ({ref_tag}) direction similarity, macrophage nets\n({out_name})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = res/"images" / "healthy_centered_pair_direction_heatmap.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    P.to_csv(res/"tables" / "healthy_centered_pair_direction_heatmap.tsv", sep="\t")
    print(f"wrote {out} and .tsv\n{P.round(2).to_string()}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--ref-tag", default="healthy_pinnacle_macrophage")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.ref_tag))
