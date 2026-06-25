"""Compare two within-cell-type shift signatures and ask how their top movers overlap.

Pair A: shift between two networks (e.g. crohn_mac_resident <-> crohn_mac_inflammatory)
Pair B: shift between two others (e.g. ild_macrophage_monocyte_derived <-> ild_macrophage_interstitial)

For proteins present in ALL FOUR networks (so both shifts are defined), we plot shift_A vs shift_B,
classify each protein by whether it is a top mover in A, B, or both, and quantify the overlap
(Spearman/Pearson correlation, Jaccard of the top-K sets, shared top movers). This is the direct
"shared disease axis" test: do two different diseases' macrophage activation axes move the same
proteins?

Outputs (results/<out_name>/influence_analysis/):
  shift_overlap_<A1>_<A2>__VS__<B1>_<B2>.png   (scatter + shared-mover bars)
  shift_overlap_<...>.tsv                        (per-protein shift_a, shift_b, func_class)

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_shift_overlap.py \
      --out-name crohn_alzheimer_ild_embedding \
      --pair-a crohn_mac_resident crohn_mac_inflammatory \
      --pair-b ild_macrophage_monocyte_derived ild_macrophage_interstitial --topk 30
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
from scipy.stats import pearsonr, spearmanr

DE_PPI = Path("mlp_mods/de_ppi")
FUNC_TSV = DE_PPI / "protein_function.tsv"


def shift_col(cols, a, b):
    for c in (f"shift_{a}_{b}", f"shift_{b}_{a}"):
        if c in cols:
            return c
    raise SystemExit(f"no shift column for {a} <-> {b}")


def main(out_name, pair_a, pair_b, topk=30) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    sh = pd.read_csv(res / "embedding_shift.tsv", sep="\t")
    fc = pd.read_csv(FUNC_TSV, sep="\t").set_index("symbol")["func_class"]
    a1, a2 = pair_a
    b1, b2 = pair_b
    ca = shift_col(sh.columns, a1, a2)
    cb = shift_col(sh.columns, b1, b2)

    m = sh[sh.node_type == "protein"][["node_id", ca, cb]].dropna()
    m = m.rename(columns={ca: "shift_a", cb: "shift_b"})
    m["func_class"] = m.node_id.map(fc).fillna("unmapped")
    labA, labB = f"{a1} ↔ {a2}", f"{b1} ↔ {b2}"
    print(f"{len(m)} proteins present in all four networks", flush=True)

    sr = spearmanr(m.shift_a, m.shift_b)
    pr = pearsonr(m.shift_a, m.shift_b)
    print(f"\ncorrelation of the two shift signatures over shared proteins:")
    print(f"  Spearman rho = {sr.statistic:.3f} (p={sr.pvalue:.1e})")
    print(f"  Pearson  r   = {pr.statistic:.3f} (p={pr.pvalue:.1e})")

    topA = set(m.sort_values("shift_a", ascending=False).head(topk).node_id)
    topB = set(m.sort_values("shift_b", ascending=False).head(topk).node_id)
    both = topA & topB
    jac = len(both) / len(topA | topB)
    print(f"\ntop-{topk} movers: |A|={len(topA)} |B|={len(topB)} shared={len(both)} "
          f"Jaccard={jac:.3f}")
    print(f"shared top movers: {sorted(both)}")
    print(f"\nTOP {topk} in A ({labA}):\n  " +
          ", ".join(m.sort_values('shift_a', ascending=False).head(topk).node_id))
    print(f"\nTOP {topk} in B ({labB}):\n  " +
          ", ".join(m.sort_values('shift_b', ascending=False).head(topk).node_id))

    def cls(r):
        ia, ib = r.node_id in topA, r.node_id in topB
        return "top in BOTH" if ia and ib else "top in A only" if ia else "top in B only" if ib else "neither"
    m["group"] = m.apply(cls, axis=1)
    gcol = {"top in BOTH": "#d62728", "top in A only": "#1f77b4",
            "top in B only": "#2ca02c", "neither": "0.75"}

    m.sort_values("shift_a", ascending=False).to_csv(
        res / "tables" / f"shift_overlap_{a1}_{a2}__VS__{b1}_{b2}.tsv", sep="\t", index=False)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    # scatter
    for g in ["neither", "top in A only", "top in B only", "top in BOTH"]:
        s = m[m.group == g]
        ax.scatter(s.shift_a, s.shift_b, s=18 if g != "neither" else 8,
                   c=gcol[g], alpha=0.8 if g != "neither" else 0.35, linewidths=0, label=f"{g} ({len(s)})")
    lim = max(m.shift_a.max(), m.shift_b.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.5)
    for _, r in m[m.group != "neither"].iterrows():
        ax.annotate(r.node_id, (r.shift_a, r.shift_b), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"shift  [{labA}]  (Crohn macrophage axis)")
    ax.set_ylabel(f"shift  [{labB}]  (ILD macrophage axis)")
    ax.set_title(f"Shift signatures: Crohn vs ILD macrophage axes\n"
                 f"Spearman ρ={sr.statistic:.2f}, top-{topk} Jaccard={jac:.2f}, shared={len(both)}")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # shared top-mover bars (in both), ranked by mean of the two shifts
    if both:
        bdf = m[m.node_id.isin(both)].copy()
        bdf["mean_shift"] = bdf[["shift_a", "shift_b"]].mean(axis=1)
        bdf = bdf.sort_values("mean_shift").tail(min(30, len(bdf)))
        y = np.arange(len(bdf)); h = 0.4
        ax2.barh(y - h/2, bdf.shift_a, height=h, color="#1f77b4", label=labA)
        ax2.barh(y + h/2, bdf.shift_b, height=h, color="#2ca02c", label=labB)
        ax2.set_yticks(y); ax2.set_yticklabels(bdf.node_id, fontsize=8)
        ax2.set_xlabel("embedding shift"); ax2.set_title(f"Shared top movers (in both top-{topk})")
        ax2.legend(fontsize=8, loc="lower right")
    else:
        ax2.text(0.5, 0.5, "no proteins in both top sets", ha="center", va="center")
        ax2.set_axis_off()

    out = res / "images" / f"shift_overlap_{a1}_{a2}__VS__{b1}_{b2}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Overlap of two within-cell-type shift signatures")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_embedding")
    ap.add_argument("--pair-a", nargs=2, required=True, metavar=("TAG1", "TAG2"))
    ap.add_argument("--pair-b", nargs=2, required=True, metavar=("TAG1", "TAG2"))
    ap.add_argument("--topk", type=int, default=30)
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.pair_a, a.pair_b, a.topk))
