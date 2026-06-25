"""Visualize how proteins shift between two networks (states/cell types/diseases) in the
shared embedding, e.g. Crohn resident vs Crohn inflammatory macrophages.

Two complementary quantities, both read from results/<out_name>/:
  embedding_shift = |z_protein(B) - z_protein(A)|  (embedding_shift.tsv) -- geometric ROLE change,
                    unsigned. Degree-controlled implicitly (a protein's degree is similar in A and B,
                    so the shared degree component cancels in the difference).
  influence delta = influence(B) - influence(A)    (joint_influence.tsv)  -- SIGNED change in
                    propagation onto the dysregulated set: did the protein gain or lose influence.

Three panels:
  1. top-N proteins by embedding shift (bar, colored by GO molecular-function class)
  2. influence(A) vs influence(B) scatter (log1p), colored by embedding shift, top movers labeled
  3. signed influence delta (B-A): top gainers and losers

Only proteins present in BOTH networks are considered.

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_state_shift.py \
      --out-name crohn_alzheimer_ild_embedding \
      --tag-a crohn_mac_resident --tag-b crohn_mac_inflammatory --topn 25
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
FUNC_TSV = DE_PPI / "protein_function.tsv"


def shift_col(cols, a, b):
    for c in (f"shift_{a}_{b}", f"shift_{b}_{a}"):
        if c in cols:
            return c
    raise SystemExit(f"no shift column for {a} <-> {b}; have e.g. "
                     f"{[c for c in cols if c.startswith('shift_')][:3]} ...")


def main(out_name, tag_a, tag_b, topn=25) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    sh = pd.read_csv(res / "embedding_shift.tsv", sep="\t")
    inf = pd.read_csv(res / "joint_influence.tsv", sep="\t")
    fc = pd.read_csv(FUNC_TSV, sep="\t").set_index("symbol")["func_class"]

    scol = shift_col(sh.columns, tag_a, tag_b)
    pa, pb = f"present_{tag_a}", f"present_{tag_b}"
    ia, ib = f"influence_{tag_a}", f"influence_{tag_b}"
    for need, src in [(pa, sh), (pb, sh), (ia, inf), (ib, inf)]:
        if need not in src.columns:
            raise SystemExit(f"missing column {need}")

    m = sh[["node_id", "node_type", pa, pb, scol]].merge(
        inf[["node_id", ia, ib]], on="node_id", how="left")
    m = m[(m.node_type == "protein") & m[pa].astype(bool) & m[pb].astype(bool)].copy()
    m["shift"] = m[scol]
    m["inf_a"], m["inf_b"] = m[ia].fillna(0.0), m[ib].fillna(0.0)
    m["delta"] = m.inf_b - m.inf_a
    m["func_class"] = m.node_id.map(fc).fillna("unmapped")
    print(f"{len(m)} proteins present in BOTH {tag_a} and {tag_b}", flush=True)

    top_shift = m.sort_values("shift", ascending=False).head(topn)
    print(f"\nTOP {topn} by embedding shift ({tag_a} -> {tag_b}):")
    print(top_shift[["node_id", "shift", "inf_a", "inf_b", "delta", "func_class"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # palette for functional classes present among top movers + scatter
    cats = sorted(m.func_class.unique())
    cm = plt.cm.tab20 if len(cats) > 10 else plt.cm.tab10
    pal = {c: cm(i % cm.N) for i, c in enumerate(cats)}

    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, 0.32 * topn + 2)))

    # panel 1: top-N embedding shift bar
    ax = axes[0]
    ts = top_shift.iloc[::-1]
    ax.barh(range(len(ts)), ts["shift"], color=[pal[f] for f in ts.func_class])
    ax.set_yticks(range(len(ts))); ax.set_yticklabels(ts.node_id, fontsize=8)
    ax.set_xlabel("embedding shift |z(B) - z(A)|")
    ax.set_title(f"top {topn} role shifts\n{tag_a}  ->  {tag_b}")
    seen = ts.func_class.unique()
    ax.legend(handles=[plt.Line2D([0], [0], marker="s", ls="", color=pal[f], label=f) for f in cats if f in seen],
              fontsize=6, loc="lower right", framealpha=0.9)

    # panel 2: signed influence delta (gainers / losers)
    ax = axes[1]
    gain = m.sort_values("delta", ascending=False).head(topn // 2)
    lose = m.sort_values("delta").head(topn // 2)
    dd = pd.concat([lose.iloc[::-1], gain]).drop_duplicates("node_id")
    colors = ["#c0392b" if v > 0 else "#2c7fb8" for v in dd.delta]
    ax.barh(range(len(dd)), dd.delta, color=colors)
    ax.set_yticks(range(len(dd))); ax.set_yticklabels(dd.node_id, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"influence delta  ({tag_b} - {tag_a})")
    ax.set_title("biggest influence gainers (red) / losers (blue)")

    fig.suptitle(f"Protein shifts: {tag_a}  ->  {tag_b}  ({out_name})", y=1.02)
    out = res / "images" / f"shift_{tag_a}__vs__{tag_b}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}", flush=True)

    m[["node_id", "func_class", "shift", "inf_a", "inf_b", "delta"]] \
        .sort_values("shift", ascending=False) \
        .to_csv(res / "tables" / f"shift_{tag_a}__vs__{tag_b}.tsv", sep="\t", index=False)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualize protein shifts between two networks")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_embedding")
    ap.add_argument("--tag-a", required=True)
    ap.add_argument("--tag-b", required=True)
    ap.add_argument("--topn", type=int, default=25)
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.tag_a, a.tag_b, a.topn))
