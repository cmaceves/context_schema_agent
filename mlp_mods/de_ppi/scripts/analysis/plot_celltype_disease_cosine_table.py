"""Render the cell_type/disease per-protein cosine matrix as a table.

Reads results/<out_name>/disease_celltype_protein_cosine.tsv (long form, i<=j), rebuilds the full
symmetric matrix, and writes both a heatmap-table PNG and a full-square TSV into
results/<out_name>/influence_analysis/tables/.

Run (.venv):
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_celltype_disease_cosine_table.py \
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


def main(out_name) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "disease_celltype_protein_cosine.tsv", sep="\t")
    groups = sorted(set(df.group_a) | set(df.group_b))
    n_prot = int(df.n_proteins.iloc[0])

    M = pd.DataFrame(np.nan, index=groups, columns=groups, dtype=float)
    for _, r in df.iterrows():
        M.loc[r.group_a, r.group_b] = r.mean_protein_cosine
        M.loc[r.group_b, r.group_a] = r.mean_protein_cosine

    tables = res / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    tsv = tables / "celltype_disease_cosine_matrix.tsv"
    M.to_csv(tsv, sep="\t")

    fig, ax = plt.subplots(figsize=(1.1 * len(groups) + 2, 1.0 * len(groups) + 1))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=8)
    for i in range(len(groups)):
        for j in range(len(groups)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title(f"mean per-protein cosine between cell_type/disease groups\n"
                 f"consensus-centered shift; {n_prot} proteins common to all groups", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean per-protein cosine")
    png = tables / "celltype_disease_cosine_matrix.png"
    fig.tight_layout(); fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {tsv}\nwrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
