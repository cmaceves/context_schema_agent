"""Formalize the cell-type / tissue / disease / cell-state comparison of network similarity into TSV
tables, for BOTH similarity metrics side by side:

  avg_magnitude_change = mean Euclidean distance ‖Z_A[p]-Z_B[p]‖ over shared proteins  (pair_shift heatmap)
  avg_direction_cos    = min‖dev‖-weighted mean cos of consensus-centered deviations    (pair_direction heatmap)

Healthy baselines and donor-split controls are EXCLUDED — only real disease-state networks are compared.

Tables written to results/<out_name>/influence_analysis/tables/:
  factor_combinations.tsv  every (cell type, tissue, disease) same/diff combination -> mean shift, mean cos, n pairs
  factor_flip.tsv          from the all-same baseline, flip ONE factor at a time (cell type | disease | tissue)
  state_<celltype>_<tissue>.tsv   disease x cell-state, TISSUE HELD FIXED, for each (cell type, tissue)
                                  containing >=2 diseases with marker-harmonized states (-> macrophage/colon)
  README.txt               definitions, network list, and which state tables were skipped (and why)

Reads the two precomputed matrices (run the two heatmap scripts first):
  pair_shift_heatmap_mean.tsv  and  pair_direction_heatmap.tsv

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/dump_comparison_tables.py \
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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_celltype, tag_tissue, tag_disease, tag_state


def included(tags):
    """real disease-state networks only: drop healthy baselines and donor-split controls."""
    return [t for t in tags if not t.startswith("healthy_") and "split" not in t]


def agg(pairs, S, D):
    """mean shift, mean cos, n over a list of (a,b) network pairs (NaN cells dropped per metric)."""
    sh = [float(S.loc[a, b]) for a, b in pairs if pd.notna(S.loc[a, b])]
    co = [float(D.loc[a, b]) for a, b in pairs if pd.notna(D.loc[a, b])]
    return (np.mean(sh) if sh else np.nan), (np.mean(co) if co else np.nan), len(pairs)


def main(out_name) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(res / "tables" / "pair_shift_heatmap_mean.tsv", sep="\t", index_col=0)
    D = pd.read_csv(res / "tables" / "pair_direction_heatmap.tsv", sep="\t", index_col=0)
    tags = included([t for t in S.index if t in D.index])
    out = res / "tables"; out.mkdir(parents=True, exist_ok=True)
    skipped = []

    # ---- Table 1: every (cell type, tissue, disease) same/diff combination ----
    rows = []
    for a, b in combinations(tags, 2):
        rows.append((tag_celltype(a) == tag_celltype(b), tag_tissue(a) == tag_tissue(b),
                     tag_disease(a) == tag_disease(b), tag_state(a) == tag_state(b),
                     float(S.loc[a, b]), float(D.loc[a, b])))
    df = pd.DataFrame(rows, columns=["ct_same", "tissue_same", "disease_same", "state_same", "shift", "cos"])
    sm = lambda v: "same" if v else "diff"
    combo = (df.groupby(["ct_same", "tissue_same", "disease_same", "state_same"])
             .agg(avg_magnitude_change=("shift", "mean"), avg_direction_cos=("cos", "mean"),
                  n_pairs=("shift", "size"))
             .reset_index())
    for c in ["ct_same", "tissue_same", "disease_same", "state_same"]:
        combo[c] = combo[c].map(sm)
    combo = combo.rename(columns={"ct_same": "cell_type", "tissue_same": "tissue",
                                  "disease_same": "disease", "state_same": "cell_state"})
    combo = combo.sort_values("avg_direction_cos", ascending=False).round(3)
    combo.to_csv(out / "factor_combinations.tsv", sep="\t", index=False)

    # ---- Table 2: flip ONE factor from the all-same baseline (cell type / tissue / disease, pooling
    # over cell state -- "all four same" is empty since each context is a single network) ----
    def bucket(ct, ti, dz):
        m = (df.ct_same == ct) & (df.tissue_same == ti) & (df.disease_same == dz)
        return df.loc[m, "shift"].mean(), df.loc[m, "cos"].mean(), int(m.sum())
    bs, bc, bn = bucket(True, True, True)
    frows = [("baseline (all same)", bs, bs, 0.0, bc, bc, 0.0, bn)]
    for lab, key in [("cell type", (False, True, True)), ("disease", (True, True, False)),
                     ("tissue", (True, False, True))]:
        fs, fc, fn = bucket(*key)
        frows.append((f"flip {lab}", bs, fs, fs - bs, bc, fc, fc - bc, fn))
    flip = pd.DataFrame(frows, columns=[
        "flip", "avg_magnitude_change_baseline", "avg_magnitude_change_flipped", "avg_magnitude_change_delta",
        "avg_direction_cos_baseline", "avg_direction_cos_flipped", "avg_direction_cos_delta", "n_pairs"]).round(3)
    flip.to_csv(out / "factor_flip.tsv", sep="\t", index=False)

    # ---- Table 3..k: disease x cell-state, TISSUE FIXED, where >=2 diseases share a (cell type, tissue) ----
    made = []
    cts = sorted({tag_celltype(t) for t in tags})
    for ct in cts:
        for ti in sorted({tag_tissue(t) for t in tags if tag_celltype(t) == ct}):
            grp = [t for t in tags if tag_celltype(t) == ct and tag_tissue(t) == ti]
            ndis = len({tag_disease(t) for t in grp})
            if ndis < 2:
                skipped.append(f"{ct}/{ti}: only {ndis} disease in this tissue -> no cross-disease state contrast")
                continue
            rows = []
            for a, b in combinations(grp, 2):
                rows.append((tag_disease(a) == tag_disease(b), tag_state(a) == tag_state(b),
                             float(S.loc[a, b]), float(D.loc[a, b])))
            g = pd.DataFrame(rows, columns=["disease_same", "state_same", "shift", "cos"])
            tab = (g.groupby(["disease_same", "state_same"])
                   .agg(avg_magnitude_change=("shift", "mean"), avg_direction_cos=("cos", "mean"),
                        n_pairs=("shift", "size"))
                   .reset_index())
            for c in ["disease_same", "state_same"]:
                tab[c] = tab[c].map(sm)
            tab = tab.rename(columns={"disease_same": "disease", "state_same": "cell_state"}).round(3)
            fn = out / f"state_{ct}_{ti}.tsv"
            tab.to_csv(fn, sep="\t", index=False)
            made.append((ct, ti, sorted({tag_disease(t) for t in grp}), fn.name))

    # ---- README ----
    lines = ["Network similarity comparison tables", "=" * 38, "",
             "metrics (both reported side by side per row):",
             "  avg_magnitude_change = mean Euclidean distance ||Z_A[p]-Z_B[p]|| over shared proteins, averaged over the",
             "                         row's network pairs (how far proteins move; magnitude only, sign/direction-blind)",
             "  avg_direction_cos    = min||dev||-weighted mean cosine of consensus-centered deviation vectors, averaged",
             "                         over the row's pairs (which way proteins move; +1 same direction, -1 opposite)",
             "", f"networks compared ({len(tags)}, healthy + donor-split controls EXCLUDED):",
             "  " + ", ".join(tags), "",
             "factor_combinations.tsv : mean shift & cos for every same/diff combination of cell type, tissue, disease",
             "factor_flip.tsv         : change in shift & cos when ONE factor is flipped from the all-same baseline",
             "state_<ct>_<tissue>.tsv : disease x cell-state with TISSUE FIXED (only where >=2 diseases share the tissue;",
             "                          states are marker-harmonized by state_split.py, so labels are comparable)", ""]
    if made:
        lines.append("state tables emitted:")
        lines += [f"  {fn}  ({ct}/{ti}: {' vs '.join(dz)})" for ct, ti, dz, fn in made]
    lines += ["", "state tables skipped (tissue fixed -> need >=2 diseases in one tissue):"]
    lines += [f"  {s}" for s in skipped]
    (out / "README.txt").write_text("\n".join(lines) + "\n")

    print(f"wrote tables to {out}/\n")
    print("== factor_combinations.tsv =="); print(combo.to_string(index=False))
    print("\n== factor_flip.tsv =="); print(flip.to_string(index=False))
    for ct, ti, dz, fn in made:
        print(f"\n== {fn}  ({ct}/{ti}: {' vs '.join(dz)}) ==")
        print(pd.read_csv(out / fn, sep="\t").to_string(index=False))
    if skipped:
        print("\nskipped state tables:"); [print("  " + s) for s in skipped]
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
