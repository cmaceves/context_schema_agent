"""Per-state EXPRESSED-protein lists by MEAN NORMALIZED EXPRESSION (CP10k), for the
expression-defined-node-set experiment (build_literature_weighted_influence.py --expressed-backbone).

A gene is "expressed" in a state population if its MEAN CP10k expression (counts per 10k, i.e.
library-size-normalize each cell to total 1e4, then average over the state's cells) is >= CP10K_CUTOFF.
This is depth-robust: per-cell normalization removes the sequencing-depth confound that made a raw
detection-fraction floor collapse shallow populations (e.g. resident macrophages, ~595 UMI/cell, gave
only 1328 genes at detect>=6.5% while inflammatory gave 6334; mean-CP10k makes them comparable). It is
the field-standard way to define "genes expressed in a population" (cf. mean normalized expression /
pseudobulk CPM filtering). No ambient blacklist (a normalized-expression cutoff filters low-level
ambient naturally, and the existing blacklist is macrophage-specific).

Writes de_ppi/expressed_genes_threshold/<build>.txt (separate dir -> does not touch other files).

Run:
  .venv/bin/python mlp_mods/de_ppi/precompute_expressed_threshold.py --cutoff 0.5
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
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path("mlp_mods/de_ppi")
RS = HERE.parents[0] / "rank_shifts"
EXPR = HERE.parents[0] / "01_expression"
OUT = HERE / "expressed_genes_threshold"; OUT.mkdir(exist_ok=True)
CP10K_CUTOFF = 0.5

# paired cell types: build prefix -> (h5ad with symbol var_names, cell_states.tsv [positional align])
PAIRED = {
    "macrophage_crohn":      (RS / "macrophage_crohn_paired/pulled_macrophages.h5ad", RS / "macrophage_crohn_states/cell_states.tsv"),
    "microglia_alzheimers":  (RS / "microglia_alzheimers_paired/pulled_microglia.h5ad", RS / "microglia_alzheimers_states/cell_states.tsv"),
    "fibroblast_alzheimers": (RS / "fibroblast_alzheimers_paired/pulled_fibroblast.h5ad", RS / "fibroblast_alzheimers_states/cell_states.tsv"),
    "stem_crohn":            (RS / "stem_crohn_paired/pulled_stem.h5ad", RS / "stem_crohn_states/cell_states.tsv"),
    "macrophage_ild":        (RS / "macrophage_ild_paired/pulled_macrophage.h5ad", RS / "macrophage_ild_states/cell_states.tsv"),
}
# (build_name, cell-type prefix, state label)  -- the 15 networks in the embedding
PAIRED_BUILDS = [
    ("macrophage_crohn_inflammatory", "macrophage_crohn", "inflammatory"),
    ("macrophage_crohn_resident", "macrophage_crohn", "resident"),
    ("stem_crohn_proliferating", "stem_crohn", "proliferating"),
    ("microglia_alzheimers_homeostatic", "microglia_alzheimers", "homeostatic"),
    ("microglia_alzheimers_dam", "microglia_alzheimers", "dam"),
    ("microglia_alzheimers_interferon", "microglia_alzheimers", "interferon"),
    ("microglia_alzheimers_proliferating", "microglia_alzheimers", "proliferating"),
    ("fibroblast_alzheimers_homeostatic", "fibroblast_alzheimers", "homeostatic"),
    ("fibroblast_alzheimers_myofibroblast", "fibroblast_alzheimers", "myofibroblast"),
    ("macrophage_ild_alveolar", "macrophage_ild", "alveolar"),
    ("macrophage_ild_interstitial", "macrophage_ild", "interstitial"),
    ("macrophage_ild_monocyte_derived", "macrophage_ild", "monocyte_derived"),
]
# fibroblast_crohn loads from 01_expression slices (replicate de_scripts/fibroblast_crohn_states.load_cells)
FIBC_STATES = RS / "fibroblast_crohn_states/cell_states.tsv"
FIBC_DATASETS = {"0f4865d5-8000-4f68-8ac7-f5efea9e5e70", "19053a82-9c89-4fb8-bd19-d7b1800b0b7b",
                 "8e47ed12-c658-4252-b126-381df8d52a3d"}
FIBC_BUILDS = [("fibroblast_crohn_homeostatic", "homeostatic"),
               ("fibroblast_crohn_inflammatory", "inflammatory"),
               ("fibroblast_crohn_myofibroblast", "myofibroblast")]


def expressed(a, mask, cutoff) -> list[str]:
    sub = a[mask]
    X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    Xn = sp.diags(1e4 / tot) @ X                       # CP10k per cell
    meancp = np.asarray(Xn.mean(0)).ravel()            # mean normalized expression per gene
    return sorted(pd.Index(sub.var_names)[meancp >= cutoff])


def write(build, genes, n):
    (OUT / f"{build}.txt").write_text("\n".join(genes) + "\n")
    print(f"{build:38s} n_cells={n:6d}  expressed(meanCP10k>={CP10K_CUTOFF})={len(genes)}", flush=True)


def load_fibroblast_crohn():
    # augmented paired h5ad (existing SI slices + cxg net-new ileum cohort); var_names already symbols
    return ad.read_h5ad(RS / "fibroblast_crohn_paired" / "pulled_fibroblasts.h5ad")


def main():
    cache: dict = {}
    for build, prefix, state in PAIRED_BUILDS:
        src, states_tsv = PAIRED[prefix]
        if src not in cache:
            cache[src] = ad.read_h5ad(src)
        a = cache[src]
        st = pd.read_csv(states_tsv, sep="\t", index_col=0)
        mask = st["state"].values == state
        write(build, expressed(a, mask, CP10K_CUTOFF), int(mask.sum()))

    a = load_fibroblast_crohn()
    st = pd.read_csv(FIBC_STATES, sep="\t", index_col=0)
    assert len(st) == a.n_obs, f"fibroblast_crohn states {len(st)} != cells {a.n_obs}"
    for build, state in FIBC_BUILDS:
        mask = st["state"].values == state
        write(build, expressed(a, mask, CP10K_CUTOFF), int(mask.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=CP10K_CUTOFF, help="mean CP10k threshold")
    CP10K_CUTOFF = ap.parse_args().cutoff
    main()
