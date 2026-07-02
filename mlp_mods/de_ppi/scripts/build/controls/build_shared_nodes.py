"""Precompute a SHARED node set per cell type, so every network of that cell type uses the SAME nodes
(removes depth-driven membership churn between studies). A gene is in the set if, over depth-QC'd cells
pooled across all sources of the cell type, it is detected in >= DETECT fraction AND clears DETECT in
>= MIN_DATASETS independent datasets (so it isn't a single-deep-study artifact). Intersected with OmniPath.

Cell QC: keep cells with >= QC_COUNTS counts and >= QC_GENES genes (drops low-quality/ambient cells).

Output: de_ppi/shared_nodes/<celltype>.txt  (one gene symbol per line)
Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/build_shared_nodes.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

RS = Path("mlp_mods/rank_shifts")
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
OUT = Path("mlp_mods/de_ppi/shared_nodes")
QC_COUNTS = 500     # min counts/cell
QC_GENES = 300      # min genes/cell
DETECT = 0.10
MIN_DATASETS = 2

CELLTYPE_SOURCES = {
    "macrophage": ["macrophage_crohn", "macrophage_crohn_colon", "macrophage_crohn_rep",
                   "macrophage_uc_smillie", "macrophage_ild"],
    "fibroblast": ["fibroblast_crohn", "fibroblast_alzheimers"],
    "stem": ["stem_crohn"],
    "microglia": ["microglia_alzheimers"],
}


def qc_mask(X):
    counts = np.asarray(X.sum(1)).ravel()
    genes = np.asarray((X > 0).sum(1)).ravel()
    return (counts >= QC_COUNTS) & (genes >= QC_GENES)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    op = pd.read_csv(OMNI, sep="\t")
    omni_genes = set(op.src) | set(op.dst)

    for ct, srcs in CELLTYPE_SOURCES.items():
        total_cells = 0
        det_cells = defaultdict(float)              # gene -> # QC'd cells detecting it (pooled)
        dataset_hits = defaultdict(set)             # gene -> set of dataset_ids where detect>=DETECT
        for src in srcs:
            h5 = next((RS / f"{src}_paired").glob("pulled_*.h5ad"), None)
            states = RS / f"{src}_states" / "cell_states.tsv"
            if h5 is None or not states.exists():
                continue
            a = ad.read_h5ad(h5)
            ds = pd.read_csv(states, sep="\t", index_col=0)["dataset_id"].astype(str).values
            X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
            keep = qc_mask(X)
            X = X[keep]; ds = ds[keep]; vn = pd.Index(a.var_names)
            total_cells += X.shape[0]
            det = np.asarray((X > 0).sum(0)).ravel()                 # detected-cell count per gene
            for gi in np.nonzero(det)[0]:
                det_cells[vn[gi]] += det[gi]
            for d in pd.unique(ds):                                  # per-dataset detection fraction
                sub = X[ds == d]
                if sub.shape[0] == 0:
                    continue
                frac = np.asarray((sub > 0).mean(0)).ravel()
                for gi in np.nonzero(frac >= DETECT)[0]:
                    dataset_hits[vn[gi]].add(d)
            print(f"  {ct}: +{src} ({int(keep.sum())}/{len(keep)} cells pass QC)", flush=True)
        shared = sorted(g for g in det_cells
                        if det_cells[g] / total_cells >= DETECT and len(dataset_hits[g]) >= MIN_DATASETS
                        and g in omni_genes)
        (OUT / f"{ct}.txt").write_text("\n".join(shared) + "\n")
        print(f"== {ct}: {total_cells} QC'd cells -> shared node set = {len(shared)} genes "
              f"(detect>={DETECT} pooled, in >={MIN_DATASETS} datasets, OmniPath-incident)\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
