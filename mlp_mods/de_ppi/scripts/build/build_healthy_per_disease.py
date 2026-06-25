"""Per-(disease, cell type) HEALTHY baseline networks from the NORMAL arm.

Replaces the 4 cross-cell-type PINNACLE healthy anchors with 6 tissue/dataset-matched baselines:
each is built from that disease dataset's NORMAL-arm cells (obs.disease=='normal') of the cell type,
expressed node set (mean CP10k >= cutoff over normal cells), OmniPath topology, NEUTRAL weights
(no rank-shift; healthy = the reference state). So a disease net's displacement from its OWN healthy
is tissue-controlled and isolates the disease effect.

Writes into results/crohn_alzheimer_ild_uc_embedding_expressed/networks/healthy_<disease>_<celltype>/.

Run:
  .venv/bin/python mlp_mods/de_ppi/build_healthy_per_disease.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path("mlp_mods/de_ppi")
RS = HERE.parents[0] / "rank_shifts"
EXPR = HERE.parents[0] / "01_expression"
OMNI = HERE.parents[0] / "omnipath_directed" / "omnipath_global_directed.tsv"
OUT = HERE / "results" / "crohn_alzheimer_ild_uc_embedding_expressed" / "networks"
CP10K_CUTOFF = 0.5
FIBC_DATASETS = {"0f4865d5-8000-4f68-8ac7-f5efea9e5e70", "19053a82-9c89-4fb8-bd19-d7b1800b0b7b",
                 "8e47ed12-c658-4252-b126-381df8d52a3d"}

# tag -> paired h5ad (normal arm = obs.disease=='normal'); symbol var_names
PAIRED = {
    "healthy_crohn_macrophage": RS / "macrophage_crohn_paired/pulled_macrophages.h5ad",
    "healthy_ild_macrophage":   RS / "macrophage_ild_paired/pulled_macrophage.h5ad",
    "healthy_alz_microglia":    RS / "microglia_alzheimers_paired/pulled_microglia.h5ad",
    "healthy_alz_fibroblast":   RS / "fibroblast_alzheimers_paired/pulled_fibroblast.h5ad",
    "healthy_crohn_stem":       RS / "stem_crohn_paired/pulled_stem.h5ad",
}


def expressed(a) -> set:
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    meancp = np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()
    return set(pd.Index(a.var_names)[meancp >= CP10K_CUTOFF])


def load_fibroblast_crohn_normal():
    no = ad.read_h5ad(EXPR / "normal/small_intestine/fibroblast.h5ad")
    no = no[no.obs.dataset_id.astype(str).isin(FIBC_DATASETS)].copy()
    no.var_names = no.var["feature_name"].astype(str).values
    no.var_names_make_unique()
    return no


def build(tag, genes):
    op = pd.read_csv(OMNI, sep="\t")
    op = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    incident = genes & (set(op.src) | set(op.dst))
    prot = sorted(incident)
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed",
                  "direction": "", "sender_weight": 1.0}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    e = op[op.src.isin(incident) & op.dst.isin(incident)]
    pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath",
                  "edge_property": "", "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"{tag:28s} {len(prot)} proteins ({len(genes)} expressed), {len(e)} edges (neutral)", flush=True)


def main():
    for tag, h5 in PAIRED.items():
        a = ad.read_h5ad(h5)
        a = a[a.obs["disease"].astype(str) == "normal"].copy()
        print(f"  {tag}: {a.n_obs} normal cells", flush=True)
        build(tag, expressed(a))
    a = load_fibroblast_crohn_normal()
    print(f"  healthy_crohn_fibroblast: {a.n_obs} normal cells", flush=True)
    build("healthy_crohn_fibroblast", expressed(a))


if __name__ == "__main__":
    main()
