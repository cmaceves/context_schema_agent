"""scVI staging -> build adapter: turn one cell type's scVI staging AnnData (run_scvi.py) into the
per-context PPI networks the joint encoder consumes (HISTORY.md step "_expressed_scvi").

This is the previously-missing staging->build adapter. For each (arm, tissue, cell type, state) context
present in the staging, it writes results/<build>/networks/<tag>/{network_nodes,network_edges}.tsv, where:

  - NODE MEMBERSHIP is derived from RAW COUNTS (layers["counts"]): genes with mean CP10k >= floor over the
    context's cells, intersected with OmniPath-incident nodes. scVI supplies feature VALUES + states, never
    the gene universe (METHODS.md section 6 scVI note), so membership uses the same raw-count rule as the
    ComBat build -- keeping the two builds' membership comparable.
  - EXPRESSION FEATURE is the scVI batch-corrected pseudobulk: mean over the context's cells of the
    scVI-normalized expression X (already CP10k, library 1e4), then log1p -- the exact analog of the ComBat
    build's `log1p(mean per-cell CP10k)` feature (build_pooled_controls._expr), but on the scVI scale.
  - EDGES are OmniPath directed over the node set, NEUTRAL weights (1.0), matching the main build.

arm = healthy (disease=="normal") else the disease slug; tissue is per-source (SRC_TISSUE). States are the
INTEGRATED scVI-latent Leiden states from staging (comparable across studies), so the scVI network set is NOT
1:1 with the ComBat build (e.g. ILD macrophages are re-stated resident/inflammatory/proliferating).

Run (.venv; staging must exist from run_scvi.py):
  .venv/bin/python mlp_mods/de_ppi/scripts/embed/adapt_scvi_build.py --celltype macrophage
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

BUILD = "crohn_alzheimer_ild_uc_embedding_expressed_scvi"
RES = Path("mlp_mods/de_ppi/results") / BUILD
STAGING = RES / "scvi_staging"
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
CP10K_CUTOFF = 0.5     # mean CP10k membership floor (raw counts), matches precompute_expressed_threshold.py
MIN_CELLS = 50         # skip contexts below this (matches build_pooled_disease per-study threshold)

# disease obs value -> arm slug (normal -> healthy). Everything else keeps disease as the arm.
DISEASE_ARM = {"normal": "healthy", "Crohn disease": "crohn",
               "ulcerative colitis": "uc", "interstitial lung disease": "ild"}
# source -> tissue (staging obs carries no tissue; per-source, matches build_pooled_controls SOURCES)
SRC_TISSUE = {
    "macrophage_ild": "lung",
    "macrophage_crohn": "ileum",
    "macrophage_crohn_rep": "ileum",
    "macrophage_crohn_colon": "colon",
    "macrophage_uc_smillie": "colon",
    "macrophage_garrido_crohn": "colon",
    "macrophage_garrido_uc": "colon",
}


def mean_cp10k(counts: sp.spmatrix) -> np.ndarray:
    """mean over cells of per-cell CP10k, from RAW counts (n_cells x n_genes)."""
    X = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1.0
    return np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()


def log1p_pseudobulk(norm: sp.spmatrix) -> np.ndarray:
    """log1p of the mean over cells of the scVI-normalized expression (already CP10k)."""
    X = norm.tocsr() if sp.issparse(norm) else sp.csr_matrix(norm)
    return np.log1p(np.asarray(X.mean(0)).ravel())


def write_network(ndir: Path, node_ids: list[str], expression: pd.Series, op: pd.DataFrame) -> tuple[int, int]:
    genes = set(node_ids)
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    inc = genes & (set(o.src) | set(o.dst))          # keep only OmniPath-incident nodes (matches main build)
    prot = sorted(inc)
    o = o[o.src.isin(inc) & o.dst.isin(inc)]
    ndir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed",
                  "direction": "", "sender_weight": 1.0,
                  "expression": expression.reindex(prot).fillna(0.0).values}
                 ).to_csv(ndir / "network_nodes.tsv", sep="\t", index=False)
    pd.DataFrame({"source": o.src, "target": o.dst, "edge_origin": "OmniPath", "edge_property": "",
                  "weight": 1.0, "direction": ""}).to_csv(ndir / "network_edges.tsv", sep="\t", index=False)
    return len(prot), len(o)


def main(celltype: str) -> int:
    h5 = STAGING / f"{celltype}.h5ad"
    assert h5.exists(), f"missing scVI staging {h5} (run run_scvi.py --celltype {celltype})"
    print(f"loading {h5} ...", flush=True)
    a = ad.read_h5ad(h5)
    a.obs_names_make_unique()
    op = pd.read_csv(OMNI, sep="\t")

    obs = a.obs
    unknown = set(obs.source.unique()) - set(SRC_TISSUE)
    assert not unknown, f"sources without a tissue mapping: {unknown}"
    arm = obs.disease.astype(str).map(DISEASE_ARM)
    assert arm.notna().all(), f"unmapped disease values: {set(obs.disease.unique()) - set(DISEASE_ARM)}"
    tissue = obs.source.astype(str).map(SRC_TISSUE)
    state = obs.state.astype(str)
    ctx = pd.DataFrame({"arm": arm.values, "tissue": tissue.values, "state": state.values}, index=obs.index)

    var_names = a.var_names
    out_root = RES / "networks"
    out_root.mkdir(parents=True, exist_ok=True)
    groups = ctx.groupby(["arm", "tissue", "state"], sort=True)
    print(f"{celltype}: {a.n_obs} cells, {a.n_vars} genes -> {groups.ngroups} candidate contexts", flush=True)

    n_written = 0
    for (armv, tisv, stv), sub in groups:
        if len(sub) < MIN_CELLS:
            print(f"  skip {armv}_{tisv}_{celltype}_{stv}: {len(sub)} cells < {MIN_CELLS}", flush=True)
            continue
        rows = obs.index.get_indexer(sub.index)
        counts = a.layers["counts"][rows]
        norm = a.X[rows]
        expressed = pd.Index(var_names)[mean_cp10k(counts) >= CP10K_CUTOFF]
        expr_feat = pd.Series(log1p_pseudobulk(norm), index=pd.Index(var_names))
        expr_feat = expr_feat[~expr_feat.index.duplicated()]
        tag = f"{armv}_{tisv}_{celltype}_{stv}"
        n_prot, n_edge = write_network(out_root / tag, list(expressed), expr_feat, op)
        print(f"  {tag:42s} {len(sub):6d} cells  expressed={len(expressed):5d}  "
              f"nodes={n_prot:5d}  edges={n_edge:6d}", flush=True)
        n_written += 1

    print(f"\nwrote {n_written} networks -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", default="macrophage")
    a = ap.parse_args()
    raise SystemExit(main(a.celltype))
