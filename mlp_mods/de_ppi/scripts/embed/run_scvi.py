"""scVI per cell type: between-study normalization + integrated cell-state assignment (METHODS.md §3/§6).

For one cell type, concatenate all its paired sources, train scVI with batch_key=study8 (donor-overlap
grouped), and emit ONE staging AnnData the build can consume:
  obs:  study8, disease, donor_id, dataset_id, source, leiden, state
  obsm: X_scVI (latent; used for the Leiden states)
  X:    scVI batch-corrected NORMALIZED expression (library 1e4) over the cell type's SHARED node genes
  layers["counts"]: raw counts over those genes
States = Leiden on the scVI latent, named by the same marker signatures state_split uses (markers only NAME).

Run (GPU venv): .venv_scvi/bin/python mlp_mods/de_ppi/scripts/embed/run_scvi.py --celltype macrophage
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import scvi

RS = Path("mlp_mods/rank_shifts")
OUT = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi/scvi_staging")
SHARED = {p.stem: sorted(p.read_text().split()) for p in Path("mlp_mods/de_ppi/shared_nodes").glob("*.txt")}
QC_COUNTS, QC_GENES = 500, 300

CELLTYPE_SOURCES = {
    "macrophage": ["macrophage_crohn", "macrophage_crohn_colon", "macrophage_crohn_rep",
                   "macrophage_uc_smillie", "macrophage_ild", "macrophage_garrido_crohn", "macrophage_garrido_uc"],
    "fibroblast": ["fibroblast_crohn", "fibroblast_alzheimers"],
    "stem":       ["stem_crohn"],
    "microglia":  ["microglia_alzheimers"],
}
SIGS = {
    "macrophage": {
        "inflammatory": ["S100A8", "S100A9", "S100A12", "IL1B", "TNF", "CXCL9", "CXCL10", "CXCL11", "CCL2", "FCN1", "VCAN", "CD14"],
        "resident": ["C1QA", "C1QB", "C1QC", "MRC1", "LYVE1", "SELENOP", "FOLR2", "CD163", "MERTK", "MAF"],
        "proliferating": ["MKI67", "TOP2A", "STMN1", "TUBB"]},
}


def group_studies(obs) -> np.ndarray:
    """dataset_id -> study8 (donor-overlap connected components; label = largest-cell member [:8])."""
    ds = list(pd.unique(obs.dataset_id)); parent = {d: d for d in ds}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    d2 = defaultdict(set)
    for don, d in zip(obs.donor_id, obs.dataset_id):
        d2[don].add(d)
    for s in d2.values():
        s = list(s)
        for k in s[1:]:
            parent[find(s[0])] = find(k)
    cnt = obs.dataset_id.value_counts(); groups = defaultdict(list)
    for d in ds:
        groups[find(d)].append(d)
    lab = {}
    for members in groups.values():
        rep = max(members, key=lambda m: cnt.get(m, 0))[:8]
        for m in members:
            lab[m] = rep
    return obs.dataset_id.map(lab).values


def load_source(src):
    h5 = next((RS / f"{src}_paired").glob("pulled_*.h5ad"), None)
    st = RS / f"{src}_states" / "cell_states.tsv"
    if h5 is None or not st.exists():
        print(f"  skip {src} (missing)"); return None
    a = ad.read_h5ad(h5)
    s = pd.read_csv(st, sep="\t", index_col=0)
    if len(s) != a.n_obs:
        print(f"  skip {src} (states {len(s)} != cells {a.n_obs})"); return None
    a.obs = pd.DataFrame({"disease": s["disease"].astype(str).values,
                          "donor_id": s["donor_id"].astype(str).values,
                          "dataset_id": s["dataset_id"].astype(str).values,
                          "source": src}, index=a.obs_names)
    a.X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    return a


def main(celltype, max_epochs, n_latent, resolution) -> int:
    scvi.settings.seed = 0
    subs = [s for s in (load_source(x) for x in CELLTYPE_SOURCES[celltype]) if s is not None]
    a = ad.concat(subs, join="outer", index_unique="-")
    a.X = sp.csr_matrix(np.nan_to_num(a.X.toarray()) if sp.issparse(a.X) else np.nan_to_num(a.X))
    keep = (np.asarray(a.X.sum(1)).ravel() >= QC_COUNTS) & (np.asarray((a.X > 0).sum(1)).ravel() >= QC_GENES)
    a = a[keep].copy()
    a.obs["study8"] = group_studies(a.obs)
    a.layers["counts"] = a.X.copy()
    print(f"{celltype}: {a.n_obs} cells x {a.n_vars} genes | studies={a.obs.study8.value_counts().to_dict()}", flush=True)
    print(f"disease: {a.obs.disease.value_counts().to_dict()}", flush=True)

    # gene set = HVG (batch-aware) UNION the fixed shared node set, restricted to genes present
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat_v3", layer="counts",
                                batch_key="study8" if a.obs.study8.nunique() > 1 else None, subset=False)
    shared = [g for g in SHARED[celltype] if g in a.var_names]
    genes = sorted(set(a.var_names[a.var.highly_variable]) | set(shared))
    a = a[:, genes].copy()
    print(f"scVI gene set: {len(genes)} (HVG ∪ {len(shared)} shared)", flush=True)

    scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key="study8")
    model = scvi.model.SCVI(a, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True)
    a.obsm["X_scVI"] = model.get_latent_representation()

    # integrated states: Leiden on the scVI latent, named by marker signatures
    sc.pp.neighbors(a, use_rep="X_scVI")
    sc.tl.leiden(a, resolution=resolution, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
    norm = model.get_normalized_expression(library_size=1e4)            # cells x genes (DataFrame)
    a.X = sp.csr_matrix(norm.values)                                     # corrected normalized expression
    sigs = SIGS.get(celltype)
    if sigs:
        logn = a.copy(); logn.X = a.X.copy(); sc.pp.log1p(logn)
        for nm, gl in sigs.items():
            sc.tl.score_genes(logn, [g for g in gl if g in logn.var_names], score_name=f"sig_{nm}")
        sigdf = logn.obs[[f"sig_{nm}" for nm in sigs]]
        cl_mean = sigdf.groupby(a.obs.leiden.values).mean()
        cl2state = {cl: cl_mean.loc[cl].idxmax().replace("sig_", "") for cl in cl_mean.index}
        a.obs["state"] = a.obs.leiden.map(cl2state).astype(str).values
    else:
        a.obs["state"] = "leiden_" + a.obs.leiden.astype(str)

    OUT.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(OUT / f"{celltype}.h5ad")
    print(f"\nwrote {OUT/f'{celltype}.h5ad'}", flush=True)
    print("=== leiden -> state ===", dict(zip(a.obs.leiden, a.obs.state)) if False else "")
    print("state x disease:\n", pd.crosstab(a.obs.state, a.obs.disease).to_string(), flush=True)
    print("\nstate x study8 (batch mixing):\n", pd.crosstab(a.obs.state, a.obs.study8).to_string(), flush=True)
    for g in ["S100A8", "C1QC", "MKI67"]:
        if g in a.var_names:
            v = pd.Series(np.asarray(a[:, g].X.todense()).ravel(), index=a.obs_names)
            print(f"  scVI-norm {g} mean by disease: {v.groupby(a.obs.disease.values).mean().round(2).to_dict()}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--resolution", type=float, default=1.0)
    a = ap.parse_args()
    raise SystemExit(main(a.celltype, a.max_epochs, a.n_latent, a.resolution))
