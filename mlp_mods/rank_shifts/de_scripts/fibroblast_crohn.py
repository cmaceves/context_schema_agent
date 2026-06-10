"""One-off paired-donor DE for the FIBROBLAST_CROHN build.

Same recipe as macrophage_crohn.py, re-wired for small-intestine fibroblasts. Unlike
the macrophage build (one IBD atlas with both states), Crohn fibroblasts span THREE
datasets that each also carry a normal arm, so we keep all three for power and add
`dataset` as a batch covariate (`~dataset + disease`) instead of the single-dataset
`~disease`. Datasets without both arms after the per-donor filter are dropped to keep
the design full-rank.

Outputs (mlp_mods/rank_shifts/fibroblast_crohn_paired/):
  pseudobulk_de.tsv     gene, baseMean, log2FoldChange (crohn-healthy), pvalue, padj,
                        healthy_rank, crohn_rank, rank_shift
  pulled_fibroblasts.h5ad   raw cells (cached so reruns skip the census pull)

Run with .venv (cellxgene_census + pydeseq2):
  .venv/bin/python mlp_mods/rank_shifts/de_scripts/fibroblast_crohn.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fibroblast_crohn_paired"          # rank_shifts/fibroblast_crohn_paired/
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "pulled_fibroblasts.h5ad"

# The three small-intestine fibroblast datasets that carry BOTH a normal and a Crohn arm
# (matched protocol/depth); see 01_expression donor/dataset overlap check.
DATASETS = ["0f4865d5-8000-4f68-8ac7-f5efea9e5e70",
            "19053a82-9c89-4fb8-bd19-d7b1800b0b7b",
            "8e47ed12-c658-4252-b126-381df8d52a3d"]
MIN_CELLS_PER_DONOR = 20
MIN_DONOR_DETECT = 0.5
# fibroblast / fibrosis markers of interest + housekeeping sanity (printed only)
FIB_TARGETS = ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "PDGFRA", "PDGFRB",
               "FAP", "TIMP1", "MMP2", "TGFB1", "FN1", "VIM"]
SANITY = ["ACTB", "PPIA", "GAPDH", "B2M", "PDGFRA", "COL1A1"]


def pull():
    import anndata as ad
    if CACHE.exists():
        print(f"loading cached cells from {CACHE}", flush=True)
        return ad.read_h5ad(CACHE)
    import cellxgene_census
    ds_quoted = ", ".join(f'"{d}"' for d in DATASETS)
    filt = (f'dataset_id in [{ds_quoted}] and cell_type == "fibroblast" '
            f'and tissue_general == "small intestine" and is_primary_data == True '
            f'and disease in ["normal", "Crohn disease"]')
    print("pulling small-intestine fibroblasts (normal + Crohn) from census ...", flush=True)
    with cellxgene_census.open_soma(census_version="2025-11-08") as census:
        a = cellxgene_census.get_anndata(
            census, organism="Homo sapiens", obs_value_filter=filt, X_name="raw",
            obs_column_names=["donor_id", "dataset_id", "tissue", "disease", "assay"],
            var_column_names=["feature_id", "feature_name"])
    a.var_names = a.var["feature_name"].astype(str).values
    a.var_names_make_unique()
    a.write_h5ad(CACHE)
    print(f"cached {a.n_obs} cells x {a.n_vars} genes -> {CACHE}", flush=True)
    return a


def pseudobulk(a, disease):
    """Per-donor summed raw counts for one disease state -> (donor x gene) int DataFrame,
    plus per-donor cell count and donor->dataset map (for the batch covariate)."""
    m = a[a.obs.disease == disease]
    donors = m.obs.donor_id.astype(str).values
    ds_of = dict(zip(m.obs.donor_id.astype(str), m.obs.dataset_id.astype(str)))
    keep = pd.Series(donors).value_counts()
    keep = keep[keep >= MIN_CELLS_PER_DONOR].index.tolist()
    rows, idx = [], []
    X = m.X.tocsr() if sp.issparse(m.X) else sp.csr_matrix(m.X)
    for d in keep:
        rows.append(np.asarray(X[donors == d].sum(axis=0)).ravel())
        idx.append(d)
    df = pd.DataFrame(np.vstack(rows), index=idx, columns=m.var_names).round().astype(int)
    ncells = pd.Series(donors).value_counts().reindex(keep)
    return df, ncells, {d: ds_of[d] for d in keep}


def main() -> int:
    a = pull()
    print(f"\ncells: {a.n_obs} | disease: {a.obs.disease.value_counts().to_dict()}", flush=True)

    h, hc, h_ds = pseudobulk(a, "normal")
    c, cc, c_ds = pseudobulk(a, "Crohn disease")
    print(f"\nper-donor pseudobulk: healthy donors={len(h)} (median {hc.median():.0f} cells), "
          f"crohn donors={len(c)} (median {cc.median():.0f} cells)", flush=True)
    print(f"donors shared between states (true pairing): {len(set(h.index) & set(c.index))}", flush=True)

    # keep only datasets present in BOTH arms after the per-donor filter (full-rank design)
    both = set(h_ds.values()) & set(c_ds.values())
    h = h.loc[[d for d in h.index if h_ds[d] in both]]
    c = c.loc[[d for d in c.index if c_ds[d] in both]]
    print(f"both-arm datasets kept: {len(both)} -> {sorted(both)}", flush=True)
    print(f"  donors after dataset filter: healthy={len(h)}, crohn={len(c)}", flush=True)

    # align genes, detection filter (>=50% donors detected in EACH arm)
    genes = h.columns.intersection(c.columns)
    h, c = h[genes], c[genes]
    det = ((h > 0).mean(0) >= MIN_DONOR_DETECT) & ((c > 0).mean(0) >= MIN_DONOR_DETECT)
    genes = genes[det.values]
    print(f"genes after detection filter (>= {MIN_DONOR_DETECT:.0%} donors both arms): {len(genes)}", flush=True)

    counts = pd.concat([h[genes], c[genes]], axis=0)
    meta = pd.DataFrame({
        "disease": ["healthy"] * len(h) + ["crohn"] * len(c),
        "dataset": [h_ds[d] for d in h.index] + [c_ds[d] for d in c.index],
    }, index=counts.index)
    # one dataset -> fall back to ~disease (no batch term to estimate)
    design = "~dataset + disease" if meta.dataset.nunique() > 1 else "~disease"
    print(f"DESeq design: {design}", flush=True)

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, contrast=["disease", "crohn", "healthy"], quiet=True)
    res.summary()
    df = res.results_df.copy()
    df.index.name = "gene"

    # within-arm expression rank (by mean CPM) + rank shift, for edge-weight context
    def cpm_mean_rank(mat):
        cpm = mat.div(mat.sum(1), axis=0) * 1e6
        return np.log1p(cpm).mean(0).rank(ascending=False)   # rank 1 = highest expressed
    df["healthy_rank"] = cpm_mean_rank(h[genes]).reindex(df.index).values
    df["crohn_rank"] = cpm_mean_rank(c[genes]).reindex(df.index).values
    df["rank_shift"] = df["crohn_rank"] - df["healthy_rank"]
    df = df.sort_values("padj")
    df.to_csv(OUT / "pseudobulk_de.tsv", sep="\t")

    sig = df[df.padj < 0.05]
    print(f"\n=== RESULT: genes with padj<0.05 = {len(sig)} of {len(df)} tested ===")
    print(f"  |log2FC|>1 & padj<0.05: {((df.padj<0.05)&(df.log2FoldChange.abs()>1)).sum()}")
    cols = ["baseMean", "log2FoldChange", "padj", "healthy_rank", "crohn_rank", "rank_shift"]
    print("\ntop 20 by padj (crohn vs healthy fibroblasts):")
    print(df.head(20)[cols].round(3).to_string())
    print("\nfibroblast/fibrosis markers:")
    print(df.reindex([g for g in FIB_TARGETS if g in df.index])[cols].round(3).to_string())
    print("\nsanity (housekeeping should be flat, padj~1):")
    print(df.reindex([g for g in SANITY if g in df.index])[["baseMean", "log2FoldChange", "padj"]].round(3).to_string())
    print(f"\nwrote {OUT/'pseudobulk_de.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
