"""One-off paired-donor DE for the MACROPHAGE_CROHN build (params preserved verbatim).

Goal: see what dysregulation signal actually survives once the depth/batch confound
is removed. We pull BOTH disease states from the SAME dataset (a37f857c) so depth and
protocol are matched, restrict to small-intestine macrophages, aggregate to per-donor
pseudobulk (kills per-cell dropout), and run a negative-binomial DE (pydeseq2) of
Crohn-vs-healthy macrophages with donors as replicates.

This is the DIRECT macrophage state contrast (Crohn-mac vs healthy-mac), NOT the
PINNACLE macrophage-vs-rest marker contrast.

Outputs (mlp_mods/rank_shifts/macrophage_crohn_paired/):
  pseudobulk_de.tsv     gene, baseMean, log2FC (crohn-healthy), pvalue, padj,
                        healthy_rank, crohn_rank, rank_shift
  pulled_macrophages.h5ad   the raw cells (cached so reruns skip the census pull)

Run with .venv (cellxgene_census + pydeseq2):
  .venv/bin/python mlp_mods/rank_shifts/de_scripts/macrophage_crohn.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "macrophage_crohn_paired"          # rank_shifts/macrophage_crohn_paired/
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "pulled_macrophages.h5ad"

DATASET = "a37f857c-779f-464e-9310-3db43a1811e7"   # the IBD atlas that has BOTH states
MIN_CELLS_PER_DONOR = 20                            # donors with fewer macrophages are dropped
MIN_DONOR_DETECT = 0.5                              # gene must be detected in >=50% donors of EACH arm
IBD_TARGETS = ["MAPK14", "NR3C1", "PPARG", "PRKCE", "JAK2", "CD14",
               "PIKFYVE", "CUL4A", "NFKB1", "STAT3", "TNF", "ITGA4"]
SANITY = ["CD14", "ACTB", "PPIA", "FKBP1A", "CD68", "LYZ", "C1QA", "FCGR3A"]


def pull():
    import anndata as ad
    if CACHE.exists():
        print(f"loading cached cells from {CACHE}", flush=True)
        return ad.read_h5ad(CACHE)
    import cellxgene_census
    filt = (f'dataset_id == "{DATASET}" and cell_type == "macrophage" '
            f'and tissue_general == "small intestine" and is_primary_data == True '
            f'and disease in ["normal", "Crohn disease"]')
    print("pulling small-intestine macrophages (normal + Crohn) from census ...", flush=True)
    with cellxgene_census.open_soma(census_version="2025-11-08") as census:
        a = cellxgene_census.get_anndata(
            census, organism="Homo sapiens", obs_value_filter=filt, X_name="raw",
            obs_column_names=["donor_id", "tissue", "disease", "assay"],
            var_column_names=["feature_id", "feature_name"])
    a.var_names = a.var["feature_name"].astype(str).values
    a.var_names_make_unique()
    a.write_h5ad(CACHE)
    print(f"cached {a.n_obs} cells x {a.n_vars} genes -> {CACHE}", flush=True)
    return a


def pseudobulk(a, disease):
    """Per-donor summed raw counts for one disease state -> (donor x gene) int DataFrame."""
    m = a[a.obs.disease == disease]
    donors = m.obs.donor_id.astype(str).values
    keep = pd.Series(donors).value_counts()
    keep = keep[keep >= MIN_CELLS_PER_DONOR].index.tolist()
    rows = []
    idx = []
    X = m.X.tocsr() if sp.issparse(m.X) else sp.csr_matrix(m.X)
    for d in keep:
        sel = donors == d
        rows.append(np.asarray(X[sel].sum(axis=0)).ravel())
        idx.append(d)
    df = pd.DataFrame(np.vstack(rows), index=idx, columns=m.var_names).round().astype(int)
    ncells = pd.Series(donors).value_counts().reindex(keep)
    return df, ncells


def main() -> int:
    a = pull()
    print(f"\ncells: {a.n_obs} | disease: {a.obs.disease.value_counts().to_dict()}", flush=True)
    tot = np.asarray((a.X.sum(1))).ravel()
    for dz in ["normal", "Crohn disease"]:
        sub = tot[(a.obs.disease == dz).values]
        print(f"  {dz}: median depth/cell = {np.median(sub):.0f}", flush=True)

    h, hc = pseudobulk(a, "normal")
    c, cc = pseudobulk(a, "Crohn disease")
    shared = sorted(set(h.index) & set(c.index))
    print(f"\nper-donor pseudobulk: healthy donors={len(h)} (median {hc.median():.0f} cells), "
          f"crohn donors={len(c)} (median {cc.median():.0f} cells)", flush=True)
    print(f"donors shared between states (true pairing): {len(shared)}", flush=True)

    # align genes, detection filter (>=50% donors detected in EACH arm)
    genes = h.columns.intersection(c.columns)
    h, c = h[genes], c[genes]
    det = ((h > 0).mean(0) >= MIN_DONOR_DETECT) & ((c > 0).mean(0) >= MIN_DONOR_DETECT)
    genes = genes[det.values]
    print(f"genes after detection filter (>= {MIN_DONOR_DETECT:.0%} donors both arms): {len(genes)}", flush=True)

    counts = pd.concat([h[genes], c[genes]], axis=0)
    meta = pd.DataFrame({"disease": ["healthy"] * len(h) + ["crohn"] * len(c)}, index=counts.index)

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts, metadata=meta, design="~disease", quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, contrast=["disease", "crohn", "healthy"], quiet=True)
    res.summary()
    df = res.results_df.copy()
    df.index.name = "gene"

    # within-arm expression rank (by mean CPM) + rank shift, for edge-weight context
    def cpm_mean_rank(mat):
        cpm = mat.div(mat.sum(1), axis=0) * 1e6
        mean = np.log1p(cpm).mean(0)
        return mean.rank(ascending=False)        # rank 1 = highest expressed
    hr, cr = cpm_mean_rank(h[genes]), cpm_mean_rank(c[genes])
    df["healthy_rank"] = hr.reindex(df.index).values
    df["crohn_rank"] = cr.reindex(df.index).values
    df["rank_shift"] = df["crohn_rank"] - df["healthy_rank"]
    df = df.sort_values("padj")
    df.to_csv(OUT / "pseudobulk_de.tsv", sep="\t")

    sig = df[df.padj < 0.05]
    print(f"\n=== RESULT: genes with padj<0.05 = {len(sig)} of {len(df)} tested ===")
    print(f"  |log2FC|>1 & padj<0.05: {((df.padj<0.05)&(df.log2FoldChange.abs()>1)).sum()}")
    print("\ntop 20 by padj (crohn vs healthy macrophages):")
    cols = ["baseMean", "log2FoldChange", "padj", "healthy_rank", "crohn_rank", "rank_shift"]
    print(df.head(20)[cols].round(3).to_string())

    print("\nIBD targets:")
    sub = df.reindex([g for g in IBD_TARGETS if g in df.index])
    print(sub[cols].round(3).to_string())

    print("\nsanity (housekeeping should be flat, padj~1):")
    sub = df.reindex([g for g in SANITY if g in df.index])
    print(sub[["baseMean", "log2FoldChange", "padj"]].round(3).to_string())
    print(f"\nwrote {OUT/'pseudobulk_de.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
