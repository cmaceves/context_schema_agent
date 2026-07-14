"""Prep step for SCENIC (GRNBoost2) generation — see seq_context/SEQ_CONTEXT_EMBED.md.

Turns a cell type's scVI staging AnnData into per-(arm, tissue, state) contexts EXACTLY as adapt_scvi_build.py
does (DISEASE_ARM + SRC_TISSUE maps), and exports, per context, the RAW COUNTS matrix (cells x genes) GRNBoost2
consumes, plus that cell type's gene list and TF list. Run in .venv_scvi.

TF list = symbols with func_class == 'transcription_regulator' in de_ppi/protein_function.tsv, intersected with
the cell type's expressed-gene universe (broader than the curated Lambert set; noted in the doc).

Usage: prep_contexts.py --celltype {macrophage|fibroblast|microglia|stem}
Out: seq_context/scenic/inputs/<celltype>/{genes.txt, tfs.txt, context_cells.tsv, <tag>/counts.npz}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

DE_PPI = Path("mlp_mods/de_ppi")        # source data lives here
SEQ = Path("mlp_mods/seq_context")      # this workspace
STAGING_DIR = DE_PPI / "results/crohn_alzheimer_ild_uc_embedding_expressed_scvi/scvi_staging"
MARKERS = SEQ / "scenic/state_markers.tsv"   # published per-state panels (fibroblast/microglia/stem)
MIN_CELLS = 50  # matches adapt_scvi_build.py

# disease obs value -> arm slug (extends adapt_scvi_build's map with Alzheimer)
DISEASE_ARM = {"normal": "healthy", "Crohn disease": "crohn", "ulcerative colitis": "uc",
               "interstitial lung disease": "ild", "Alzheimer disease": "alz"}
# per-source tissue (staging obs carries no tissue). Crohn fibroblast/stem h5ads dropped tissue -> 'intestine';
# alzheimer sources -> 'brain' (cortical regions collapsed).
SRC_TISSUE = {"macrophage_ild": "lung", "macrophage_crohn": "ileum", "macrophage_crohn_rep": "ileum",
              "macrophage_crohn_colon": "colon", "macrophage_uc_smillie": "colon",
              "macrophage_garrido_crohn": "colon", "macrophage_garrido_uc": "colon",
              "fibroblast_crohn": "intestine", "fibroblast_alzheimers": "brain",
              "microglia_alzheimers": "brain", "stem_crohn": "intestine"}


def marker_states(a, celltype, tissue_arr):
    """TISSUE-AWARE published-marker state assignment (state_markers.tsv). Score each (tissue,state) panel on
    log-normalized counts; each cell = argmax over the panels of ITS tissue. Returns state array or None."""
    md = pd.read_csv(MARKERS, sep="\t")
    md = md[md.cell_type == celltype]
    if md.empty:
        return None
    an = ad.AnnData(a.layers["counts"].copy(), var=a.var.copy())     # score on lognorm CP10k, not scVI values
    sc.pp.normalize_total(an, target_sum=1e4); sc.pp.log1p(an)
    present = set(an.var_names)
    scores = {}                                                       # (tissue,state) -> per-cell score
    for _, r in md.iterrows():
        mk = [g for g in r.markers.split(",") if g in present]
        key = (r.tissue, r.state)
        if len(mk) >= 2:
            sc.tl.score_genes(an, mk, score_name="_s")
            scores[key] = an.obs["_s"].to_numpy()
        else:
            scores[key] = np.full(an.n_obs, -np.inf)
        print(f"    {celltype} {r.tissue}/{r.state}: {len(mk)}/{len(r.markers.split(','))} markers present", flush=True)
    out = np.array(["unassigned"] * a.n_obs, dtype=object)
    for tis in np.unique(tissue_arr):
        keys = [k for k in scores if k[0] == tis]
        if not keys:
            continue
        mat = np.stack([scores[k] for k in keys], axis=1)             # cells x states(of this tissue)
        best = np.array([k[1] for k in keys])[mat.argmax(1)]
        m = tissue_arr == tis
        out[m] = best[m]
    return out


def main(celltype):
    OUT = SEQ / "scenic/inputs" / celltype
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(STAGING_DIR / f"{celltype}.h5ad")
    obs = a.obs
    unknown = set(obs.source.astype(str)) - set(SRC_TISSUE)
    assert not unknown, f"sources without tissue map: {unknown}"
    arm = obs.disease.astype(str).map(DISEASE_ARM)
    assert arm.notna().all(), f"unmapped disease: {set(obs.disease.unique()) - set(DISEASE_ARM)}"
    tissue = obs.source.astype(str).map(SRC_TISSUE)
    # macrophage keeps its curated named states; new cell types use published marker panels (tissue-aware)
    ms = None if celltype == "macrophage" else marker_states(a, celltype, tissue.values)
    state = pd.Series(ms, index=obs.index).astype(str) if ms is not None else obs.state.astype(str)
    print("  state counts:", dict(state.value_counts()), flush=True)
    tag = arm.values + "_" + tissue.values + "_" + celltype + "_" + state.values

    genes = list(a.var_names)
    (OUT / "genes.txt").write_text("\n".join(genes) + "\n")

    # TF list from local func annotation, intersected with the gene universe
    pf = pd.read_csv(DE_PPI / "protein_function.tsv", sep="\t")
    tf_all = set(pf.loc[pf.func_class == "transcription_regulator", "symbol"].astype(str))
    tfs = [g for g in genes if g in tf_all]
    (OUT / "tfs.txt").write_text("\n".join(tfs) + "\n")
    print(f"genes={len(genes)}  TFs(in universe)={len(tfs)} / {len(tf_all)} annotated", flush=True)

    counts = a.layers["counts"]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)

    rows = []
    for t in sorted(pd.unique(tag)):
        idx = np.where(tag == t)[0]
        n = len(idx)
        keep = n >= MIN_CELLS
        rows.append({"context": t, "n_cells": n, "kept": keep})
        if not keep:
            print(f"  SKIP {t:42s} {n:6d} cells < {MIN_CELLS}", flush=True)
            continue
        d = OUT / t
        d.mkdir(exist_ok=True)
        sp.save_npz(d / "counts.npz", counts[idx])
        (d / "obs_names.txt").write_text("\n".join(a.obs_names[idx]) + "\n")
        print(f"  {t:42s} {n:6d} cells  -> {d.relative_to(SEQ)}/counts.npz", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "context_cells.tsv", sep="\t", index=False)
    kept = sum(r["kept"] for r in rows)
    print(f"\ncontexts total={len(rows)}  kept(>= {MIN_CELLS})={kept}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", default="macrophage")
    main(ap.parse_args().celltype)
