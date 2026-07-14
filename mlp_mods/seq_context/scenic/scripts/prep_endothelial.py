"""Prep the multi-disease endothelial h5ad for SCENIC — no scVI. obs.arm + obs.tissue_slug (set at pull time)
-> context = <arm>_<tissue>_endo_<state>. Gene universe = top-N variance HVG ∪ present TFs ∪ OT/drug-target
genes ∪ state markers. 5 marker-scored EC states from state_markers.tsv (cell_type=endothelial). Run in .venv_scvi.

Out: seq_context/scenic/inputs/endothelial/{genes.txt, tfs.txt, context_cells.tsv, <tag>/counts.npz}
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

SEQ = Path("mlp_mods/seq_context")
DE_PPI = Path("mlp_mods/de_ppi")
INFILE = "mlp_mods/01_expression/new_celltypes/endothelial.h5ad"
STATE_MARKERS = SEQ / "scenic/state_markers.tsv"
OUT = SEQ / "scenic/inputs/endothelial"
MIN_CELLS = 50
N_HVG = 6000


def panels():
    sm = pd.read_csv(STATE_MARKERS, sep="\t"); sm = sm[sm.cell_type == "endothelial"]
    return {r.state: r.markers.split(",") for r in sm.itertuples()}


def target_genes():
    g = set()
    for f in glob.glob("mlp_mods/opentargets_associations/*.tsv"):
        df = pd.read_csv(f, sep="\t"); g |= set(df.loc[df.score_indirect > 0.3, "gene_symbol"].astype(str))
    for f in glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"):
        g |= set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(INFILE)
    a.var_names = a.var.feature_name.astype(str)
    a = a[:, ~a.var_names.duplicated()].copy()

    an = a.copy(); sc.pp.normalize_total(an, target_sum=1e4); sc.pp.log1p(an)
    ncells = np.asarray((a.X > 0).sum(0)).ravel()
    X = an.X
    mean = np.asarray(X.mean(0)).ravel()
    var = np.asarray(X.multiply(X).mean(0)).ravel() - mean ** 2
    cand = np.where(ncells >= 10)[0]
    hvg = set(np.array(an.var_names)[cand[np.argsort(-var[cand])[:N_HVG]]])
    tr = set(pd.read_csv(DE_PPI / "protein_function.tsv", sep="\t")
             .query("func_class == 'transcription_regulator'").symbol.astype(str))
    tgts = target_genes(); present = set(a.var_names)
    pan = panels()
    markers = {g for lst in pan.values() for g in lst}
    keep = sorted(hvg | (tr & present) | (tgts & present) | (markers & present))
    a = a[:, keep].copy(); an = an[:, keep].copy()
    genes = list(a.var_names)
    (OUT / "genes.txt").write_text("\n".join(genes) + "\n")
    tfs = [g for g in genes if g in tr]
    (OUT / "tfs.txt").write_text("\n".join(tfs) + "\n")
    print(f"endothelial: genes={len(genes)} (HVG={len(hvg)})  tfs(sources)={len(tfs)}  states={list(pan)}", flush=True)

    scores = {}
    for st, mk in pan.items():
        mk = [g for g in mk if g in an.var_names]
        if len(mk) >= 2:
            sc.tl.score_genes(an, mk, score_name="_s"); scores[st] = an.obs["_s"].to_numpy()
        else:
            scores[st] = np.full(an.n_obs, -np.inf)
    keys = list(scores); state = np.array(keys)[np.stack([scores[k] for k in keys], 1).argmax(1)]
    tag = (a.obs.arm.astype(str).values + "_" + a.obs.tissue_slug.astype(str).values + "_endo_" + state)
    counts = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    rows = []
    for t in sorted(set(tag)):
        idx = np.where(tag == t)[0]; keep_ctx = len(idx) >= MIN_CELLS
        rows.append({"context": t, "n_cells": len(idx), "kept": keep_ctx})
        if keep_ctx:
            d = OUT / t; d.mkdir(exist_ok=True); sp.save_npz(d / "counts.npz", counts[idx])
        print(f"  {'SKIP ' if not keep_ctx else '     '}{t:44s} {len(idx):6d}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "context_cells.tsv", sep="\t", index=False)
    print(f"kept {sum(r['kept'] for r in rows)}/{len(rows)} contexts", flush=True)


if __name__ == "__main__":
    main()
