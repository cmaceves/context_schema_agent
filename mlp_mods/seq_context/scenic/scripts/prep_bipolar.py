"""Prep the Bipolar-I brain h5ad for SCENIC — no scVI. obs.arm (bipolar/healthy) + obs.tissue_slug=brain +
obs.cell_type_fam (oligodendrocyte/astrocyte/glutamatergic_neuron) -> context = <arm>_brain_<celltype>_<state>.
Three cell types share ONE inputs dir (inputs/bipolar/); state is marker-scored PER cell type (state_markers.tsv
rows for that cell_type), argmax within the cell type. Gene universe = top-N variance HVG ∪ present TFs ∪
OT/drug-target genes ∪ all-3-celltype state markers. Run in .venv_scvi.

Out: seq_context/scenic/inputs/bipolar/{genes.txt, tfs.txt, context_cells.tsv, <tag>/counts.npz}
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
INFILE = "mlp_mods/01_expression/new_celltypes/bipolar.h5ad"
STATE_MARKERS = SEQ / "scenic/state_markers.tsv"
OUT = SEQ / "scenic/inputs/bipolar"
CELLTYPES = ["oligodendrocyte", "astrocyte", "glutamatergic_neuron"]
MIN_CELLS = 50
N_HVG = 6000


def panels():
    sm = pd.read_csv(STATE_MARKERS, sep="\t"); sm = sm[sm.cell_type.isin(CELLTYPES)]
    out = {}
    for ct, grp in sm.groupby("cell_type"):
        out[ct] = {r.state: r.markers.split(",") for r in grp.itertuples()}
    return out


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
    markers = {g for ct in pan.values() for lst in ct.values() for g in lst}
    keep = sorted(hvg | (tr & present) | (tgts & present) | (markers & present))
    a = a[:, keep].copy(); an = an[:, keep].copy()
    genes = list(a.var_names)
    (OUT / "genes.txt").write_text("\n".join(genes) + "\n")
    tfs = [g for g in genes if g in tr]
    (OUT / "tfs.txt").write_text("\n".join(tfs) + "\n")
    print(f"bipolar: genes={len(genes)} (HVG={len(hvg)})  tfs(sources)={len(tfs)}  celltypes={list(pan)}", flush=True)

    # state assignment PER cell type (score that ct's panels on that ct's cells only, argmax)
    fam = a.obs.cell_type_fam.astype(str).to_numpy()
    state = np.array(["unassigned"] * a.n_obs, dtype=object)
    for ct, states in pan.items():
        cmask = fam == ct
        if cmask.sum() < MIN_CELLS:
            print(f"  {ct}: only {cmask.sum()} cells, skipped", flush=True); continue
        sub = an[cmask].copy()
        scores = {}
        for st, mk in states.items():
            mk = [g for g in mk if g in an.var_names]
            if len(mk) >= 2:
                sc.tl.score_genes(sub, mk, score_name="_s"); scores[st] = sub.obs["_s"].to_numpy()
            else:
                scores[st] = np.full(sub.n_obs, -np.inf)
        keys = list(scores)
        assign = np.array(keys)[np.stack([scores[k] for k in keys], 1).argmax(1)]
        state[cmask] = assign
        print(f"  {ct}: states={keys} -> {pd.Series(assign).value_counts().to_dict()}", flush=True)

    tag = (a.obs.arm.astype(str).values + "_brain_" + fam + "_" + state.astype(str))
    counts = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    rows = []
    for t in sorted(set(tag)):
        idx = np.where(tag == t)[0]; keep_ctx = len(idx) >= MIN_CELLS
        rows.append({"context": t, "n_cells": len(idx), "kept": keep_ctx})
        if keep_ctx:
            d = OUT / t; d.mkdir(exist_ok=True); sp.save_npz(d / "counts.npz", counts[idx])
        print(f"  {'SKIP ' if not keep_ctx else '     '}{t:48s} {len(idx):6d}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "context_cells.tsv", sep="\t", index=False)
    print(f"kept {sum(r['kept'] for r in rows)}/{len(rows)} contexts", flush=True)


if __name__ == "__main__":
    main()
