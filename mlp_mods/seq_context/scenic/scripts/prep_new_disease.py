"""Prep a pulled new-disease h5ad (01_expression/new_diseases/<slug>.h5ad) for SCENIC — no scVI.
Gene symbols from var.feature_name; gene universe = top-N HVG UNION {all present TFs, state-marker genes,
OT/drug-target genes} (bounds GRNBoost2 without censoring low-abundance regulators/targets/markers);
marker-based states; per-context raw-count export. Output mirrors prep_contexts.py. Run in .venv_scvi.

Usage: prep_new_disease.py --slug heart_valve --tissue heart --arm "heart valve disorder=hvd"
Out: seq_context/scenic/inputs/<slug>/{genes.txt, tfs.txt, context_cells.tsv, <tag>/counts.npz}
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

SEQ = Path("mlp_mods/seq_context")
DE_PPI = Path("mlp_mods/de_ppi")
MIN_CELLS = 50
N_HVG = 6000
MARKERS = {
    "macrophage": {
        "inflammatory": ["S100A8", "S100A9", "S100A12", "IL1B", "TNF", "CXCL9", "CXCL10", "CXCL11", "CCL2", "FCN1", "VCAN", "CD14"],
        "resident": ["C1QA", "C1QB", "C1QC", "MRC1", "LYVE1", "SELENOP", "FOLR2", "CD163", "MERTK", "MAF"],
        "proliferating": ["MKI67", "TOP2A", "STMN1", "TUBB"]},
    "fibroblast": {
        "inflammatory": ["IL11", "IL13RA2", "IL24", "CHI3L1", "WNT5A", "TNC", "PDPN"],
        "myofibroblast": ["ACTA2", "TAGLN", "MYH11", "DES", "PDGFRB"],
        "stromal": ["PDGFRA", "CD34", "SFRP2", "WNT2B", "BMP4", "BMP5"]},
}


def target_genes():
    g = set()
    for f in glob.glob("mlp_mods/opentargets_associations/*.tsv"):
        df = pd.read_csv(f, sep="\t")
        g |= set(df.loc[df.score_indirect > 0.3, "gene_symbol"].astype(str))
    for f in glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"):
        g |= set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="heart_valve")
    ap.add_argument("--infile", default=None)
    ap.add_argument("--tissue", default="heart")
    ap.add_argument("--arm", action="append", default=[])
    args = ap.parse_args()
    arm_map = dict(kv.split("=") for kv in args.arm) or {"heart valve disorder": "hvd", "normal": "healthy"}
    arm_map.setdefault("normal", "healthy")
    infile = args.infile or f"mlp_mods/01_expression/new_diseases/{args.slug}.h5ad"
    OUT = SEQ / "scenic/inputs" / args.slug; OUT.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(infile)
    a.var_names = a.var.feature_name.astype(str)
    a = a[:, ~a.var_names.duplicated()].copy()

    an = a.copy(); sc.pp.normalize_total(an, target_sum=1e4); sc.pp.log1p(an)
    # robust HVG: top-N by variance among genes detected in >=10 cells (avoids cell_ranger bin errors on 61k sparse genes)
    ncells = np.asarray((a.X > 0).sum(0)).ravel()
    X = an.X
    mean = np.asarray(X.mean(0)).ravel()
    var = np.asarray(X.multiply(X).mean(0)).ravel() - mean ** 2
    cand = np.where(ncells >= 10)[0]
    hvg = set(np.array(an.var_names)[cand[np.argsort(-var[cand])[:N_HVG]]])
    tr = set(pd.read_csv(DE_PPI / "protein_function.tsv", sep="\t")
             .query("func_class == 'transcription_regulator'").symbol.astype(str))
    markers = {g for panels in MARKERS.values() for lst in panels.values() for g in lst}
    tgts = target_genes()
    present = set(a.var_names)
    keep = sorted((hvg | (tr & present) | (markers & present) | (tgts & present)))
    a = a[:, keep].copy(); an = an[:, keep].copy()
    genes = list(a.var_names)
    (OUT / "genes.txt").write_text("\n".join(genes) + "\n")
    tfs = [g for g in genes if g in tr]
    (OUT / "tfs.txt").write_text("\n".join(tfs) + "\n")
    print(f"{args.slug}: genes={len(genes)} (HVG={len(hvg)}, +TF/marker/target force-kept)  tfs(sources)={len(tfs)}", flush=True)

    arm = a.obs.disease.astype(str).map(arm_map)
    assert arm.notna().all(), f"unmapped disease: {set(a.obs.disease) - set(arm_map)}"
    ct = a.obs.cell_type.astype(str)
    state = np.array(["unassigned"] * a.n_obs, dtype=object)
    for cell, panels in MARKERS.items():
        mask = (ct == cell).to_numpy()
        if mask.sum() == 0:
            continue
        sub = an[mask]
        scores = {}
        for st, mk in panels.items():
            mk = [g for g in mk if g in an.var_names]
            if len(mk) >= 2:
                sc.tl.score_genes(sub, mk, score_name="_s"); scores[st] = sub.obs["_s"].to_numpy()
            else:
                scores[st] = np.full(sub.n_obs, -np.inf)
        keys = list(scores); mat = np.stack([scores[k] for k in keys], 1)
        state[np.where(mask)[0]] = np.array(keys)[mat.argmax(1)]
    tag = arm.values + "_" + args.tissue + "_" + ct.values + "_" + state
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
