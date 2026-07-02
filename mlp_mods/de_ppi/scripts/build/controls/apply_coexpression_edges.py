"""Context-specific edge weighting by COEXPRESSION (per-donor pseudobulk, ComBat-study-corrected) on EVERY edge.

Motivation: `min(expr_i,expr_j)` (apply_joint_abundance_edges) is first-order in the SAME mean-expression node
feature the encoder already sees, so it adds no independent channel. Coexpression is a SECOND moment -- gene-gene
covariation across donors -- which is NOT recoverable from the per-node mean, so it is a genuinely orthogonal edge
signal. This build weights each OmniPath edge (i->j) by |corr(i,j)|.

Coexpression is estimated PER BIOLOGICAL CONTEXT (arm, tissue, celltype, state) and applied to EVERY network of
that context -- the main pooled net AND all its control / per-study / split / loo variants. So the message-passing
operator is identical across a context's networks (they still differ only in the per-node expression FEATURE, which
is what the controls are designed to vary), and the encoder trains + infers on ONE consistent weight regime (unlike
the earlier colon-only build, which mixed coexpr and neutral weights and produced artifacts).

Per context:
  1. Gather that context's cells (arm x state) from the matching source atlases (ATLAS registry below). States TSVs
     carry donor_id / disease / state / dataset_id, row-aligned to each h5ad. Donor-samples are keyed (dataset_id,
     donor_id) and de-duplicated across atlases (e.g. Garrido HC cells shared by the crohn/uc Garrido files).
  2. Per-donor pseudobulk over the node universe: value = log1p(mean CP10k) over the donor's cells (>= MIN_CELLS).
  3. ComBat-correct the donor x gene matrix across STUDY (batch = dataset_id), location-only, no biological covariate
     (arm+state constant within a context) -- reuses combat_ls from apply_combat_expression.
  4. Cache the corrected donor x gene matrix. For each network of the context, slice its node genes, Pearson-correlate
     across donors (--spearman for ranks), and set edge weight = max(|corr(src,dst)|, EPS).
Contexts with < MIN_DONORS usable donors leave their networks NEUTRAL (1.0) and are reported. `state=allstates`
control nets pool all states of (arm,tissue,celltype); _splitA/_splitB suffixes reuse the state's coexpression.

Clones <src> -> <dst> (node files / edge lists / direction / expression feature unchanged; only edge `weight`).
After this: retrain the encoder on <dst> (--expr-feat), re-infer controls, compare with
factor_representation_compare / expr_change_vs_embshift.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/apply_coexpression_edges.py \
        --src crohn_alzheimer_ild_uc_embedding_expressed_combat_loc \
        --dst crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "mlp_mods/de_ppi/scripts/build/controls")
from apply_combat_expression import combat_ls                       # ComBat location/scale, no EB shrinkage

RES = Path("mlp_mods/de_ppi/results")
RS = Path("mlp_mods/rank_shifts")
EPS = 1e-2                 # weight floor (matches apply_joint_abundance_edges): strictly positive & log-finite
MIN_CELLS = 10             # min cells for a donor-sample to enter the pseudobulk
MIN_DONORS = 5             # min donor-samples to trust a context's correlation (else that context stays neutral)

# (tissue, celltype) -> source atlases (base name = <base>_paired/pulled_*.h5ad + <base>_states/cell_states.tsv).
# An atlas contributes to an arm only if its states TSV carries that arm's disease label. Donor-samples shared
# across atlases (same dataset_id+donor_id) are de-duplicated, so listing the Garrido files under both IBD arms is safe.
ATLAS = {
    ("brain", "fibroblast"): ["fibroblast_alzheimers"],
    ("brain", "microglia"):  ["microglia_alzheimers"],
    ("colon", "macrophage"): ["macrophage_crohn_colon", "macrophage_uc_smillie",
                              "macrophage_garrido_crohn", "macrophage_garrido_uc"],
    ("gut", "fibroblast"):   ["fibroblast_crohn"],
    ("ileum", "macrophage"): ["macrophage_crohn", "macrophage_crohn_rep"],
    ("ileum", "stem"):       ["stem_crohn"],
    ("lung", "macrophage"):  ["macrophage_ild"],
}
ARM_LABEL = {"healthy": "normal", "crohn": "Crohn disease", "uc": "ulcerative colitis",
             "alz": "Alzheimer disease", "ild": "interstitial lung disease"}

_SRC_CACHE: dict = {}
_CTX_CACHE: dict = {}                                               # (arm,tissue,ct,state) -> corrected donor x gene DF or None


def load_source(base: str):
    """(meta[donor_id,disease,state,dataset_id], CP10k CSR, var_names) for an atlas base name. Cached."""
    if base in _SRC_CACHE:
        return _SRC_CACHE[base]
    h5 = next((RS / f"{base}_paired").glob("pulled_*.h5ad"))
    a = ad.read_h5ad(h5)
    st = pd.read_csv(RS / f"{base}_states/cell_states.tsv", sep="\t", index_col=0)
    assert len(st) == a.n_obs, f"{base}: states {len(st)} != cells {a.n_obs}"
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1.0
    cp10k = (sp.diags(1e4 / tot) @ X).tocsr()                      # counts -> CP10k (log1p after donor mean)
    meta = st[["donor_id", "disease", "state", "dataset_id"]].reset_index(drop=True)
    _SRC_CACHE[base] = (meta, cp10k, a.var_names)
    return _SRC_CACHE[base]


def context_matrix(arm: str, tissue: str, ct: str, state: str, universe: pd.Index, spearman: bool):
    """Corrected donor x universe-gene matrix for a context (state='allstates' pools states). Cached; None if thin."""
    key = (arm, tissue, ct, state)
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    label = ARM_LABEL[arm]
    rows, index, studies, seen = [], [], [], set()
    for base in ATLAS.get((tissue, ct), []):
        meta, cp10k, var = load_source(base)
        if label not in set(meta["disease"].astype(str)):
            continue
        gpos = var.get_indexer(universe); have = gpos >= 0
        sel = meta["disease"].astype(str) == label
        if state != "allstates":
            sel &= meta["state"].astype(str) == state
        if not sel.any():
            continue
        sub = meta[sel]; cprows = cp10k[sel.to_numpy()]
        for (ds, donor), grp in sub.groupby(["dataset_id", "donor_id"], sort=False):
            skey = f"{ds}|{donor}"
            if skey in seen:
                continue                                            # same donor-sample already taken from another atlas
            pos = np.where(sub.index.isin(grp.index))[0]
            if pos.size < MIN_CELLS:
                continue
            mean_cp = np.asarray(cprows[pos].mean(0)).ravel()
            vec = np.zeros(len(universe)); vec[have] = np.log1p(mean_cp[gpos[have]])
            rows.append(vec); index.append(skey); studies.append(str(ds)); seen.add(skey)
    if len(rows) < MIN_DONORS:
        _CTX_CACHE[key] = None
        return None
    X = np.vstack(rows)
    if len(set(studies)) > 1:                                      # remove between-study location shift (no bio covariate)
        X = combat_ls(X, np.array(studies), pd.DataFrame(index=range(len(rows))), scale=False)
    if spearman:
        X = pd.DataFrame(X).rank().to_numpy()
    df = pd.DataFrame(X, index=index, columns=universe)
    _CTX_CACHE[key] = df
    return df


def reweight(net_dir: Path, arm, tissue, ct, state, universe, spearman) -> tuple[str, int, int]:
    edges = pd.read_csv(net_dir / "network_edges.tsv", sep="\t", keep_default_na=False)
    pb = context_matrix(arm, tissue, ct, state, universe, spearman)
    if pb is None:
        edges["weight"] = 1.0                                      # NEUTRAL: too few donors in this context
        edges.to_csv(net_dir / "network_edges.tsv", sep="\t", index=False)
        return ("neutral(thin)", len(edges), 0)
    nodes = pd.read_csv(net_dir / "network_nodes.tsv", sep="\t", keep_default_na=False)
    genes = nodes.loc[nodes["node_type"] == "protein", "node_id"].tolist()
    C = np.abs(np.nan_to_num(np.corrcoef(pb[genes].to_numpy(), rowvar=False), nan=0.0))   # gene x gene across donors
    gpos = {g: i for i, g in enumerate(genes)}
    ok = edges["source"].isin(gpos) & edges["target"].isin(gpos)
    w = np.full(len(edges), EPS)
    si = edges.loc[ok, "source"].map(gpos).to_numpy(); ti = edges.loc[ok, "target"].map(gpos).to_numpy()
    w[ok.to_numpy()] = np.maximum(C[si, ti], EPS)
    edges["weight"] = w
    edges.to_csv(net_dir / "network_edges.tsv", sep="\t", index=False)
    return ("coexpr", len(edges), pb.shape[0])


def parse_main(name: str):
    """<arm>_<tissue>_<celltype>_<state> -> (arm,tissue,ct,state) or None if not a modelled context."""
    p = name.split("_")
    if len(p) < 4:
        return None
    arm, tissue, ct, state = p[0], p[1], p[2], "_".join(p[3:])
    return (arm, tissue, ct, state) if arm in ARM_LABEL and (tissue, ct) in ATLAS else None


def parse_control(name: str):
    """<arm>_<study8|loo>_<tissue>_<celltype>_<state>[_splitA|_splitB] -> (arm,tissue,ct,state)."""
    p = name.split("_")
    if len(p) < 5:
        return None
    arm, tissue, ct = p[0], p[2], p[3]
    state = "_".join(p[4:])
    for suf in ("_splitA", "_splitB"):
        if state.endswith(suf):
            state = state[: -len(suf)]
    return (arm, tissue, ct, state) if arm in ARM_LABEL and (tissue, ct) in ATLAS else None


def gene_universe(d: Path) -> pd.Index:
    genes = set()
    for sub in ("networks", "controls/networks"):
        for nd in (d / sub).iterdir():
            nf = nd / "network_nodes.tsv"
            if nf.exists():
                n = pd.read_csv(nf, sep="\t", keep_default_na=False, usecols=["node_id", "node_type"])
                genes |= set(n.loc[n["node_type"] == "protein", "node_id"])
    return pd.Index(sorted(genes))


def main(src: str, dst: str, spearman: bool) -> int:
    s, d = RES / src, RES / dst
    for sub in ("networks", "controls/networks"):
        srcp, dstp = s / sub, d / sub
        if not srcp.exists():
            raise SystemExit(f"missing {srcp}")
        if dstp.exists():
            shutil.rmtree(dstp)
        shutil.copytree(srcp, dstp)

    universe = gene_universe(d)
    print(f"gene universe: {len(universe)}   correlation: {'spearman' if spearman else 'pearson'}\n")
    n_ok = n_neutral = 0
    for sub, parse in (("networks", parse_main), ("controls/networks", parse_control)):
        nets = sorted(n.name for n in (d / sub).iterdir() if (n / "network_edges.tsv").exists())
        print(f"--- {sub}: {len(nets)} networks ---")
        for name in nets:
            ctx = parse(name)
            if ctx is None:
                print(f"  {name:52s} SKIP (unmapped context)"); continue
            status, ne, nd = reweight(d / sub / name, *ctx, universe, spearman)
            n_ok += status == "coexpr"; n_neutral += status != "coexpr"
            if sub == "networks" or status != "coexpr":            # print all main nets; for controls only the neutral ones
                print(f"  {name:52s} {status:14s} donors={nd:3d} edges={ne}", flush=True)
    print(f"\ncoexpr-weighted networks: {n_ok}   left neutral (thin/unmapped): {n_neutral}")
    print(f"wrote {d}")
    print("next: retrain encoder on <dst> (--expr-feat), re-infer controls, then the comparison scripts")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    ap.add_argument("--dst", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--spearman", action="store_true", help="rank correlation (default: Pearson on ComBat-corrected pseudobulk)")
    a = ap.parse_args()
    raise SystemExit(main(a.src, a.dst, a.spearman))
