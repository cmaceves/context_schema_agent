"""Healthy-arm control networks (h, i, j) for the control ladder (see de_ppi/CONTROLS.md), emitted into
results/crohn_alzheimer_ild_uc_embedding_expressed/controls/networks/<tag>/ alongside the disease-arm
controls. ALL neutral (expression-only, CP10k>=0.5).

These measure variation among NORMAL/healthy networks (no disease signal), as a disease-independent floor:
  h healthy_donor_split  per (study, tissue, celltype, state): two donor halves            within-study
  i healthy_between_study per (study, tissue, celltype, state): pooled-donor net (paired across studies)
  j healthy_cell_type     per (study, tissue): all-states-pooled net (paired across cell types)  within-study

Naming: healthy_<study8>_<tissue>_<celltype>_<state>[ _split{A,B} | _allstates ].

Redundancy handling (verified empirically): the integrated pools are already cell-deduped, but the pan-GI
depositions (40a0ade8/80a2c5b6/e6aaf5a4) share DONORS (same patients, different cells). So we (1) collapse
dataset_ids that share donors into ONE study (donor-overlap connected components) -> study8 = the member
with the most cells, and (2) drop residual exact-duplicate cells by hash (safety net). This stops the
between-study control (i) from comparing a collection against itself.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/build_healthy_controls.py [--sources s ...]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, hashlib, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

RS = Path("mlp_mods/rank_shifts")
OUT = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed/controls/networks")
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
DETECT_FLOOR = 0.10   # detection fraction (genes detected in >= this fraction of cells); matches main build
MIN_CELLS = 250       # relaxed to admit Garrido colon-macrophage healthy arm (287 cells after cell-QC)
MIN_DONOR_CELLS = 20
SEED = 0
NORMAL = {"normal", "healthy"}

# source -> (tissue, celltype). disease slug is forced to 'healthy' for these controls.
SOURCES = {
    "microglia_alzheimers":   ("brain", "microglia"),
    "fibroblast_alzheimers":  ("brain", "fibroblast"),
    "macrophage_crohn":       ("ileum", "macrophage"),
    "macrophage_crohn_colon": ("colon", "macrophage"),
    "macrophage_crohn_rep":   ("colon", "macrophage"),
    "macrophage_garrido_crohn": ("colon", "macrophage"),   # Garrido GSE214695 healthy colon macrophages (HC arm)
    "fibroblast_crohn":       ("gut",   "fibroblast"),
    "stem_crohn":             ("ileum", "stem"),
    "macrophage_uc_smillie":  ("colon", "macrophage"),
    "macrophage_ild":         ("lung",  "macrophage"),
}

op = pd.read_csv(OMNI, sep="\t")
QC_COUNTS, QC_GENES = 500, 300     # cell-level depth QC
PSEUDOBULK = os.environ.get("DE_PPI_PSEUDOBULK") == "1"
SHARED = {p.stem: sorted(p.read_text().split())                # fixed per-celltype node set (build_shared_nodes.py)
          for p in Path("mlp_mods/de_ppi/shared_nodes").glob("*.txt")}


def _expr(X, var_names):
    """log1p expression: pseudobulk (sum counts, normalize once; depth-weighted) if PSEUDOBULK else mean CP10k."""
    if PSEUDOBULK:
        g = np.asarray(X.sum(0)).ravel(); tt = g.sum() or 1.0
        s = pd.Series(np.log1p(1e4 * g / tt), index=pd.Index(var_names))
    else:
        tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
        s = pd.Series(np.log1p(np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()), index=pd.Index(var_names))
    return s[~s.index.duplicated()]


def _hash_rows(X):
    Xc = X.tocsr()
    return np.array([hashlib.sha1(Xc[i].indices.tobytes() + np.round(Xc[i].data, 3).tobytes()).hexdigest()
                     for i in range(Xc.shape[0])])


def load_healthy(src):
    h5 = next((RS / f"{src}_paired").glob("pulled_*.h5ad"), None)
    states = RS / f"{src}_states" / "cell_states.tsv"
    if h5 is None or not states.exists():
        return None
    a = ad.read_h5ad(h5)
    st = pd.read_csv(states, sep="\t", index_col=0)
    assert len(st) == a.n_obs, f"{src}: cell_states {len(st)} != n_obs {a.n_obs}"
    a.obs["state"] = st["state"].astype(str).values
    a.obs["dataset_id"] = st["dataset_id"].astype(str).values
    a.obs["donor_id"] = a.obs["donor_id"].astype(str)
    h = a[a.obs.disease.astype(str).str.lower().isin(NORMAL)].copy()
    if h.n_obs == 0:
        return None
    X = h.X.tocsr() if sp.issparse(h.X) else sp.csr_matrix(h.X)
    qc = (np.asarray(X.sum(1)).ravel() >= QC_COUNTS) & (np.asarray((X > 0).sum(1)).ravel() >= QC_GENES)  # depth QC
    h = h[qc].copy()
    if h.n_obs == 0:
        return None
    # (1) collapse donor-sharing dataset_ids into study groups; (2) drop residual exact-dup cells
    X = h.X.tocsr() if sp.issparse(h.X) else sp.csr_matrix(h.X)
    keep = pd.Series(_hash_rows(X)).duplicated().values == False
    h = h[keep].copy()
    h.obs["study"] = group_studies(h.obs)
    return h


def group_studies(obs) -> np.ndarray:
    """dataset_id -> study8 (donor-overlap connected components; label = largest-cell member)."""
    ds = list(pd.unique(obs.dataset_id))
    parent = {d: d for d in ds}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    d2ds = defaultdict(set)
    for don, d in zip(obs.donor_id, obs.dataset_id):
        d2ds[don].add(d)
    for dss in d2ds.values():
        dss = list(dss)
        for k in dss[1:]:
            union(dss[0], k)
    cnt = obs.dataset_id.value_counts()
    groups = defaultdict(list)
    for d in ds:
        groups[find(d)].append(d)
    label = {}
    for members in groups.values():
        rep = max(members, key=lambda m: cnt.get(m, 0))[:8]
        for m in members:
            label[m] = rep
    return obs.dataset_id.map(label).values


def neutral(a, tag, mask, celltype):
    if mask.sum() < MIN_CELLS:
        return False
    sub = a[mask]
    X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    log_expr = _expr(X, sub.var_names)
    prot = SHARED[celltype]                                      # FIXED shared node set for this cell type
    genes = set(prot)
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed",
                  "direction": "", "sender_weight": 1.0,
                  "expression": log_expr.reindex(prot).fillna(0.0).values}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    pd.DataFrame({"source": o.src, "target": o.dst, "edge_origin": "OmniPath", "edge_property": "",
                  "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"  {tag:54s} {int(mask.sum()):6d} cells -> {len(prot)} proteins", flush=True)
    return True


def emit_source(src):
    tissue, ct = SOURCES[src]
    h = load_healthy(src)
    if h is None:
        print(f"== {src}: no healthy arm, skip ==", flush=True); return
    obs = h.obs
    print(f"== {src} ({tissue}/{ct}) healthy: {h.n_obs} cells, "
          f"studies={sorted(set(obs.study))} ==", flush=True)
    for grp in sorted(obs.study.unique()):
        gm = (obs.study == grp).values
        base = f"healthy_{grp}_{tissue}_{ct}"
        neutral(h, f"{base}_allstates", gm, ct)                              # (j) cell type
        for s in sorted(obs.state[gm].unique()):
            sm = gm & (obs.state == s).values
            if not neutral(h, f"{base}_{s}", sm, ct):                        # (i) between study
                continue
            vc = obs.donor_id[sm].value_counts()
            donors = sorted(vc[vc >= MIN_DONOR_CELLS].index)
            if len(donors) < 2:
                continue
            perm = np.random.default_rng(SEED).permutation(donors); half = len(perm) // 2
            halves = {"A": list(perm[:half]), "B": list(perm[half:])}
            hmask = {k: sm & np.isin(obs.donor_id.values, d) for k, d in halves.items()}
            if any(m.sum() < MIN_CELLS for m in hmask.values()):
                continue
            for hk, dons in halves.items():                                  # (h) donor split
                tag = f"{base}_{s}_split{hk}"
                if neutral(h, tag, hmask[hk], ct):
                    (OUT / tag / f"donors_split{hk}.txt").write_text("\n".join(dons) + "\n")


def main(sources) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for src in (sources or SOURCES):
        if src not in SOURCES:
            print(f"  unknown source {src}", flush=True); continue
        emit_source(src)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--networks-out", default=None, help="write control networks here (default: the _expressed build)")
    a = ap.parse_args()
    if a.networks_out:
        OUT = Path(a.networks_out)
    raise SystemExit(main(a.sources))
