"""Control m: HEALTHY leave-one-study-out (LOO) pools, to evaluate whether POOLING studies reduces the
between-study expression batch. For each (tissue, celltype, state) with >= MIN_STUDIES independent healthy
study groups, and for each held-out group g, emit two networks into controls/networks/:

  healthy_loopool<g8>_<tissue>_<celltype>_<state>    = pooled cells of ALL groups EXCEPT g  (N-1 studies)
  healthy_loosingle<g8>_<tissue>_<celltype>_<state>  = cells of group g alone               (held-out study)

Downstream (control_m_healthy_loo.py) pairs loopool<g> vs loosingle<g>: if pooling reduces batch, this
cosine sits ABOVE the single-vs-single floor (control i) and climbs toward the within-study ceiling (h).
Both members are built here so study-group ids always match. Same recipe as the other controls: cell-depth
QC, cross-source pool + exact-cell dedup, FIXED shared node set per cell type, log1p(mean CP10k) expression.

NOTE: pairing/scoring is done by control_m_healthy_loo.py; compare_controls.py ignores 'loo' tags.
Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/build_healthy_loo.py --networks-out <dir>
"""
from __future__ import annotations

import argparse, hashlib, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

RS = Path("mlp_mods/rank_shifts")
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
QC_COUNTS, QC_GENES = 500, 300
MIN_CELLS = 350
MIN_STUDIES = 3                  # need >=3 healthy study groups in a context to leave one out
NORMAL = {"normal", "healthy"}
SHARED = {p.stem: sorted(p.read_text().split()) for p in Path("mlp_mods/de_ppi/shared_nodes").glob("*.txt")}
PSEUDOBULK = os.environ.get("DE_PPI_PSEUDOBULK") == "1"
op = pd.read_csv(OMNI, sep="\t")


def _expr(X, var_names):
    """log1p expression: pseudobulk (sum counts, normalize once; depth-weighted) if PSEUDOBULK else mean CP10k."""
    if PSEUDOBULK:
        g = np.asarray(X.sum(0)).ravel(); tt = g.sum() or 1.0
        s = pd.Series(np.log1p(1e4 * g / tt), index=pd.Index(var_names))
    else:
        tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
        s = pd.Series(np.log1p(np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()), index=pd.Index(var_names))
    return s[~s.index.duplicated()]

GROUPS = {
    ("ileum", "macrophage"): ["macrophage_crohn"],
    ("colon", "macrophage"): ["macrophage_crohn_colon", "macrophage_crohn_rep", "macrophage_uc_smillie"],
    ("lung", "macrophage"): ["macrophage_ild"],
    ("gut", "fibroblast"): ["fibroblast_crohn"],
    ("brain", "fibroblast"): ["fibroblast_alzheimers"],
    ("ileum", "stem"): ["stem_crohn"],
    ("brain", "microglia"): ["microglia_alzheimers"],
}


def hash_rows(X):
    Xc = X.tocsr()
    return np.array([hashlib.sha1(Xc[i].indices.tobytes() + np.round(Xc[i].data, 3).tobytes()).hexdigest()
                     for i in range(Xc.shape[0])])


def load_normal(src):
    h5 = next((RS / f"{src}_paired").glob("pulled_*.h5ad"), None)
    states = RS / f"{src}_states" / "cell_states.tsv"
    if h5 is None or not states.exists():
        return None
    a = ad.read_h5ad(h5)
    st = pd.read_csv(states, sep="\t", index_col=0)
    if len(st) != a.n_obs:
        return None
    a.obs["state"] = st["state"].astype(str).values
    a.obs["dataset_id"] = st["dataset_id"].astype(str).values
    a.obs["donor_id"] = a.obs["donor_id"].astype(str)
    h = a[a.obs.disease.astype(str).str.lower().isin(NORMAL)].copy()
    if h.n_obs == 0:
        return None
    X = h.X.tocsr() if sp.issparse(h.X) else sp.csr_matrix(h.X)
    keep = (np.asarray(X.sum(1)).ravel() >= QC_COUNTS) & (np.asarray((X > 0).sum(1)).ravel() >= QC_GENES)
    h = h[keep].copy()
    h.obs = h.obs[["state", "dataset_id", "donor_id"]]
    return h if h.n_obs else None


def group_studies(obs) -> np.ndarray:
    """dataset_id -> study8 (donor-overlap connected components; label = largest-cell member)."""
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
    out = {}
    for members in groups.values():
        rep = max(members, key=lambda m: cnt.get(m, 0))[:8]
        for m in members:
            out[m] = rep                                 # key on FULL dataset_id (matches other builders)
    return obs.dataset_id.map(out).values


def neutral(a, tag, mask, celltype, out):
    if mask.sum() < MIN_CELLS:
        return False
    sub = a[mask]
    X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    log_expr = _expr(X, sub.var_names)
    prot = SHARED[celltype]; genes = set(prot)
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    d = out / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed", "direction": "",
                  "sender_weight": 1.0, "expression": log_expr.reindex(prot).fillna(0.0).values}
                 ).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    pd.DataFrame({"source": o.src, "target": o.dst, "edge_origin": "OmniPath", "edge_property": "",
                  "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"  {tag:48s} {int(mask.sum()):6d} cells -> {len(prot)} proteins", flush=True)
    return True


def main(out_dir) -> int:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for (tissue, ct), srcs in GROUPS.items():
        subs = [s for s in (load_normal(x) for x in srcs) if s is not None]
        if not subs:
            continue
        pooled = ad.concat(subs, join="outer", index_unique="-") if len(subs) > 1 else subs[0]
        X = pooled.X.tocsr() if sp.issparse(pooled.X) else sp.csr_matrix(pooled.X)
        pooled = pooled[~pd.Series(hash_rows(X)).duplicated().values].copy()      # cross-source dedup
        pooled.obs["study"] = group_studies(pooled.obs)
        for state in sorted(pooled.obs.state.astype(str).unique()):
            sm = (pooled.obs.state.astype(str) == state).values
            sc = pooled.obs.study[sm].value_counts()
            groups = sorted(sc[sc >= MIN_CELLS].index)               # study groups with enough cells in this state
            if len(groups) < MIN_STUDIES:
                continue
            print(f"== {tissue}/{ct}/{state}: {len(groups)} healthy study groups -> LOO ==", flush=True)
            studies = pooled.obs.study.values
            for g in groups:
                base = f"{tissue}_{ct}_{state}"
                neutral(pooled, f"healthy_loopool{g}_{base}", sm & (studies != g) & np.isin(studies, groups), ct, out)
                neutral(pooled, f"healthy_loosingle{g}_{base}", sm & (studies == g), ct, out)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks-out", required=True)
    raise SystemExit(main(ap.parse_args().networks_out))
