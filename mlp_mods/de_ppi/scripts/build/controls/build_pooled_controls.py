"""Pooled control networks (a-e) for the a-g control design (see de_ppi/CONTROLS.md), emitted into
results/crohn_alzheimer_ild_uc_embedding_expressed/controls/networks/<tag>/.

ALL neutral (expression-only, CP10k>=0.5, sender_weight=1.0) so the variation comparisons are
apples-to-apples. The builder only EMITS networks; pairing them into the a-g rows (and the
within/between-study label) is done downstream by the comparison script.

Naming convention (CONTROLS.md): <disease>_<dataset8>_<tissue>_<celltype>_<cellstate>, with suffixes:
  ..._<state>            per-(dataset, state) pooled-donor network        -> serves (b) between-study, (c) state
  ..._<state>_split{A,B} donor halves of that (dataset, state)            -> serves (a) sampling floor
  ..._allstates          per-(dataset) all-states-pooled network          -> serves (d) cell-type, (e) tissue
Each donor-split dir also gets donors_split{A,B}.txt (the donor ids in each half).

Sources = the 9 rank_shifts paired atlases; each cell_states.tsv carries donor_id/disease/state/dataset_id.
The disease contrast is between-study only (control g, formed downstream across diseases); there is no
within-study disease control since no source carries two diseases in one dataset.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/build_pooled_controls.py [--sources s1 s2 ...]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, os
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
MIN_CELLS = 300       # relaxed 350->300 to admit Garrido colon-macrophage arms (healthy ~317 cells)
MIN_DONOR_CELLS = 20
SEED = 0

# source -> (disease_slug, tissue, celltype_slug). h5ad/states paths derived from the source name.
SOURCES = {
    "microglia_alzheimers":   ("alz",   "brain", "microglia"),
    "fibroblast_alzheimers":  ("alz",   "brain", "fibroblast"),
    "macrophage_crohn":       ("crohn", "ileum", "macrophage"),
    "macrophage_crohn_colon": ("crohn", "colon", "macrophage"),
    "macrophage_garrido_crohn": ("crohn", "colon", "macrophage"),   # 2nd Crohn colon study (Garrido GSE214695)
    "macrophage_garrido_uc":    ("uc",    "colon", "macrophage"),   # 2nd-source UC colon (Garrido GSE214695)
    "fibroblast_crohn":       ("crohn", "gut",   "fibroblast"),
    "stem_crohn":             ("crohn", "ileum", "stem"),
    "macrophage_uc_smillie":  ("uc",    "colon", "macrophage"),
    "macrophage_ild":         ("ild",   "lung",  "macrophage"),
}

op = pd.read_csv(OMNI, sep="\t")
QC_COUNTS, QC_GENES = 500, 300     # cell-level depth QC
PSEUDOBULK = os.environ.get("DE_PPI_PSEUDOBULK") == "1"   # depth-weighted expression feature (vs mean CP10k)
SHARED = {p.stem: sorted(p.read_text().split())                # fixed per-celltype node set (build_shared_nodes.py)
          for p in Path("mlp_mods/de_ppi/shared_nodes").glob("*.txt")}


def _expr(X, var_names):
    """per-gene log1p expression feature. PSEUDOBULK: sum raw counts then CP10k-normalize once
    (depth-weighted, dropout-robust). else: mean of per-cell CP10k (dropout-sensitive)."""
    if PSEUDOBULK:
        g = np.asarray(X.sum(0)).ravel(); tt = g.sum() or 1.0
        s = pd.Series(np.log1p(1e4 * g / tt), index=pd.Index(var_names))
    else:
        tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
        s = pd.Series(np.log1p(np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()), index=pd.Index(var_names))
    return s[~s.index.duplicated()]


def load_source(src):
    h5 = next((RS / f"{src}_paired").glob("pulled_*.h5ad"), None)
    states = RS / f"{src}_states" / "cell_states.tsv"
    if h5 is None or not states.exists():
        print(f"  skip {src}: missing h5ad or cell_states.tsv", flush=True)
        return None
    a = ad.read_h5ad(h5)
    st = pd.read_csv(states, sep="\t", index_col=0)
    assert len(st) == a.n_obs, f"{src}: cell_states rows {len(st)} != n_obs {a.n_obs}"
    a.obs["state"] = st["state"].astype(str).values                # positional (cell_states written in obs order)
    a.obs["dataset_id"] = st["dataset_id"].astype(str).values
    a.obs["donor_id"] = a.obs["donor_id"].astype(str)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)     # cell-level depth QC
    keep = (np.asarray(X.sum(1)).ravel() >= QC_COUNTS) & (np.asarray((X > 0).sum(1)).ravel() >= QC_GENES)
    return a[keep].copy()


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
    print(f"  {tag:52s} {int(mask.sum()):6d} cells -> {len(prot)} proteins", flush=True)
    return True


NORMAL = {"normal", "healthy"}


def group_studies(obs) -> np.ndarray:
    """dataset_id -> study8 (donor-overlap connected components; label = largest-cell member). Matches the
    healthy controls so disease between-study (b) never compares two depositions of the same collection."""
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
    label = {}
    for members in groups.values():
        rep = max(members, key=lambda m: cnt.get(m, 0))[:8]
        for m in members:
            label[m] = rep
    return obs.dataset_id.map(label).values


def emit_source(src, pool_main=False):
    dz, tissue, ct = SOURCES[src]
    a = load_source(src)
    if a is None:
        return
    obs = a.obs
    is_dis = (~obs.disease.astype(str).str.lower().isin(NORMAL)).values   # DISEASE cells only (was unfiltered = bug)
    if pool_main:                                                         # 30 main networks: per state, pool studies
        print(f"== {src} MAIN (disease cells, studies pooled) ==", flush=True)
        for s in sorted(pd.unique(obs.state[is_dis])):
            neutral(a, f"{dz}_{tissue}_{ct}_{s}", is_dis & (obs.state == s).values, ct)
        return
    print(f"== {src} per-study (disease cells, donor-overlap grouped) ({dz}/{tissue}/{ct}) ==", flush=True)
    study = np.array([""] * len(obs), dtype=object)
    di = np.where(is_dis)[0]
    study[di] = group_studies(obs.iloc[di])              # collapse donor-sharing datasets into one study
    for grp in sorted(set(study[di])):
        dm = is_dis & (study == grp)
        base = f"{dz}_{grp}_{tissue}_{ct}"
        # (d/e) all-states-pooled per dataset
        neutral(a, f"{base}_allstates", dm, ct)
        for s in sorted(obs.state[dm].unique()):
            sm = dm & (obs.state == s).values
            # (b/c) per-(dataset, state) pooled-donor network
            if not neutral(a, f"{base}_{s}", sm, ct):
                continue
            # (a) donor split of that (dataset, state)
            vc = obs.donor_id[sm].value_counts()
            donors = sorted(vc[vc >= MIN_DONOR_CELLS].index)
            if len(donors) < 2:
                continue
            perm = np.random.default_rng(SEED).permutation(donors); h = len(perm) // 2
            halves = {"A": list(perm[:h]), "B": list(perm[h:])}
            hmask = {k: sm & np.isin(obs.donor_id.values, dons) for k, dons in halves.items()}
            if any(m.sum() < MIN_CELLS for m in hmask.values()):
                continue                                           # need BOTH halves; orphan split is unpairable
            for half, dons in halves.items():
                tag = f"{base}_{s}_split{half}"
                if neutral(a, tag, hmask[half], ct):
                    (OUT / tag / f"donors_split{half}.txt").write_text("\n".join(dons) + "\n")


def main(sources, pool_main=False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for src in (sources or SOURCES):
        if src not in SOURCES:
            print(f"  unknown source {src} (known: {', '.join(SOURCES)})", flush=True); continue
        emit_source(src, pool_main=pool_main)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=None, help="subset of sources (default: all)")
    ap.add_argument("--networks-out", default=None, help="write control networks here (default: the _expressed build)")
    ap.add_argument("--pool-main", action="store_true",
                    help="build the ~30 MAIN networks: disease cells, per (disease,tissue,celltype,state), "
                         "studies POOLED (no per-dataset split). Default builds per-study disease controls.")
    a = ap.parse_args()
    if a.networks_out:
        OUT = Path(a.networks_out)
    raise SystemExit(main(a.sources, a.pool_main))
