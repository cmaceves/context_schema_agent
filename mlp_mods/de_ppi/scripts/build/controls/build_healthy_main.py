"""MAIN healthy networks: one per (tissue, cell type, cell state), pooling NORMAL cells across ALL sources
in that tissue x cell-type group (cross-source), with exact-cell dedup. Neutral edges, detection-fraction
node inclusion, log1p(mean CP10k) expression column -- same guidelines as the disease main networks
(build_pooled_controls --pool-main). Tags: healthy_<tissue>_<celltype>_<state>.

These are TRAINED networks (the healthy arm of the shared space), not per-study controls.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/build_healthy_main.py --networks-out <dir>
"""
from __future__ import annotations

import argparse, hashlib, os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

RS = Path("mlp_mods/rank_shifts")
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
DETECT_FLOOR = 0.10
MIN_CELLS = 350
NORMAL = {"normal", "healthy"}

# (tissue, celltype) -> sources whose NORMAL cells pool into one healthy network per state
GROUPS = {
    ("ileum", "macrophage"): ["macrophage_crohn"],
    ("colon", "macrophage"): ["macrophage_crohn_colon", "macrophage_crohn_rep", "macrophage_uc_smillie"],
    ("lung", "macrophage"): ["macrophage_ild"],
    ("gut", "fibroblast"): ["fibroblast_crohn"],
    ("brain", "fibroblast"): ["fibroblast_alzheimers"],
    ("ileum", "stem"): ["stem_crohn"],
    ("brain", "microglia"): ["microglia_alzheimers"],
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


def load_normal(src):
    h5 = next((RS / f"{src}_paired").glob("pulled_*.h5ad"), None)
    states = RS / f"{src}_states" / "cell_states.tsv"
    if h5 is None or not states.exists():
        return None
    a = ad.read_h5ad(h5)
    st = pd.read_csv(states, sep="\t", index_col=0)
    if len(st) != a.n_obs:
        print(f"  WARN {src}: states {len(st)} != n_obs {a.n_obs}, skip", flush=True); return None
    a.obs["state"] = st["state"].astype(str).values
    a.obs["donor_id"] = a.obs["donor_id"].astype(str)
    h = a[a.obs.disease.astype(str).str.lower().isin(NORMAL)].copy()
    h.obs = h.obs[["state", "donor_id"]]
    if h.n_obs == 0:
        return None
    X = h.X.tocsr() if sp.issparse(h.X) else sp.csr_matrix(h.X)     # cell-level depth QC
    keep = (np.asarray(X.sum(1)).ravel() >= QC_COUNTS) & (np.asarray((X > 0).sum(1)).ravel() >= QC_GENES)
    h = h[keep].copy()
    return h if h.n_obs else None


def hash_rows(X):
    Xc = X.tocsr()
    return np.array([hashlib.sha1(Xc[i].indices.tobytes() + np.round(Xc[i].data, 3).tobytes()).hexdigest()
                     for i in range(Xc.shape[0])])


def neutral(a, tag, mask, out, celltype):
    if mask.sum() < MIN_CELLS:
        return False
    sub = a[mask]
    X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    log_expr = _expr(X, sub.var_names)
    prot = SHARED[celltype]                                      # FIXED shared node set for this cell type
    genes = set(prot)
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    d = out / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed", "direction": "",
                  "sender_weight": 1.0, "expression": log_expr.reindex(prot).fillna(0.0).values}
                 ).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    pd.DataFrame({"source": o.src, "target": o.dst, "edge_origin": "OmniPath", "edge_property": "",
                  "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"  {tag:46s} {int(mask.sum()):6d} cells -> {len(prot)} proteins", flush=True)
    return True


def main(out_dir) -> int:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for (tissue, ct), srcs in GROUPS.items():
        subs = [s for s in (load_normal(x) for x in srcs) if s is not None]
        if not subs:
            print(f"== {tissue}/{ct}: no normal cells, skip ==", flush=True); continue
        pooled = ad.concat(subs, join="outer", index_unique="-") if len(subs) > 1 else subs[0]
        X = pooled.X.tocsr() if sp.issparse(pooled.X) else sp.csr_matrix(pooled.X)
        keep = ~pd.Series(hash_rows(X)).duplicated().values        # cross-source exact-cell dedup
        pooled = pooled[keep].copy()
        print(f"== healthy {tissue}/{ct}: {pooled.n_obs} cells from {len(subs)} source(s) "
              f"({len(keep) - keep.sum()} dup cells dropped) ==", flush=True)
        for s in sorted(pooled.obs.state.astype(str).unique()):
            neutral(pooled, f"healthy_{tissue}_{ct}_{s}", (pooled.obs.state.astype(str) == s).values, out, ct)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks-out", required=True)
    raise SystemExit(main(ap.parse_args().networks_out))
