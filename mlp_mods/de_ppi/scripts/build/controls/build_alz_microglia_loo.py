"""LEAVE-ONE-OUT batch probe for the (already pooled) AD-microglia build. The current alz_microglia_*
networks pool 3 datasets (203025fe, ac0c6561, cff99df2). Here we rebuild each microglia STATE three
times, each leaving ONE dataset out (so each LOO uses the other two), as NEUTRAL expression-only
networks (cleanest apples-to-apples: the LOO variants differ only in which studies' cells defined the
expressed node set). Comparing alz_microglia_<state>_loo{1,2,3} pairwise asks: is the pooled AD signal
robust to which studies are included (replicated), or does dropping one study move it a lot (batch /
one-study-driven)?

Reuses the pooled Leiden state assignments (microglia_alzheimers_states/cell_states.tsv) so the state
DEFINITION is held fixed; only the dataset membership of the expressed set changes. Networks are written
into the expressed embedding; re-embed + regenerate after.

Run: .venv/bin/python mlp_mods/de_ppi/build_alz_microglia_loo.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path("mlp_mods/de_ppi"); RS = Path("mlp_mods/rank_shifts")
NET = HERE / "results/crohn_alzheimer_ild_uc_embedding_expressed/networks"
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
H5 = RS / "microglia_alzheimers_paired/pulled_microglia.h5ad"
STATES = RS / "microglia_alzheimers_states/cell_states.tsv"
DATASETS = ["203025fe-fa99-4d57-81da-458ed8f0c334",
            "ac0c6561-7a48-4185-af6f-af799f699172",
            "cff99df2-4904-44f7-9173-ff837f95606e"]
CP10K_CUTOFF = 0.5
MIN_CELLS = 50


def expressed(a, mask):
    sub = a[mask]; X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    meancp = np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()
    return sorted(pd.Index(sub.var_names)[meancp >= CP10K_CUTOFF])


def build_neutral(a, tag, mask, op):
    genes = set(expressed(a, mask))
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    inc = genes & (set(o.src) | set(o.dst)); prot = sorted(inc); d = NET / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed", "direction": "", "sender_weight": 1.0}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    e = o[o.src.isin(inc) & o.dst.isin(inc)]
    pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath", "edge_property": "", "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"  {tag:42s} {int(mask.sum()):6d} cells -> {len(prot)} proteins, {len(e)} edges", flush=True)


def main():
    a = ad.read_h5ad(H5)
    st = pd.read_csv(STATES, sep="\t", index_col=0)
    assert len(st) == a.n_obs, f"states {len(st)} != cells {a.n_obs}"
    ds = st["dataset_id"].astype(str).values
    state = st["state"].astype(str).values
    op = pd.read_csv(OMNI, sep="\t")
    for i, leave_out in enumerate(DATASETS, start=1):
        keep = ds != leave_out
        print(f"\nLOO{i}: leave out {leave_out[:8]} -> {keep.sum()} cells from the other 2", flush=True)
        for s in sorted(set(state)):
            mask = keep & (state == s)
            if mask.sum() < MIN_CELLS:
                print(f"  skip {s} (only {int(mask.sum())} cells)", flush=True); continue
            build_neutral(a, f"alz_microglia_{s}_loo{i}", mask, op)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
