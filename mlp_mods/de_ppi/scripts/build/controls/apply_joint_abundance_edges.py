"""Context-specific edge weighting by JOINT ABUNDANCE (min gate).

For each network, set every directed edge's weight to min(expr_source, expr_target) using that network's OWN
node expression feature — so the message-passing OPERATOR becomes context-specific: an edge is "active" only
if BOTH partners are expressed (and pruned to ~0 if either is absent). min() is deliberately NON-separable, so
it survives the receiver-side row-normalization (a separable gate like expr_i*expr_j would cancel the receiver
factor and collapse to plain sender-abundance weighting).

Copies <src> -> <dst> and rewrites ONLY network_edges.tsv `weight` (node files / expression feature / the
edge list / direction are untouched), for BOTH main and control networks. Then: retrain the encoder on <dst>
(--expr-feat), re-infer controls, and compare to raw with compare_controls + factor_representation_compare.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/apply_joint_abundance_edges.py \
        --src crohn_alzheimer_ild_uc_embedding_expressed_combat_loc \
        --dst crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_minedge
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("mlp_mods/de_ppi/results")
EPS = 1e-2   # weight floor: keep strictly positive so the encoder's edge-weight-recon head (log weight) is finite;
             # also clip the (occasionally slightly-negative) ComBat-corrected expression to >= 0 before gating.


def gate_dir(nets_dir: Path) -> tuple[int, int, int]:
    """Rewrite each net's edge weights to max(min(expr_src, expr_dst), EPS). Returns (n_nets, n_edges, n_at_floor)."""
    n_nets = n_edges = n_floor = 0
    for d in sorted(nets_dir.iterdir()):
        nf, ef = d / "network_nodes.tsv", d / "network_edges.tsv"
        if not (nf.exists() and ef.exists()):
            continue
        nodes = pd.read_csv(nf, sep="\t", keep_default_na=False)
        expr = pd.Series(nodes["expression"].astype(float).clip(lower=0.0).values, index=nodes["node_id"].values)
        e = pd.read_csv(ef, sep="\t", keep_default_na=False)
        ws = e["source"].map(expr).fillna(0.0).to_numpy()
        wt = e["target"].map(expr).fillna(0.0).to_numpy()
        w = np.maximum(np.minimum(ws, wt), EPS)               # floor to EPS (effectively-pruned but log-finite)
        e["weight"] = w
        e.to_csv(ef, sep="\t", index=False)
        n_nets += 1
        n_edges += len(w)
        n_floor += int((w <= EPS).sum())
    return n_nets, n_edges, n_floor


def main(src: str, dst: str) -> int:
    s, d = RES / src, RES / dst
    for sub in ("networks", "controls/networks"):
        srcp, dstp = s / sub, d / sub
        if not srcp.exists():
            raise SystemExit(f"missing {srcp}")
        if dstp.exists():
            shutil.rmtree(dstp)
        shutil.copytree(srcp, dstp)
    for label, sub in (("main", "networks"), ("control", "controls/networks")):
        nn, ne, nz = gate_dir(d / sub)
        print(f"{label}: min-gated {nn} networks, {ne} edges, {nz} at floor EPS ({100*nz/max(ne,1):.1f}%)", flush=True)
    print(f"\nwrote {d}  (node files + expression feature unchanged; only edge weights = min(expr_src,expr_dst))", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    ap.add_argument("--dst", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_minedge")
    a = ap.parse_args()
    raise SystemExit(main(a.src, a.dst))
