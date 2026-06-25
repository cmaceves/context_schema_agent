"""2D scatter of the LEARNED NODE EMBEDDINGS for the inflammatory vs resident macrophage
state builds, overlaying the OpenTargets DRUG-TARGET nodes against a DEGREE-MATCHED NULL
set of nodes. This is the visual companion to the degree-matched null test: if the targets
occupy a distinct region of the embedding (vs. degree-matched controls), the embedding
carries target-relevant structure beyond out-degree; if targets and the null overlap, the
embedding mostly encodes connectivity.

embed_influence.py trains the encoder but only persists the Jacobian influence, not the
64-d node vectors z, so this retrains each state's encoder (same code, same seed ->
reproduces the saved run), takes z = encoder(A), and projects to 2D with PCA.

Per panel:
  - grey      : all other nodes (background)
  - amber     : degree-matched null nodes (one per target, sampled from the same
                out-degree decile, excluding targets; seed-locked)
  - red       : OpenTargets macrophage drug-target nodes present in the network

Output: de_ppi/results/macrophage_crohn_inflammatory/influence_analysis/state_embedding_scatter.png

Run with .venv (needs torch; build_literature_weighted_influence.py must have run for both builds):
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_state_embedding_scatter.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

DE_PPI = Path("mlp_mods/de_ppi")
sys.path.insert(0, str(DE_PPI))
from config import load_build
from embed_influence import (Encoder, BilinearDecoder, build_operator, train,
                             DIM, LAYERS, EPOCHS, LR, NEG_RATIO, HOLDOUT, SEED)

BUILDS = ["macrophage_crohn_inflammatory", "macrophage_crohn_resident"]
OT_POSITIVE = (DE_PPI.parent / "03_opentargets_rebuild"
               / "positive_proteins_EFO_0003767_pinnacle.json")


def ot_targets() -> set[str]:
    d = json.load(open(OT_POSITIVE))
    return set(d["EFO_0003767"]["macrophage"])


def degree_matched_null(nodes: pd.DataFrame, target_mask: np.ndarray,
                        outdeg: np.ndarray, rng) -> np.ndarray:
    """One control node per target, drawn from the same out-degree decile (targets excluded)."""
    bins = pd.qcut(pd.Series(outdeg).rank(method="first"), 10, labels=False).to_numpy()
    null_idx = []
    taken = set(np.where(target_mask)[0])
    for ti in np.where(target_mask)[0]:
        pool = [i for i in np.where(bins == bins[ti])[0] if i not in taken and i not in null_idx]
        if pool:
            null_idx.append(int(rng.choice(pool)))
    mask = np.zeros(len(nodes), dtype=bool)
    mask[null_idx] = True
    return mask


def embed(build: str):
    """Retrain the build's encoder (seed-locked) and return the 2D-PCA of its node embeddings."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu")
    cfg = load_build(build)
    nodes = pd.read_csv(cfg.network_nodes, sep="\t", keep_default_na=False)
    edges = pd.read_csv(cfg.network_edges, sep="\t", keep_default_na=False)
    idx = {g: i for i, g in enumerate(nodes["node_id"])}
    n = len(idx)
    A = build_operator(edges, idx, device)
    model, dec = Encoder(n, DIM, LAYERS).to(device), BilinearDecoder(DIM).to(device)
    ps = edges["source"].map(idx).to_numpy()
    pd_ = edges["target"].map(idx).to_numpy()
    print(f"[{build}] training encoder ({n} nodes, {len(edges)} edges)...", flush=True)
    train(A, model, dec, ps, pd_, n, device, EPOCHS, LR, NEG_RATIO, HOLDOUT, rng)
    model.eval()
    with torch.no_grad():
        z = model(A).cpu().numpy()
    # PCA to 2D (center + top-2 singular vectors)
    zc = z - z.mean(0)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    xy = zc @ vt[:2].T
    var = (s[:2] ** 2) / (s ** 2).sum() * 100
    nodes["x"], nodes["y"] = xy[:, 0], xy[:, 1]
    nodes["outdeg"] = nodes["node_id"].map(edges.groupby("source").size()).fillna(0).astype(int)
    return nodes, var


def main() -> int:
    targets = ot_targets()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, build in zip(axes, BUILDS):
        nodes, var = embed(build)
        rng = np.random.default_rng(0)
        tgt = nodes["node_id"].isin(targets).to_numpy()
        null = degree_matched_null(nodes, tgt, nodes["outdeg"].to_numpy(), rng)
        other = ~tgt & ~null
        ax.scatter(nodes.loc[other, "x"], nodes.loc[other, "y"],
                   s=5, c="0.8", alpha=0.45, label="other nodes", linewidths=0)
        ax.scatter(nodes.loc[null, "x"], nodes.loc[null, "y"],
                   s=42, c="#e69f00", alpha=0.85, marker="s",
                   label=f"degree-matched null (n={null.sum()})", linewidths=0)
        ax.scatter(nodes.loc[tgt, "x"], nodes.loc[tgt, "y"],
                   s=55, c="#c0392b", alpha=0.9,
                   label=f"OpenTargets targets (n={tgt.sum()})", linewidths=0)
        state = build.split("_")[-1]
        ax.set_title(f"{state} macrophage embedding\n(n={len(nodes)} nodes)")
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    out = (DE_PPI / "results" / BUILDS[0] / "images" / "state_embedding_scatter.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
