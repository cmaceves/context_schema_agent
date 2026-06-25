"""Flattening check: compare a network's JOINT embedding (trained in the shared encoder over many
networks) against a freshly-trained SINGLE-network embedding of the same graph.

If the shared encoder is flattening per-network structure, the joint embedding of network g should
(a) occupy fewer effective dimensions, (b) reorganize local neighborhoods, and/or (c) separate g's
own edges less well than a dedicated single-network embedding. We use alignment-invariant metrics
(no need to align the two spaces):

  eff_dim       participation ratio (Sigma lam)^2 / Sigma lam^2 of the embedding covariance
                (1..DIM; lower = more collapsed). Compare single vs joint.
  knn_jaccard   mean Jaccard overlap of each node's k nearest neighbours between single & joint
                (1 = identical local structure; low = reorganized/flattened)
  edge_auc      decoder-free edge reconstruction: cosine(z_i,z_j) on g's real edges vs random
                negatives -> ROC-AUC. Compare single vs joint (joint < single = lost g-structure).

Reads networks from results/<out_name>/networks/<tag>/ and the joint embeddings from
results/<out_name>/embeddings.npz.

Run with .venv:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/compare_joint_vs_single.py \
      --out-name crohn_alzheimer_embedding --tags crohn_mac_inflammatory crohn_mac_resident crohn_mac_proliferating
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

DE_PPI = Path("mlp_mods/de_ppi")
sys.path.insert(0, str(DE_PPI))
from embed_influence import Encoder, BilinearDecoder, build_operator, train, \
    DIM, LAYERS, EPOCHS, LR, NEG_RATIO, HOLDOUT, SEED


def eff_dim(Z):
    """Participation ratio of the embedding covariance eigenvalues (1..dim; lower = collapsed)."""
    lam = np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T))
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def knn_jaccard(A, B, k=15):
    """Mean Jaccard overlap of each row's k-NN neighbour set between embeddings A and B (same rows)."""
    nA = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(return_distance=False)[:, 1:]
    nB = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(return_distance=False)[:, 1:]
    j = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(nA, nB)]
    return float(np.mean(j))


def edge_auc(Z, src, dst, rng):
    """Decoder-free: cosine(z_src,z_dst) on real edges vs random negatives -> ROC-AUC."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    pos = (Zn[src] * Zn[dst]).sum(1)
    neg = (Zn[src] * Zn[rng.integers(0, len(Z), len(dst))]).sum(1)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


def main(out_name, tags, seed=SEED) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = DE_PPI / "results" / out_name
    (root / "tables").mkdir(parents=True, exist_ok=True); (root / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(root / "embeddings.npz", allow_pickle=True)
    jt_id, jt_tags, jt_Z, jt_present = d["node_id"], list(d["tags"]), d["Z"], d["present"]

    rows = []
    for tag in tags:
        nd = pd.read_csv(root / "networks" / tag / "network_nodes.tsv", sep="\t", keep_default_na=False)
        ed = pd.read_csv(root / "networks" / tag / "network_edges.tsv", sep="\t", keep_default_na=False)
        ids = nd["node_id"].tolist()
        idx = {g: i for i, g in enumerate(ids)}
        n = len(ids)
        sw = nd["sender_weight"].astype(float).to_numpy()
        A = build_operator(ed, idx, device, self_weight=sw)
        src = ed["source"].map(idx).to_numpy(); dst = ed["target"].map(idx).to_numpy()

        # single-network embedding (same hyperparams/seed as the joint encoder)
        torch.manual_seed(seed)
        model = Encoder(n, DIM, LAYERS).to(device); dec = BilinearDecoder(DIM).to(device)
        print(f"[{tag}] training single-network encoder ...", flush=True)
        train(A, model, dec, src, dst, n, device, EPOCHS, LR, NEG_RATIO, HOLDOUT, np.random.default_rng(seed))
        with torch.no_grad():
            Z_single = model(A).cpu().numpy()

        # joint embedding for this tag, aligned to the single network's node order
        ti = jt_tags.index(tag)
        jrow = {g: i for i, g in enumerate(jt_id)}
        Z_joint = jt_Z[ti][[jrow[g] for g in ids]]

        ed_single, ed_joint = eff_dim(Z_single), eff_dim(Z_joint)
        jac = knn_jaccard(Z_single, Z_joint)
        auc_single = edge_auc(Z_single, src, dst, np.random.default_rng(0))
        auc_joint = edge_auc(Z_joint, src, dst, np.random.default_rng(0))
        rows.append((tag, n, ed_single, ed_joint, jac, auc_single, auc_joint))
        print(f"  eff_dim single={ed_single:.1f} joint={ed_joint:.1f} | knn_jaccard={jac:.3f} | "
              f"edge_auc single={auc_single:.3f} joint={auc_joint:.3f}", flush=True)

    df = pd.DataFrame(rows, columns=["tag", "n_nodes", "eff_dim_single", "eff_dim_joint",
                                     "knn_jaccard", "edge_auc_single", "edge_auc_joint"]).round(3)
    out = root / "tables" / "joint_vs_single_flattening.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print("\n=== joint vs single (flattening) ===")
    print(df.to_string(index=False))
    print(f"\nwrote {out}", flush=True)
    print("read: eff_dim_joint << single OR low knn_jaccard OR edge_auc_joint << single => flattening", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="joint vs single-network embedding flattening check")
    ap.add_argument("--out-name", default="crohn_alzheimer_embedding")
    ap.add_argument("--tags", nargs="+",
                    default=["crohn_mac_inflammatory", "crohn_mac_resident", "crohn_mac_proliferating"])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.tags, a.seed))
