"""GRNBoost2 per context -> regulatory-neighbor edge lists (link-prediction labels).
See de_ppi/SEQ_CONTEXT_EMBED.md. Run in .venv_scenic (arboreto + dask).

For each context dir under seq_context/scenic/inputs/<tag>/ (counts.npz, aligned to inputs/genes.txt), run
GRNBoost2 with TFs = inputs/tfs.txt (restricted to genes expressed in the context). Large contexts are
subsampled to --cap cells for tractability. Writes seq_context/scenic/networks/<tag>/edges.tsv
(columns: tf, target, importance) — the full ranked co-expression network; thresholding happens downstream.

Usage:
  .venv_scenic/bin/python run_grnboost2.py                 # all contexts
  .venv_scenic/bin/python run_grnboost2.py --only <tag>    # one context (smoke test)
  .venv_scenic/bin/python run_grnboost2.py --cap 5000 --seed 0 --workers 8
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEQ = Path("mlp_mods/seq_context")
INP = SEQ / "scenic/inputs"                 # set per --celltype in main()
OUT = SEQ / "scenic/networks"               # flat; context tags are unique across cell types
MIN_CELLS_EXPR = 3  # a gene must be detected in >= this many cells in the context to be a node


def load_context(tag, genes):
    d = INP / tag
    X = sp.load_npz(d / "counts.npz").tocsc()  # cells x genes
    return X


def run_one(tag, genes, tfs_all, cap, seed, client):
    from arboreto.algo import grnboost2
    d_out = OUT / tag
    ef = d_out / "edges.tsv"
    if ef.exists() and ef.stat().st_size > 0:                # resume: skip already-built contexts
        print(f"  {tag:42s} SKIP (edges.tsv exists, {ef.stat().st_size//1_000_000} MB)", flush=True)
        return
    d_out.mkdir(parents=True, exist_ok=True)
    X = load_context(tag, genes)                          # cells x genes (csc)
    rng = np.random.default_rng(seed)
    if X.shape[0] > cap:
        sel = rng.choice(X.shape[0], size=cap, replace=False)
        X = X[sel]
    # keep genes expressed in this context (nonzero in >= MIN_CELLS_EXPR cells)
    det = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = det >= MIN_CELLS_EXPR
    gkeep = [g for g, k in zip(genes, keep) if k]
    Xk = X[:, keep].toarray().astype(np.float32)
    tfs = [g for g in tfs_all if g in set(gkeep)]
    t0 = time.time()
    df = pd.DataFrame(Xk, columns=gkeep)
    net = grnboost2(expression_data=df, tf_names=tfs, client_or_address=client, verbose=False, seed=seed)
    net.columns = ["tf", "target", "importance"]
    net.to_csv(d_out / "edges.tsv", sep="\t", index=False)
    print(f"  {tag:42s} cells={Xk.shape[0]:5d} genes={len(gkeep):5d} tfs={len(tfs):4d} "
          f"edges={len(net):7d}  {time.time()-t0:6.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--only", default=None)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    global INP
    INP = SEQ / "scenic/inputs" / args.celltype

    genes = (INP / "genes.txt").read_text().split()
    tfs_all = (INP / "tfs.txt").read_text().split()
    cc = pd.read_csv(INP / "context_cells.tsv", sep="\t")
    tags = [args.only] if args.only else list(cc.loc[cc.kept, "context"])

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1, memory_limit=0, dashboard_address=None)
    client = Client(cluster)
    print(f"dask: {args.workers} workers | cap={args.cap} | {len(tags)} contexts", flush=True)
    try:
        for tag in tags:
            run_one(tag, genes, tfs_all, args.cap, args.seed, client)
    finally:
        client.close(); cluster.close()


if __name__ == "__main__":
    main()
