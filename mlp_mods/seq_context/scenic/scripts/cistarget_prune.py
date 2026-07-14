"""cisTarget pruning of GRNBoost2 adjacencies -> motif-supported regulons (cleaner labels).
See seq_context/SEQ_CONTEXT_EMBED.md. Run in the `pyscenic` mamba env.

Per context: raw GRNBoost2 edges.tsv (adjacencies) + that context's expression (counts.npz) -> co-expression
modules -> cisTarget motif enrichment (v10 hg38 gene-based rankings) -> regulons. Writes the pruned TF->target
edges to networks/<tag>/edges_cistarget.tsv (columns: tf, target, weight). Raw edges.tsv untouched.

Usage (pyscenic env):
  mamba run -n pyscenic python cistarget_prune.py --celltype macrophage --only <tag>   # smoke test
  mamba run -n pyscenic python cistarget_prune.py --celltype macrophage --workers 12    # all contexts
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
DB = Path("db/cistarget")
RANKINGS = ["hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
            "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"]
MOTIF_ANNOT = DB / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"


def load_expr(inp_dir, tag, genes, cap, rng):
    X = sp.load_npz(inp_dir / tag / "counts.npz").tocsr()
    if X.shape[0] > cap:
        X = X[rng.choice(X.shape[0], size=cap, replace=False)]
    return pd.DataFrame(X.toarray().astype(np.float32), columns=genes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--only", default=None)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pyscenic.utils import modules_from_adjacencies
    from pyscenic.prune import prune2df, df2regulons
    from ctxcore.rnkdb import FeatherRankingDatabase

    # macrophage inputs are flat (inputs/); other cell types under inputs/<celltype>/
    inp_dir = SEQ / "scenic/inputs" / args.celltype
    if not (inp_dir / "context_cells.tsv").exists():
        inp_dir = SEQ / "scenic/inputs"
    genes = (inp_dir / "genes.txt").read_text().split()
    cc = pd.read_csv(inp_dir / "context_cells.tsv", sep="\t")
    tags = [args.only] if args.only else list(cc.loc[cc.kept, "context"])
    dbs = [FeatherRankingDatabase(str(DB / r), name=r.split(".")[0]) for r in RANKINGS]
    rng = np.random.default_rng(args.seed)

    for tag in tags:
        out_f = NET / tag / "edges_cistarget.tsv"
        if out_f.exists() and out_f.stat().st_size > 0:                 # resume: skip done contexts
            print(f"  {tag:42s} SKIP (edges_cistarget.tsv exists)", flush=True)
            continue
        if not (NET / tag / "edges.tsv").exists():                      # overlap: GRNBoost2 not done yet
            print(f"  {tag:42s} WAIT (no edges.tsv yet)", flush=True)
            continue
        t0 = time.time()
        adj = pd.read_csv(NET / tag / "edges.tsv", sep="\t").rename(
            columns={"tf": "TF", "target": "target", "importance": "importance"})
        ex = load_expr(inp_dir, tag, genes, args.cap, rng)
        modules = list(modules_from_adjacencies(adj, ex))
        df = prune2df(dbs, modules, str(MOTIF_ANNOT),
                      client_or_address="custom_multiprocessing", num_workers=args.workers)
        regulons = df2regulons(df)
        rows = [(r.transcription_factor, g, w) for r in regulons for g, w in r.gene2weight.items()]
        out = pd.DataFrame(rows, columns=["tf", "target", "weight"])
        out.to_csv(NET / tag / "edges_cistarget.tsv", sep="\t", index=False)
        print(f"  {tag:42s} modules={len(modules):4d} regulons={len(regulons):4d} "
              f"TFs={out.tf.nunique():4d} edges={len(out):7d}  {time.time()-t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
