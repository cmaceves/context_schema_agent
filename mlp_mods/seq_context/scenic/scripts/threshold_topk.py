"""Threshold raw GRNBoost2 adjacency -> top-k regulatory neighbors per TF (link-prediction positive labels).
See seq_context/SEQ_CONTEXT_EMBED.md. Pure pandas (run in .venv_scvi).

For each context under seq_context/scenic/networks/<tag>/edges.tsv (raw ranked tf,target,importance), keep each
TF's top-K targets by importance and write edges_topk.tsv beside it (raw edges.tsv untouched). K is FIXED across
all contexts so context-lift comparisons are fair. Prints per-context edge counts + density.

Usage: .venv_scvi/bin/python threshold_topk.py [--k 50]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    rows = []
    for d in sorted(p for p in NET.iterdir() if (p / "edges.tsv").exists()):
        raw = pd.read_csv(d / "edges.tsv", sep="\t")
        # rank targets within each TF by importance, keep top-K
        top = (raw.sort_values("importance", ascending=False)
                  .groupby("tf", sort=False).head(args.k)
                  .reset_index(drop=True))
        top.to_csv(d / "edges_topk.tsv", sep="\t", index=False)
        n_tf = top.tf.nunique()
        rows.append({"context": d.name, "raw_edges": len(raw), "topk_edges": len(top),
                     "n_tf": n_tf, "mean_targets_per_tf": round(len(top) / max(n_tf, 1), 1)})
        print(f"  {d.name:42s} raw={len(raw):8d} -> topk={len(top):7d}  tfs={n_tf:4d}  "
              f"~{len(top)/max(n_tf,1):4.1f} targets/tf", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(NET.parent / "topk_summary.tsv", sep="\t", index=False)
    print(f"\nk={args.k}  contexts={len(rows)}  "
          f"topk edges: min={df.topk_edges.min()} max={df.topk_edges.max()} "
          f"median={int(df.topk_edges.median())}", flush=True)


if __name__ == "__main__":
    main()
