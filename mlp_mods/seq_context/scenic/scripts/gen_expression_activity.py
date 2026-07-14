"""Generate per-context per-gene expression labels for the architecture-B expression auxiliary loss.
Per context: CPM+log1p of counts.npz, per-gene mean over cells. Z-score each gene across contexts. Output a
(context x gene) matrix. Trainer target for (protein p, context c) = expression[c, p] (masked to genes present).

Run: .venv_scvi/bin/python mlp_mods/seq_context/scenic/scripts/gen_expression_activity.py
Out: mlp_mods/seq_context/scenic/expression_activity.tsv   (index = context tag, columns = genes, z-scored)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
INP = SEQ / "scenic/inputs"
MIN_CTX_FRAC = 0.5               # keep a gene only if measured in >= this fraction of contexts


def locate(tag):
    hits = list(INP.glob(f"**/{tag}/counts.npz"))
    if not hits:
        return None, None
    cp = hits[0]; genes = cp.parent.parent / "genes.txt"
    return (cp, genes) if genes.exists() else (None, None)


def context_mean_expr(cp, genes):
    X = sp.load_npz(cp).tocsr().astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    X = X.multiply(1e4 / lib[:, None]).tocsr(); X.data = np.log1p(X.data)
    return pd.Series(np.asarray(X.mean(0)).ravel(), index=genes.read_text().split())


def main():
    tags = sorted({p.parent.name for p in NET.glob("*/edges_cistarget.tsv")})
    print(f"contexts: {len(tags)}", flush=True)
    exprs = {}
    for t in tags:
        cp, gp = locate(t)
        if cp is None:
            print(f"  WARN no counts for {t}", flush=True); continue
        exprs[t] = context_mean_expr(cp, gp)
    ctx = sorted(exprs)
    E = pd.DataFrame(exprs).T.reindex(ctx)                          # contexts x genes (union), NaN where absent
    keep = E.notna().mean(0) >= MIN_CTX_FRAC
    E = E.loc[:, keep]
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)                          # z-score each gene across contexts
    out = SEQ / "scenic/expression_activity.tsv"
    E.to_csv(out, sep="\t")
    print(f"expression matrix: {E.shape[0]} contexts x {E.shape[1]} genes -> {out}", flush=True)


if __name__ == "__main__":
    main()
