"""Generate per-context pathway-activity labels for the architecture-B auxiliary loss.

For every SCENIC context (those with edges_cistarget.tsv), load its cells (counts.npz aligned to that cell type's
genes.txt), CPM+log1p normalize, take the per-gene mean over cells. Center each gene across contexts, then score each
Reactome pathway as the mean centered-expression of its member genes present in the context (>=MIN_GENES). Z-score each
pathway across contexts. Output a (context x pathway) matrix.

This is a fast pathway-activity PROXY (mean-centered pathway expression), the lighter alternative to AUCell noted in
SEQ_CONTEXT_EMBED.md (Architecture B). Same shape/interpretation; AUCell can replace this generator without touching the
trainer (which just reads the matrix).

Run: .venv_scvi/bin/python mlp_mods/seq_context/scenic/scripts/gen_pathway_activity.py
Out: mlp_mods/seq_context/scenic/pathway_activity.tsv   (index = context tag, columns = Reactome pathways, z-scored)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
INP = SEQ / "scenic/inputs"
REACTOME = Path("mlp_mods/reactome/ReactomePathways.gmt")
MIN_SET, MAX_SET = 10, 500       # pathway size (genes)
MIN_GENES = 5                    # min pathway genes present in a context to score it
MIN_CTX_FRAC = 0.9               # keep a pathway only if scored in >= this fraction of contexts


def locate(tag):
    """Return (counts.npz path, genes.txt path) for a context tag, or (None, None)."""
    for c in NET.glob(f"**/{tag}/counts.npz"):
        pass
    hits = list(INP.glob(f"**/{tag}/counts.npz"))
    if not hits:
        return None, None
    cp = hits[0]
    genes = cp.parent.parent / "genes.txt"
    return (cp, genes) if genes.exists() else (None, None)


def context_mean_expr(cp, genes):
    X = sp.load_npz(cp).tocsr().astype(np.float64)          # cells x genes
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    X = X.multiply(1e4 / lib[:, None]).tocsr()              # CPM
    X.data = np.log1p(X.data)                               # log1p
    m = np.asarray(X.mean(0)).ravel()                       # per-gene mean over cells
    g = genes.read_text().split()
    return pd.Series(m, index=g)


def reactome_sets():
    t2g = {}
    for ln in REACTOME.read_text().splitlines():
        f = ln.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        genes = set(f[2:])
        if MIN_SET <= len(genes) <= MAX_SET:
            t2g[f[0]] = genes
    return t2g


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
    E = pd.DataFrame(exprs).T.reindex(ctx)                  # contexts x genes (union), NaN where gene absent
    Ec = E.sub(E.mean(0, skipna=True), axis=1)              # center each gene across contexts
    print(f"expr matrix: {Ec.shape[0]} ctx x {Ec.shape[1]} genes", flush=True)

    t2g = reactome_sets(); geneset = set(Ec.columns)
    cols = {}
    for pw, genes in t2g.items():
        present = list(genes & geneset)
        if len(present) < MIN_GENES:
            continue
        sub = Ec[present]                                  # contexts x present-genes
        scored = sub.notna().sum(1) >= MIN_GENES
        if scored.mean() < MIN_CTX_FRAC:
            continue
        act = sub.mean(1, skipna=True)                     # mean centered-expr over pathway genes
        cols[pw] = act.fillna(act.mean())                  # impute the few missing contexts with pathway mean
    A = pd.DataFrame(cols).reindex(ctx)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)                # z-score each pathway across contexts
    out = SEQ / "scenic/pathway_activity.tsv"
    A.to_csv(out, sep="\t")
    print(f"pathway-activity matrix: {A.shape[0]} contexts x {A.shape[1]} pathways -> {out}", flush=True)


if __name__ == "__main__":
    main()
