"""All-vs-all network similarity heatmap: for every pair of networks (disease/cell/state), the
value is the PERCENT of shared proteins (present in both graphs) whose embedding shift is exactly 0
-- i.e. how interchangeable the two networks are. High % = the two contexts place most shared
proteins identically (a disease-invariant backbone); low % = the networks reorganize many proteins.

Rows/cols are ordered by hierarchical clustering on (100 - %zero) so similar networks group,
revealing which disease/cell/state networks converge regardless of disease.

Output: results/<out_name>/influence_analysis/pair_zero_shift_similarity_heatmap.png

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_pair_similarity_heatmap.py \
      --out-name crohn_alzheimer_ild_embedding
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import order_tags, block_separators, annotate_hierarchy   # shared cell-type -> tissue -> state ordering


# metric -> (per-pair reducer over the shared-protein shift vector, is_similarity, cmap, label, fmt)
METRICS = {
    "mean":         (lambda v: float(v.mean()),              False, "viridis",    "mean embedding shift",            "{:.2f}"),
    "mean_nonzero": (lambda v: float(v[v > 0].mean()) if (v > 0).any() else 0.0,
                                                             False, "viridis",    "mean shift (nonzero only)",       "{:.2f}"),
}


def main(out_name, metric="mean", restrict="all") -> int:
    reducer, is_sim, cmap_name, clabel, fmt = METRICS[metric]
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    sh = pd.read_csv(res / "embedding_shift.tsv", sep="\t")
    tags = [c[len("present_"):] for c in sh.columns
            if c.startswith("present_") and c != "present_all"]
    tagset = set(tags)
    de_by_tag = None
    if restrict == "de":     # per-pair: DE-significant (padj<0.05, 'de' source tag) in EITHER of the two networks
        de_by_tag = {}
        for t in tags:
            n = pd.read_csv(res / "networks" / t / "network_nodes.tsv", sep="\t", keep_default_na=False)
            is_de = n["source"].astype(str).apply(lambda s: "de" in s.split("|"))
            de_by_tag[t] = set(n.loc[is_de, "node_id"])
        print(f"restrict=de (per-pair, DE-in-either): per-tag DE counts "
              f"{ {t: len(s) for t, s in list(de_by_tag.items())[:3]} } ...", flush=True)

    def parse_pair(c):
        s = c[len("shift_"):]
        for a in tags:
            if s.startswith(a + "_") and s[len(a) + 1:] in tagset:
                return a, s[len(a) + 1:]
        return None

    sh = sh[sh.node_type == "protein"]
    V = pd.DataFrame(np.nan, index=tags, columns=tags, dtype=float)
    nshared = pd.DataFrame(0, index=tags, columns=tags, dtype=int)
    for c in sh.columns:
        if not c.startswith("shift_"):
            continue
        ab = parse_pair(c)
        if not ab:
            continue
        a, b = ab
        col = sh[c]
        if de_by_tag is not None:                          # restrict to genes DE in network a OR network b
            col = col[sh.node_id.isin(de_by_tag[a] | de_by_tag[b])]
        v = col.dropna()
        if len(v) == 0:
            continue
        V.loc[a, b] = V.loc[b, a] = reducer(v)
        nshared.loc[a, b] = nshared.loc[b, a] = len(v)

    tags_o = order_tags(tags)                              # cell type -> tissue -> state
    P = V.loc[tags_o, tags_o]
    N = nshared.loc[tags_o, tags_o]                         # per-cell protein count (n DE genes used)

    vmax = float(np.nanpercentile(V.values, 98))
    sz = max(12, 0.55 * len(tags_o))                       # scale figure with #networks so cells stay legible
    fig, ax = plt.subplots(figsize=(sz, sz))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("0.85")
    im = ax.imshow(np.ma.masked_invalid(P.values), cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(tags_o))); ax.set_xticklabels(tags_o, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tags_o))); ax.set_yticklabels(tags_o, fontsize=8)
    show_n = de_by_tag is not None                         # n only on the DE-restricted variant
    for i in range(len(tags_o)):
        for j in range(len(tags_o)):
            val = P.values[i, j]
            if np.isfinite(val):
                txt = f"{fmt.format(val)}\nn={int(N.values[i, j])}" if show_n else fmt.format(val)
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5 if show_n else 7,
                        color="white" if val < vmax * 0.55 else "black")
    ct_b, ti_b = block_separators(tags_o)                  # cell-type (thick) + tissue (thin) separators
    for k in ct_b:
        ax.axhline(k - 0.5, color="white", lw=1.8); ax.axvline(k - 0.5, color="white", lw=1.8)
    for k in ti_b:
        ax.axhline(k - 0.5, color="white", lw=0.7); ax.axvline(k - 0.5, color="white", lw=0.7)
    annotate_hierarchy(ax, tags_o)                         # cell-type (bold) + tissue brackets on top
    rtag = "  [per-pair: genes DE in either network]" if de_by_tag is not None else ""
    ax.set_title(f"Network shift heatmap ({out_name})\n{clabel} between network pairs "
                 f"(blocked by cell type → tissue → state){rtag}", pad=60)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=clabel)
    sfx = "_depair" if de_by_tag is not None else ""
    out = res / "images" / f"pair_shift_heatmap_{metric}{sfx}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)

    P.to_csv(res / "tables" / f"pair_shift_heatmap_{metric}{sfx}.tsv", sep="\t")
    tri = V.where(np.triu(np.ones(V.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    end = "most similar" if is_sim else "most different (largest shift)"
    print(f"\n{end} pairs by {metric}:")
    for (a, b), v in tri.head(10).items():
        print(f"  {fmt.format(v):>6}   {a}  <->  {b}  (n_shared={nshared.loc[a,b]})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="All-vs-all network shift heatmap")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_embedding")
    ap.add_argument("--metric", default="mean", choices=list(METRICS),
                    help="per-pair value: pct_zero | mean | median | mean_nonzero")
    ap.add_argument("--restrict", default="all", choices=["all", "de"],
                    help="protein set: all, or DE-union (dysregulated in any network)")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.metric, a.restrict))
