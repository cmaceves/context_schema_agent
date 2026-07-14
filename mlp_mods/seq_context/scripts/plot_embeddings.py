"""UMAP of the context-specific protein embeddings, colored by composite context factors.
See seq_context/SEQ_CONTEXT_EMBED.md. Each point = one (protein, context) 128-d embedding.
Two panels: (1) disease × tissue, (2) disease × tissue × cell_state. whitegrid, varied palette.

Usage: .venv_scvi/bin/python plot_embeddings.py --run link_v4_cistarget
Out: images/umap_embeddings_<run>.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap

SEQ = Path("mlp_mods/seq_context")


def parse(tag):
    a = tag.split("_")
    return dict(disease=a[0], tissue=a[1], cell_type=a[2], state="_".join(a[3:]))


def panel(ax, xy, labels, title, pal):
    cats = sorted(set(labels))
    colors = dict(zip(cats, sns.color_palette(pal, len(cats))))
    for c in cats:
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.35, color=colors[c], label=c, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=12); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    lg = ax.legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    for h in (getattr(lg, "legend_handles", None) or lg.legendHandles):
        h.set_alpha(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v4_cistarget")
    args = ap.parse_args()
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    emb, ctx = d["emb"], d["context"]
    p = [parse(t) for t in ctx]
    dtc = np.array([f"{x['disease']}_{x['tissue']}_{x['cell_type']}" for x in p])
    cs = np.array([f"{x['cell_type']}_{x['state']}" for x in p])
    dis = np.array([x['disease'] for x in p])
    ctype = np.array([x['cell_type'] for x in p])
    print(f"{args.run}: {len(emb)} pts, {emb.shape[1]}-d | UMAP...", flush=True)
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 13)); axes = axes.ravel()
    panel(axes[0], um, dtc, "UMAP — disease × tissue × cell type", "husl")
    panel(axes[1], um, cs, "UMAP — cell type × cell state", "tab10")
    panel(axes[2], um, dis, "UMAP — disease", "Set2")
    panel(axes[3], um, ctype, "UMAP — cell type", "Dark2")
    fig.suptitle(f"{args.run} context-specific protein embeddings (n={len(emb)})", fontsize=13)
    fig.tight_layout()
    (SEQ / "images").mkdir(exist_ok=True)
    out = SEQ / "images" / f"umap_embeddings_{args.run}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
