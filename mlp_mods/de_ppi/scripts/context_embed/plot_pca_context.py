"""PCA of per-network mean embeddings for a build: does the space organize by cell type, and do disease arms
separate within a cell type? Each point = one network (its mean protein embedding); color = cell type,
marker = disease arm. Uses the shared plot_style.

Output: results/<main>/images/pca_networks.png

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/plot_pca_context.py --main-name <build>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
from plot_style import apply_style, ARM_COLOR, TOL

MARK = {"healthy": "o", "crohn": "s", "uc": "^", "alz": "D", "ild": "P"}


def project(X, embed):
    """2D projection; returns (xy, (xlabel, ylabel))."""
    if embed == "umap":
        import umap
        xy = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
        return xy, ("UMAP1", "UMAP2")
    p = PCA(n_components=2, random_state=0); xy = p.fit_transform(X); e = p.explained_variance_ratio_
    return xy, (f"PC1 ({e[0]*100:.1f}% var)", f"PC2 ({e[1]*100:.1f}% var)")


def main(main_name, level, embed) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    tags = [str(t) for t in e["tags"]]
    arm = [t.split("_")[0] for t in tags]; cell = [t.split("_")[2] for t in tags]
    state = ["_".join(t.split("_")[3:]) for t in tags]
    cells = sorted(set(cell))
    cpal = dict(zip(cells, [TOL[k] for k in ["blue", "red", "green", "purple", "cyan", "yellow"][:len(cells)]]))
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 7))

    if level == "network":
        M = np.stack([Z[i][pres[i]].mean(0) for i in range(len(tags))])
        xy, axlab = project(M, embed)
        for i in range(len(tags)):
            ax.scatter(xy[i, 0], xy[i, 1], c=[cpal[cell[i]]], marker=MARK.get(arm[i], "o"),
                       s=90, edgecolors="0.3", linewidths=0.6)
            ax.annotate(state[i][:5], (xy[i, 0], xy[i, 1]), fontsize=5.5, color="0.25",
                        xytext=(3, 3), textcoords="offset points")
        extra = "text = cell state"; fname = f"{embed}_networks.png"
    else:  # protein: every (protein, network) embedding as a point
        pts, ci, ai = [], [], []
        for i in range(len(tags)):
            m = pres[i]; pts.append(Z[i][m]); ci += [cell[i]] * int(m.sum()); ai += [arm[i]] * int(m.sum())
        X = np.vstack(pts); ci = np.array(ci); ai = np.array(ai)
        xy, axlab = project(X, embed)
        for a in [x for x in MARK if x in set(ai)]:              # scatter per (arm) so marker differs; color by cell
            for c in cells:
                s = (ai == a) & (ci == c)
                if s.any():
                    ax.scatter(xy[s, 0], xy[s, 1], c=[cpal[c]], marker=MARK[a], s=4, alpha=0.15,
                               edgecolors="none", rasterized=True)
        extra = f"{len(X):,} (protein × network) points"; fname = f"{embed}_proteins.png"

    from matplotlib.lines import Line2D
    cl = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cpal[c], markersize=9, label=c) for c in cells]
    ml = [Line2D([0], [0], marker=MARK[a], color="0.3", markerfacecolor="0.7", markersize=9, label=a)
          for a in MARK if a in set(arm)]
    l1 = ax.legend(handles=cl, title="cell type", loc="upper left", fontsize=8); ax.add_artist(l1)
    ax.legend(handles=ml, title="arm", loc="lower right", fontsize=8)
    ax.set_xlabel(axlab[0]); ax.set_ylabel(axlab[1])
    ax.set_title(f"{'Per-network mean' if level=='network' else 'All-protein'} embedding {embed.upper()} — "
                 f"{main_name.split('_')[-1]}\n(color = cell type, marker = disease arm, {extra})", fontsize=10)
    fig.tight_layout()
    out = res / "images"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / fname, dpi=150)
    print(f"wrote {out/fname}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", required=True)
    ap.add_argument("--level", choices=("network", "protein"), default="network")
    ap.add_argument("--embed", choices=("pca", "umap"), default="pca")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.level, a.embed))
