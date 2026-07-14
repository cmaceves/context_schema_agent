"""PCA of the UNCOUPLED scvi build's (protein x network) embeddings, colored by disease arm / cell state / tissue
(3 panels). Cell type is constant (macrophage). Shows what organizes the uncoupled space, given that a protein's
versions are NOT pulled together there.

Output: images/pca_by_context.png
Run: .venv/bin/python .../pca_by_context.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

DIRP = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi")
PAL = {
    "disease": {"crohn": "#4C72B0", "uc": "#DD8452", "ild": "#55A868", "healthy": "#999999"},
    "state":   {"inflammatory": "#e41a1c", "resident": "#377eb8", "proliferating": "#4daf4a"},
    "tissue":  {"colon": "#984ea3", "ileum": "#ff7f00", "lung": "#a65628"},
}


def main():
    z = np.load(DIRP / "embeddings.npz", allow_pickle=True)
    Z, pres, tags = z["Z"], z["present"], [str(t) for t in z["tags"]]
    fac = {"disease": [t.split("_")[0] for t in tags], "tissue": [t.split("_")[1] for t in tags],
           "state": ["_".join(t.split("_")[3:]) for t in tags]}
    pts, lab = [], {k: [] for k in PAL}
    for ti in range(len(tags)):
        idx = np.where(pres[ti])[0]; pts.append(Z[ti][idx])
        for k in PAL:
            lab[k] += [fac[k][ti]] * len(idx)
    X = np.vstack(pts); lab = {k: np.array(v) for k, v in lab.items()}
    p = PCA(n_components=2, random_state=0); xy = p.fit_transform(X); ev = p.explained_variance_ratio_
    print(f"points={len(X)}  PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key in zip(axes, ["disease", "state", "tissue"]):
        for lv, col in PAL[key].items():
            m = lab[key] == lv
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=4, c=col, alpha=0.35, edgecolors="none", rasterized=True, label=lv)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title(f"colored by {key}", fontsize=11)
        ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=l)
                           for l, c in PAL[key].items()], fontsize=8, loc="best")
    fig.suptitle("UNCOUPLED scvi build — PCA of (protein x network) embeddings (cell type = macrophage)", fontsize=12)
    fig.tight_layout()
    out = DIRP / "images"; out.mkdir(exist_ok=True)
    fig.savefig(out / "pca_by_context.png", dpi=150); plt.close(fig)
    print(f"wrote {out/'pca_by_context.png'}")


if __name__ == "__main__":
    main()
