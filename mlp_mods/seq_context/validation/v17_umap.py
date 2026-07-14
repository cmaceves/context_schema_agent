"""UMAP of v17 embeddings (protein x context), colored by disease and by state. whitegrid."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap

SEQ = Path("mlp_mods/seq_context"); SEED, NPLOT = 0, 25000
d = np.load(SEQ / "results/link_v17/embeddings.npz", allow_pickle=True)
Z = d["emb"]; disease = d["disease"]; state = d["state"]
rng = np.random.default_rng(SEED); sel = rng.choice(len(Z), min(NPLOT, len(Z)), replace=False)
emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(Z[sel])
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, lab, title in [(axes[0], disease[sel], "disease"), (axes[1], state[sel], "state")]:
    for i, v in enumerate(sorted(set(lab))):
        m = lab == v
        ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, label=f"{v} ({m.sum()})", color=plt.cm.tab10(i % 10))
    ax.set(title=f"v17 embedding (protein x context) — colored by {title}", xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend(markerscale=3, fontsize=8, loc="best")
fig.tight_layout()
out = SEQ / "images/v17_umap.png"; fig.savefig(out, dpi=130)
print(f"wrote {out}", flush=True)
