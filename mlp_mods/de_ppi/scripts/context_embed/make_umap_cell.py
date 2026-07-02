"""Per-network shift-cluster UMAP for every INFLAMMATORY MACROPHAGE disease context.

For each disease network <arm>_<tissue>_macrophage_inflammatory (arm in crohn/uc/…) with a paired healthy
network, cluster proteins by their disease-vs-healthy embedding shift r = Z_disease - Z_healthy (64-d), lay out
with UMAP, color by Leiden module, and label that disease's high-confidence OT targets (>ot_thr). One PNG per
context in results/<main>/images/umap_cell/<disease_net>_umap.png.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/make_umap_cell.py \
        --main-name crohn_alzheimer_ild_uc_context_contrastive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
from plot_style import apply_style

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
           "alz": "alzheimer_target_association_MONDO_0004975.tsv", "ild": "ild_target_association_EFO_0004244.tsv"}


def one(main_name, dtag, Z, pres, nid, idx, out, resolution, ot_thr, seed):
    arm = dtag.split("_")[0]
    ht = "healthy_" + dtag.split("_", 1)[1]
    if ht not in idx:
        print(f"  skip {dtag}: no paired healthy"); return
    di, hi = idx[dtag], idx[ht]
    m = pres[di] & pres[hi]
    r = (Z[di] - Z[hi])[m]; genes = nid[m]
    a = ad.AnnData(r.astype(np.float32)); a.obs_names = genes
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X", random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    xy = a.obsm["X_umap"]; clust = a.obs["leiden"].to_numpy()
    sc_ot = dict(zip(*[pd.read_csv(OT_DIR / OT_FILE[arm], sep="\t")[c] for c in ["gene_symbol", "score_indirect"]])) \
        if arm in OT_FILE else {}
    ot = pd.Series(genes).map(sc_ot).fillna(0).to_numpy(); tgt = ot > ot_thr

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    for k in sorted(set(clust), key=int):
        s = clust == k
        ax.scatter(xy[s, 0], xy[s, 1], s=6, alpha=0.45, color=cmap(int(k) % 20), edgecolors="none",
                   rasterized=True, label=f"{k} (n={s.sum()})")
    for j in np.where(tgt)[0]:
        ax.scatter(xy[j, 0], xy[j, 1], s=42, facecolors="none", edgecolors="black", linewidths=0.9)
        ax.annotate(genes[j], (xy[j, 0], xy[j, 1]), fontsize=6.5, fontweight="bold",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"{dtag}\nproteins clustered by disease-vs-healthy shift; circled = OT {arm} target (>{ot_thr}); "
                 f"n={m.sum()}", fontsize=9)
    ax.legend(title="module", fontsize=6, ncol=2, loc="best", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out / f"{dtag}_umap.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out/f'{dtag}_umap.png'}  (n={int(m.sum())}, modules={a.obs.leiden.nunique()}, OT>{ot_thr}={int(tgt.sum())})")


def main(main_name, resolution, ot_thr, seed) -> int:
    sc.settings.verbosity = 0
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    nid = np.asarray(e["node_id"])[pi]
    tags = list(e["tags"]); idx = {t: i for i, t in enumerate(tags)}
    out = res / "images" / "umap_cell"; out.mkdir(parents=True, exist_ok=True)
    infl_mac = [t for t in tags if t.endswith("_macrophage_inflammatory") and t.split("_")[0] != "healthy"]
    print(f"inflammatory-macrophage disease networks: {infl_mac}")
    for t in infl_mac:
        one(main_name, t, Z, pres, nid, idx, out, resolution, ot_thr, seed)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_context_contrastive")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--ot-thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.resolution, a.ot_thr, a.seed))
