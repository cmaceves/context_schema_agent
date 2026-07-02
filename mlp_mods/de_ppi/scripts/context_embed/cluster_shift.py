"""Cluster proteins by their DISEASE-SHIFT vectors within colon macrophage, and visualize.

Per protein present in the crohn/uc/healthy colon-macrophage <state> networks:
    r_crohn = Z_crohn - Z_healthy ;  r_uc = Z_uc - Z_healthy ;  X = [r_crohn , r_uc]  (128-d)
i.e. how the protein moves in each disease relative to healthy (baseline stripped, so this is NOT the
identity/degree-dominated raw position). Leiden-cluster the proteins into modules (candidate disease axes),
UMAP for layout, color by module, and LABEL the high-confidence OpenTargets IBD proteins (Crohn or UC > --ot-thr).

Outputs (results/<main>/images/, and disease_axis/):
  shift_clusters_umap.png   UMAP colored by Leiden module; OT>thr IBD proteins labeled
  shift_clusters.tsv        protein, module, umap1/2, ot_crohn, ot_uc, ibd_target

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/cluster_shift.py \
        --main-name crohn_alzheimer_ild_uc_context_contrastive --state inflammatory
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


def main(main_name, tissue, ct, state, resolution, ot_thr, seed) -> int:
    sc.settings.verbosity = 0
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    nid = np.asarray(e["node_id"])[pi]
    idx = {t: i for i, t in enumerate(list(e["tags"]))}
    tag = lambda a: f"{a}_{tissue}_{ct}_{state}"
    for a in ("crohn", "uc", "healthy"):
        if tag(a) not in idx:
            raise SystemExit(f"missing {tag(a)}")
    ic, iu, ih = idx[tag("crohn")], idx[tag("uc")], idx[tag("healthy")]
    m = pres[ic] & pres[iu] & pres[ih]
    X = np.hstack([Z[ic][m] - Z[ih][m], Z[iu][m] - Z[ih][m]]).astype(np.float32)   # [r_crohn, r_uc]
    genes = nid[m]

    a = ad.AnnData(X)
    a.obs_names = genes
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X", random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    umap = a.obsm["X_umap"]; clust = a.obs["leiden"].to_numpy()

    ots = {d: dict(zip(*[pd.read_csv(OT_DIR / f, sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))
           for d, f in [("crohn", "crohn_target_association_EFO_0000384.tsv"),
                        ("uc", "uc_target_association_EFO_0000729.tsv")]}
    oc = pd.Series(genes).map(ots["crohn"]).fillna(0).to_numpy()
    ou = pd.Series(genes).map(ots["uc"]).fillna(0).to_numpy()
    ibd = (oc > ot_thr) | (ou > ot_thr)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ncl = sorted(set(clust), key=int)
    cmap = plt.get_cmap("tab20")
    for k in ncl:
        s = clust == k
        ax.scatter(umap[s, 0], umap[s, 1], s=6, alpha=0.4, color=cmap(int(k) % 20),
                   edgecolors="none", rasterized=True, label=f"{k} (n={s.sum()})")
    # label high-confidence IBD OT proteins
    for j in np.where(ibd)[0]:
        ax.scatter(umap[j, 0], umap[j, 1], s=42, facecolors="none", edgecolors="black", linewidths=0.9)
        ax.annotate(genes[j], (umap[j, 0], umap[j, 1]), fontsize=6.5, fontweight="bold",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"Protein disease-shift [r_crohn, r_uc] clusters — {tissue} {ct} {state} — {main_name.split('_')[-1]}\n"
                 f"color = Leiden module; circled+labeled = OT IBD target (>{ot_thr})", fontsize=9.5)
    ax.legend(title="module", fontsize=6.5, ncol=2, loc="best", markerscale=1.5)
    fig.tight_layout()
    out = res / "images"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "shift_clusters_umap.png", dpi=150)

    df = pd.DataFrame({"protein": genes, "module": clust, "umap1": umap[:, 0].round(3),
                       "umap2": umap[:, 1].round(3), "ot_crohn": oc.round(3), "ot_uc": ou.round(3),
                       "ibd_target": ibd.astype(int)})
    (res / "disease_axis").mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "disease_axis" / "shift_clusters.tsv", sep="\t", index=False)
    print(f"proteins={len(genes)}  modules={len(ncl)}  IBD OT>{ot_thr} labeled={int(ibd.sum())}")
    print("IBD targets per module:", df[df.ibd_target == 1].module.value_counts().to_dict())
    print(f"wrote {out/'shift_clusters_umap.png'} and {res/'disease_axis'/'shift_clusters.tsv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_context_contrastive")
    ap.add_argument("--tissue", default="colon"); ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--state", default="inflammatory")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--ot-thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.tissue, a.celltype, a.state, a.resolution, a.ot_thr, a.seed))
