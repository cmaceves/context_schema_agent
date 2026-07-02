"""Fresh build: PINNACLE-backbone + expressed node set, ComBat expression feature, CELL-TYPE only (no state).

Per (arm, tissue, cell type) -- pooling all cell states AND studies:
  node set  = PINNACLE cell-type backbone nodes  UNION  that context's expressed proteins (mean CP10k >= 0.5),
              restricted to OmniPath-incident (so every node has >=1 edge). PINNACLE readmits proteins the
              expression detection floor drops (e.g. NOD2, TNF in macrophage).
  edges     = OmniPath DIRECTED edges over the union node set (weights NEUTRAL = 1.0).
  feature   = ComBat-corrected log1p(mean CP10k). Correction is per (tissue,celltype) group: each (arm, study)
              pseudobulk is a sample, batch = study (dataset_id), arm PRESERVED as covariate, location-only
              (reuses combat_ls). A PINNACLE node that is measured but lowly expressed keeps its real (low)
              corrected value; unmeasured nodes get 0. Groups with <2 studies are left uncorrected.
  node source column = 'expressed' or 'pinnacle' (which set readmitted it).

Writes results/<out>/networks/<arm>_<tissue>_<celltype>/  then train the joint encoder:
  .venv/bin/python mlp_mods/de_ppi/scripts/embed/joint_embed.py --out-name <out> --expr-feat

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/build_pinnacle_ct.py \
        --out crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "mlp_mods/de_ppi/scripts/build/controls")
from apply_combat_expression import combat_ls

RS = Path("mlp_mods/rank_shifts")
RES = Path("mlp_mods/de_ppi/results")
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
PINN_DIR = Path("db/pinnacle_official/ppi_edgelists")
CP10K_CUTOFF = 0.5          # detection floor: a gene must clear this (mean CP10k) to get its REAL expression feature
UNEXPRESSED_FILL = 0.0      # PINNACLE nodes below the floor stay in the graph (topology) but get this feature value
MIN_CELLS = 50

# (tissue, celltype) -> source atlases (base names of <base>_paired/pulled_*.h5ad + <base>_states/cell_states.tsv)
GROUPS = {
    ("brain", "fibroblast"): ["fibroblast_alzheimers"],
    ("brain", "microglia"):  ["microglia_alzheimers"],
    ("colon", "macrophage"): ["macrophage_crohn_colon", "macrophage_uc_smillie",
                              "macrophage_garrido_crohn", "macrophage_garrido_uc"],
    ("gut", "fibroblast"):   ["fibroblast_crohn"],
    ("ileum", "macrophage"): ["macrophage_crohn", "macrophage_crohn_rep"],
    ("ileum", "stem"):       ["stem_crohn"],
    ("lung", "macrophage"):  ["macrophage_ild"],
}
LABEL2ARM = {"normal": "healthy", "Crohn disease": "crohn", "ulcerative colitis": "uc",
             "Alzheimer disease": "alz", "interstitial lung disease": "ild"}
PINN = {"macrophage": "macrophage", "fibroblast": "fibroblast",
        "microglia": "microglial_cell", "stem": "intestinal_crypt_stem_cell"}

_CACHE: dict = {}


def load_atlas(base: str):
    if base in _CACHE:
        return _CACHE[base]
    h5 = next((RS / f"{base}_paired").glob("pulled_*.h5ad"))
    a = ad.read_h5ad(h5)
    st = pd.read_csv(RS / f"{base}_states/cell_states.tsv", sep="\t", index_col=0)
    assert len(st) == a.n_obs
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1.0
    cp10k = (sp.diags(1e4 / tot) @ X).tocsr()
    _CACHE[base] = (st.reset_index(drop=True), cp10k, a.var_names)
    return _CACHE[base]


def pinn_nodes(celltype: str) -> set[str]:
    f = PINN_DIR / f"{PINN[celltype]}.txt"
    g = set()
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            g.update(p[:2])
    return g


def main(out: str) -> int:
    omni = pd.read_csv(OMNI, sep="\t", usecols=["src", "dst"])
    omni = omni[omni.src != omni.dst].drop_duplicates()
    incident = set(omni.src) | set(omni.dst)
    net_root = RES / out / "networks"; net_root.mkdir(parents=True, exist_ok=True)

    for (tissue, ct), atlases in GROUPS.items():
        # ---- gather (arm, study) pseudobulk samples: log1p(mean CP10k), deduped by (study, arm) across atlases
        samples, seen = [], set()
        for base in atlases:
            meta, cp10k, var = load_atlas(base)
            for lbl, arm in LABEL2ARM.items():
                sel_lbl = meta["disease"].astype(str) == lbl
                if not sel_lbl.any():
                    continue
                for study in meta.loc[sel_lbl, "dataset_id"].astype(str).unique():
                    if (study, arm) in seen:
                        continue
                    mask = sel_lbl & (meta["dataset_id"].astype(str) == study)
                    if mask.sum() < MIN_CELLS:
                        continue
                    mean_cp = np.asarray(cp10k[mask.to_numpy()].mean(0)).ravel()
                    samples.append((arm, study, pd.Series(np.log1p(mean_cp), index=var)))
                    seen.add((study, arm))
        if not samples:
            continue
        arms = sorted({s[0] for s in samples})
        Mexpr = pd.DataFrame({i: s[2] for i, s in enumerate(samples)}).T           # samples x genes (log1p CP10k)
        smeta = pd.DataFrame([(s[0], s[1]) for s in samples], columns=["arm", "study"])

        # expressed set per arm (mean over that arm's samples >= cutoff, on CP10k scale)
        expr_arm = {}
        for arm in arms:
            rows = smeta.index[smeta.arm == arm]
            mean_cp = np.expm1(Mexpr.loc[rows]).mean(0)                            # back to CP10k for the floor
            expr_arm[arm] = set(mean_cp.index[mean_cp >= CP10K_CUTOFF])

        pn = pinn_nodes(ct)
        union_genes = (pn | set().union(*expr_arm.values())) & incident
        ug = [g for g in union_genes if g in Mexpr.columns]                        # measured union genes for ComBat

        # ---- ComBat-correct the union-gene sample matrix (batch=study, arm preserved); skip if <2 studies
        X = Mexpr[ug].to_numpy()
        if smeta.study.nunique() > 1:
            X = combat_ls(X, smeta.study.to_numpy(), smeta[["arm"]], scale=False)
        Xc = pd.DataFrame(X, columns=ug)

        for arm in arms:
            rows = smeta.index[smeta.arm == arm].tolist()
            feat = Xc.loc[rows].mean(0).fillna(0.0)                                # pooled corrected expr (NaN->0)
            nodes = sorted((pn | expr_arm[arm]) & incident)
            e = omni[omni.src.isin(nodes) & omni.dst.isin(nodes)]
            tag = f"{arm}_{tissue}_{ct}"
            d = net_root / tag; d.mkdir(parents=True, exist_ok=True)
            src_col = ["expressed" if g in expr_arm[arm] else "pinnacle" for g in nodes]
            # feature = real ComBat expr only for genes clearing the detection floor; sub-floor (PINNACLE-only)
            # nodes stay in the graph but get UNEXPRESSED_FILL, so their noisy low-count expression can't
            # destabilize the embedding (e.g. inflate the disease donor-split floor).
            expr_vals = [float(feat.get(g, 0.0)) if s == "expressed" else UNEXPRESSED_FILL
                         for g, s in zip(nodes, src_col)]
            pd.DataFrame({"node_id": nodes, "node_type": "protein", "source": src_col,
                          "direction": "", "sender_weight": 1.0, "expression": expr_vals}
                         ).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
            pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath",
                          "edge_property": "", "weight": 1.0, "direction": ""}
                         ).to_csv(d / "network_edges.tsv", sep="\t", index=False)
            n_pinn_only = sum(s == "pinnacle" for s in src_col)
            print(f"  {tag:26s} nodes={len(nodes):5d} (pinnacle-only readmitted {n_pinn_only:4d})  "
                  f"edges={len(e):6d}  studies={smeta.loc[rows,'study'].nunique()}", flush=True)

    print(f"\nwrote {net_root}\nnext: .venv/bin/python mlp_mods/de_ppi/scripts/embed/joint_embed.py --out-name {out} --expr-feat")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct")
    a = ap.parse_args()
    raise SystemExit(main(a.out))
