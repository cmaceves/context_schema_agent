"""Build the COLON Crohn macrophage state networks into the embedding — the tissue-matched (colon,
like Smillie UC) counterpart to the ileum Crohn build. Identical processing to build_uc_smillie.py:

  per Leiden state with a DE table: expressed node set (mean CP10k>=0.5) + rank-shift weights ->
     networks/crohn_colon_mac_<state>/   (manifest build macrophage_crohn_colon_<state>)
  disease-emergent states (cells exist, per-state DE skipped): expression-only network, neutral weights
  healthy_crohn_colon_macrophage: normal-arm colon macrophages, CP10k expressed, OmniPath, neutral

Run (after macrophage_crohn_colon.py + macrophage_crohn_colon_states.py):
  .venv/bin/python mlp_mods/de_ppi/build_crohn_colon.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
sys.path.insert(0, "mlp_mods/de_ppi")
import build_literature_weighted_influence as B

HERE = Path("mlp_mods/de_ppi"); RS = Path("mlp_mods/rank_shifts")
NET = HERE / "results/crohn_alzheimer_ild_uc_embedding_expressed/networks"
EXPR_DIR = HERE / "expressed_genes_threshold"
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
MANIFEST = Path("mlp_mods/02_build_ppi/builds_manifest.json")
H5 = str(RS / "macrophage_crohn_colon_paired/pulled_macrophages.h5ad")
STATES = RS / "macrophage_crohn_colon_states"
CP10K_CUTOFF = 0.5


def expressed(a, mask):
    sub = a[mask]; X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    meancp = np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()
    return sorted(pd.Index(sub.var_names)[meancp >= CP10K_CUTOFF])


def build_neutral(a, tag, mask):                        # expressed node set over `mask` cells, neutral weights
    m = a[mask]
    genes = set(expressed(a, mask))
    op = pd.read_csv(OMNI, sep="\t"); op = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    inc = genes & (set(op.src) | set(op.dst)); prot = sorted(inc); d = NET / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed", "direction": "", "sender_weight": 1.0}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    e = op[op.src.isin(inc) & op.dst.isin(inc)]
    pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath", "edge_property": "", "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"{tag}: {m.n_obs} cells, {len(prot)} proteins, {len(e)} edges (neutral)", flush=True)


def main():
    a = ad.read_h5ad(H5)
    st = pd.read_csv(STATES / "cell_states.tsv", sep="\t", index_col=0)
    assert len(st) == a.n_obs, f"states {len(st)} != cells {a.n_obs}"
    a.obs["state"] = st["state"].values
    sc = pd.read_csv(STATES / "state_counts.tsv", sep="\t")
    states = sc.loc[sc["DE_genes_padj_lt_0.05"].notna() & (sc["DE_genes_padj_lt_0.05"] > 0), "state"].tolist()
    emergent = [s for s in sc["state"] if s not in states and (a.obs.state == s).sum() > 0]
    print(f"Crohn-colon states with DE: {states} | disease-emergent (expression-only): {emergent}", flush=True)

    man = json.load(open(MANIFEST))
    for state in states:
        build = f"macrophage_crohn_colon_{state}"
        de_tsv = STATES / "states" / state / "pseudobulk_de.tsv"
        genes = expressed(a, (a.obs.state == state).values)
        (EXPR_DIR / f"{build}.txt").write_text("\n".join(genes) + "\n")
        man["builds"][build] = {
            "cell_type": "macrophage", "disease": "Crohn disease",
            "disease_synonyms": ["inflammatory bowel disease", "colitis"],
            "disease_slug": "crohns", "celltype_slug": f"macrophages_{state}",
            "de_table": str(de_tsv),
            "_note": "Crohn COLON macrophage Leiden STATE sub-build (dataset 518d9049, tissue-matched to "
                     "Smillie UC). expressed-backbone; --no-lit; no OpenTargets.",
            "arms": [{"name": "healthy", "h5ad": H5, "disease_filter": None, "target_label": "healthy_macrophage"},
                     {"name": "crohn", "h5ad": H5, "disease_filter": "Crohn disease", "target_label": "crohn_macrophage"}]}
        print(f"  {build}: expressed={len(genes)}  de_table={de_tsv}", flush=True)
    json.dump(man, open(MANIFEST, "w"), indent=1)

    for state in states:
        build = f"macrophage_crohn_colon_{state}"
        B.main(build, expressed_backbone=True, rank_weight_all=True,
               net_out=str(NET / f"crohn_colon_mac_{state}"), expr_genes_path=str(EXPR_DIR / f"{build}.txt"))
    for state in emergent:
        build_neutral(a, f"crohn_colon_mac_{state}", ((a.obs.state == state) & (a.obs.disease.astype(str) == "Crohn disease")).values)
    build_neutral(a, "healthy_crohn_colon_macrophage", (a.obs.disease.astype(str) == "normal").values)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
