"""Donor-split NEGATIVE controls (noise floor) for the heatmaps, added to
crohn_alzheimer_ild_uc_embedding_expressed:

  crohn_mac_inflammatory_split{A,B}: inflammatory-state donors split in two halves; each half's
     Crohn-vs-healthy DE -> expressed-backbone (REUSED inflammatory expressed set, so node set/topology
     are identical) + rank-weight-all weights. shift(A,B) = donor-sampling floor of the DISEASE weights.
  healthy_macrophage_split{A,B}: normal-arm macrophage donors split in two halves; each half's own
     CP10k expressed set + neutral weights. shift(A,B) = donor-sampling floor of the healthy/node-set.

Run: .venv/bin/python mlp_mods/de_ppi/build_donor_split_controls.py
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
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
EXPR_INF = HERE / "expressed_genes_threshold/macrophage_crohn_inflammatory.txt"
MANIFEST = Path("mlp_mods/02_build_ppi/builds_manifest.json")
H5 = "mlp_mods/rank_shifts/macrophage_crohn_paired/pulled_macrophages.h5ad"

a = ad.read_h5ad(RS / "macrophage_crohn_paired/pulled_macrophages.h5ad")
st = pd.read_csv(RS / "macrophage_crohn_states/cell_states.tsv", sep="\t", index_col=0)
a.obs["state"] = st["state"].values
inf = a[a.obs.state == "inflammatory"].copy()


def halves(adata, arm, seed):
    vc = adata.obs[adata.obs.disease == arm].donor_id.astype(str).value_counts()
    donors = sorted(vc[vc >= 20].index)
    perm = np.random.default_rng(seed).permutation(donors); h = len(perm) // 2
    return list(perm[:h]), list(perm[h:])


def pseudobulk(adata, donors):
    m = adata[adata.obs.donor_id.astype(str).isin(donors)]
    X = m.X.tocsr() if sp.issparse(m.X) else sp.csr_matrix(m.X); don = m.obs.donor_id.astype(str).values
    return pd.DataFrame(np.vstack([np.asarray(X[don == d].sum(0)).ravel() for d in donors]),
                        index=donors, columns=m.var_names).round().astype(int)


def de_half(cd, hd, out_tsv):
    h, c = pseudobulk(inf, hd), pseudobulk(inf, cd)
    g = h.columns.intersection(c.columns); h, c = h[g], c[g]
    det = ((h > 0).mean(0) >= 0.5) & ((c > 0).mean(0) >= 0.5); g = g[det.values]; h, c = h[g], c[g]
    counts = pd.concat([h, c]); meta = pd.DataFrame({"disease": ["healthy"] * len(h) + ["crohn"] * len(c)}, index=counts.index)
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts, metadata=meta, design="~disease", quiet=True); dds.deseq2()
    res = DeseqStats(dds, contrast=["disease", "crohn", "healthy"], quiet=True); res.summary()
    df = res.results_df.copy(); df.index.name = "gene"
    rk = lambda mat: (np.log1p(mat.div(mat.sum(1), axis=0) * 1e6)).mean(0).rank(ascending=False)
    df["healthy_rank"] = rk(h).reindex(df.index).values; df["crohn_rank"] = rk(c).reindex(df.index).values
    df["rank_shift"] = df["crohn_rank"] - df["healthy_rank"]
    out_tsv.parent.mkdir(parents=True, exist_ok=True); df.sort_values("padj").to_csv(out_tsv, sep="\t")
    return int((df.padj < 0.05).sum())


def build_neutral(donors, tag):
    m = a[(a.obs.disease == "normal") & (a.obs.donor_id.astype(str).isin(donors))]
    X = m.X.tocsr() if sp.issparse(m.X) else sp.csr_matrix(m.X); tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    mcp = np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel(); genes = set(pd.Index(m.var_names)[mcp >= 0.5])
    op = pd.read_csv(OMNI, sep="\t"); op = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    inc = genes & (set(op.src) | set(op.dst)); prot = sorted(inc); d = NET / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed", "direction": "", "sender_weight": 1.0}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    e = op[op.src.isin(inc) & op.dst.isin(inc)]
    pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath", "edge_property": "", "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"{tag}: {len(prot)} proteins, {len(e)} edges (neutral)", flush=True)


def main():
    cA, cB = halves(inf, "Crohn disease", 0); hA, hB = halves(inf, "normal", 0)
    print(f"Crohn-inf donors A/B = {len(cA)}/{len(cB)} | healthy-inf donors A/B = {len(hA)}/{len(hB)}", flush=True)
    man = json.load(open(MANIFEST))
    for half, (cd, hd) in {"A": (cA, hA), "B": (cB, hB)}.items():
        de_tsv = RS / f"macrophage_crohn_inflammatory_split{half}_paired/pseudobulk_de.tsv"
        ng = de_half(cd, hd, de_tsv)
        b = f"macrophage_crohn_inflammatory_split{half}"
        man["builds"][b] = {"cell_type": "macrophage", "celltype_slug": "macrophages",
            "disease": "Crohn disease", "disease_slug": "crohns", "de_table": str(de_tsv),
            "arms": [{"name": "healthy", "disease_filter": None, "h5ad": H5, "target_label": "healthy_macrophage"},
                     {"name": "crohn", "disease_filter": "Crohn disease", "h5ad": H5, "target_label": "crohn_macrophage"}],
            "opentargets_key": ["EFO_0003767", "macrophage"]}
        print(f"split{half}: {ng} DE genes", flush=True)
    json.dump(man, open(MANIFEST, "w"), indent=1)
    for half in "AB":
        B.main(f"macrophage_crohn_inflammatory_split{half}", expressed_backbone=True, rank_weight_all=True,
               net_out=str(NET / f"crohn_mac_inflammatory_split{half}"), expr_genes_path=str(EXPR_INF))
    nd = a.obs[a.obs.disease == "normal"].donor_id.astype(str).value_counts(); nd = sorted(nd[nd >= 20].index)
    perm = np.random.default_rng(1).permutation(nd); nh = len(perm) // 2
    build_neutral(list(perm[:nh]), "healthy_macrophage_splitA")
    build_neutral(list(perm[nh:]), "healthy_macrophage_splitB")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
