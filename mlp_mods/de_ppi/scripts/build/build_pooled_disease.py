"""Pool MULTIPLE studies of one (disease, cell type) into one network per state, using the union+average
rule: node set = UNION of per-study expressed sets; for the DE weights, each protein's rank-shift is the
AVERAGE of its per-study values over the studies that ranked it (a protein present in only one study
keeps that study's value); the dysregulated/direction set = significant (padj<0.05) in ANY study.

Applied to: Crohn ILEUM macrophage (a37f857c + 19053a82) and Alzheimer microglia (3 studies). Tissues
are kept separate (Crohn colon is its own single-study build, untouched). Per-study DE is computed fresh
(state_split._de, ~disease) so the average is over comparable per-study estimates.

Writes pooled networks crohn_mac_<state> / alz_microglia_<state> into the expressed embedding (overwriting
the single-study versions). Run: .venv/bin/python mlp_mods/de_ppi/build_pooled_disease.py
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
sys.path.insert(0, "mlp_mods/de_ppi"); sys.path.insert(0, "mlp_mods/rank_shifts/de_scripts")
import build_literature_weighted_influence as B
import state_split

HERE = Path("mlp_mods/de_ppi"); RS = Path("mlp_mods/rank_shifts")
NET = HERE / "results/crohn_alzheimer_ild_uc_embedding_expressed/networks"
EXPR_DIR = HERE / "expressed_genes_threshold"; OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
MANIFEST = Path("mlp_mods/02_build_ppi/builds_manifest.json"); CP10K = 0.5

POOLS = {
  "crohn_mac": dict(disease="Crohn disease", lab="crohn", slug="crohns",
    studies=[(RS/"macrophage_crohn_paired/pulled_macrophages.h5ad", RS/"macrophage_crohn_states/cell_states.tsv", None),
             (RS/"macrophage_crohn_rep_paired/pulled_macrophages.h5ad", RS/"macrophage_crohn_rep_states/cell_states.tsv", None)]),
  "alz_microglia": dict(disease="Alzheimer disease", lab="alz", slug="alz",
    studies=[(RS/"microglia_alzheimers_paired/pulled_microglia.h5ad", RS/"microglia_alzheimers_states/cell_states.tsv", d)
             for d in ["203025fe-fa99-4d57-81da-458ed8f0c334","ac0c6561-7a48-4185-af6f-af799f699172","cff99df2-4904-44f7-9173-ff837f95606e"]]),
}


def expressed(a, mask):
    sub = a[mask]; X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    return set(pd.Index(sub.var_names)[np.asarray((sp.diags(1e4/tot)@X).mean(0)).ravel() >= CP10K])


def build_neutral(genes, tag, op):
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src","dst"])
    inc = genes & (set(o.src)|set(o.dst)); prot = sorted(inc); d = NET/tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id":prot,"node_type":"protein","source":"expressed","direction":"","sender_weight":1.0}).to_csv(d/"network_nodes.tsv",sep="\t",index=False)
    e=o[o.src.isin(inc)&o.dst.isin(inc)]
    pd.DataFrame({"source":e.src,"target":e.dst,"edge_origin":"OmniPath","edge_property":"","weight":1.0,"direction":""}).to_csv(d/"network_edges.tsv",sep="\t",index=False)
    print(f"  {tag:34s} {len(prot)} proteins (NEUTRAL: no study had DE)", flush=True)


def main():
    op = pd.read_csv(OMNI, sep="\t"); man = json.load(open(MANIFEST))
    for pool, cfg in POOLS.items():
        print(f"\n===== {pool} ({cfg['disease']}, {len(cfg['studies'])} studies) =====", flush=True)
        # load each study's cells + state labels
        studies = []
        for h5, stf, dsfilt in cfg["studies"]:
            a = ad.read_h5ad(h5); st = pd.read_csv(stf, sep="\t", index_col=0)
            assert len(st) == a.n_obs
            a.obs["state"] = st["state"].astype(str).values
            if dsfilt is not None:
                a = a[st["dataset_id"].astype(str).values == dsfilt].copy()
            studies.append(a)
        states = sorted(set().union(*[set(a.obs["state"]) for a in studies]))
        for state in states:
            per_expr, per_de = [], []
            for a in studies:
                m = (a.obs["state"] == state).values
                if m.sum() < 50:
                    continue
                per_expr.append(expressed(a, m))
                df, nh, nc, ns = state_split._de(a[m].copy(), cfg["disease"], cfg["lab"])
                if df is not None:
                    per_de.append(df)
            if not per_expr:
                continue
            union_expr = sorted(set().union(*per_expr))
            build = f"{pool}_pooled_{state}"; tag = f"{pool}_{state}"
            (EXPR_DIR/f"{build}.txt").write_text("\n".join(union_expr) + "\n")
            if not per_de:                                   # no study powered for DE -> neutral union
                build_neutral(set(union_expr), tag, op); continue
            # MERGE DE: union genes; avg ranks over studies with the gene; padj = min (sig in any)
            rc = f"{cfg['lab']}_rank"
            allg = sorted(set().union(*[set(d.index) for d in per_de]))
            col = lambda c: pd.concat([d[c] for d in per_de], axis=1).reindex(allg)
            merged = pd.DataFrame({"healthy_rank": col("healthy_rank").mean(1), rc: col(rc).mean(1),
                                   "log2FoldChange": col("log2FoldChange").mean(1),
                                   "baseMean": col("baseMean").mean(1)}, index=allg)
            merged["rank_shift"] = merged[rc] - merged["healthy_rank"]
            merged["padj"] = col("padj").min(1); merged.index.name = "gene"
            de_tsv = RS/f"{pool}_pooled_states/states/{state}/pseudobulk_de.tsv"; de_tsv.parent.mkdir(parents=True, exist_ok=True)
            merged.sort_values("padj").to_csv(de_tsv, sep="\t")
            man["builds"][build] = {"cell_type": pool.split("_")[-1] if pool!="crohn_mac" else "macrophage",
                "disease": cfg["disease"], "disease_synonyms": ["pooled"], "disease_slug": cfg["slug"],
                "celltype_slug": f"x_{state}", "de_table": str(de_tsv),
                "_note": f"POOLED across {len(per_de)} studies (union node set, averaged per-study rank-shift).",
                "arms": [{"name":"healthy","h5ad":str(cfg["studies"][0][0]),"disease_filter":None,"target_label":"healthy"},
                         {"name":cfg["lab"],"h5ad":str(cfg["studies"][0][0]),"disease_filter":cfg["disease"],"target_label":"disease"}]}
            json.dump(man, open(MANIFEST,"w"), indent=1)
            B.main(build, expressed_backbone=True, rank_weight_all=True,
                   net_out=str(NET/tag), expr_genes_path=str(EXPR_DIR/f"{build}.txt"))
            print(f"  {tag:34s} pooled {len(per_de)} studies, union={len(union_expr)} proteins, {len(allg)} ranked", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
