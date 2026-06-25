"""Two INDEPENDENT single-study Alzheimer-microglia replicates (study3 left out), built exactly like the
Crohn replicate (macrophage_crohn_rep): per state, single-study pseudobulk DE where donors allow
(expressed-backbone + rank-shift weights) else neutral expression-only. This tests whether a single
study of a DIFFERENT disease also carries a large batch floor, or if that was Crohn-specific.

  s1 = 203025fe ,  s2 = ac0c6561  (study3 cff99df2 = LEFT OUT)
Pooled Leiden state assignments (microglia_alzheimers_states/cell_states.tsv) are reused so the state
DEFINITION is held fixed across the two replicates; only the study's cells differ. Tags:
alz_microglia_<state>_s1 / _s2  (parse to the same alz/brain/microglia/<state> context).

Compare alz_microglia_<state>_s1 <-> _s2 = the AD single-study batch floor, vs Crohn study-A<->B.

Run: .venv/bin/python mlp_mods/de_ppi/build_alz_microglia_replicates.py
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
sys.path.insert(0, "mlp_mods/rank_shifts/de_scripts")
import build_literature_weighted_influence as B
import state_split

HERE = Path("mlp_mods/de_ppi"); RS = Path("mlp_mods/rank_shifts")
NET = HERE / "results/crohn_alzheimer_ild_uc_embedding_expressed/networks"
EXPR_DIR = HERE / "expressed_genes_threshold"
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
MANIFEST = Path("mlp_mods/02_build_ppi/builds_manifest.json")
H5 = str(RS / "microglia_alzheimers_paired/pulled_microglia.h5ad")
STATES = RS / "microglia_alzheimers_states/cell_states.tsv"
STUDIES = {"s1": "203025fe-fa99-4d57-81da-458ed8f0c334",
           "s2": "ac0c6561-7a48-4185-af6f-af799f699172"}     # study3 cff99df2 left out
CP10K_CUTOFF = 0.5


def expressed(a, mask):
    sub = a[mask]; X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    meancp = np.asarray((sp.diags(1e4 / tot) @ X).mean(0)).ravel()
    return sorted(pd.Index(sub.var_names)[meancp >= CP10K_CUTOFF])


def build_neutral(a, tag, mask, op):
    genes = set(expressed(a, mask))
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    inc = genes & (set(o.src) | set(o.dst)); prot = sorted(inc); d = NET / tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "expressed", "direction": "", "sender_weight": 1.0}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
    e = o[o.src.isin(inc) & o.dst.isin(inc)]
    pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath", "edge_property": "", "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)
    print(f"  {tag:36s} {int(mask.sum()):6d} cells -> {len(prot)} proteins (NEUTRAL: DE too few donors)", flush=True)


def main():
    a = ad.read_h5ad(H5)
    st = pd.read_csv(STATES, sep="\t", index_col=0)
    assert len(st) == a.n_obs
    a.obs["state"] = st["state"].astype(str).values
    a.obs["dataset_id"] = st["dataset_id"].astype(str).values
    op = pd.read_csv(OMNI, sep="\t")
    man = json.load(open(MANIFEST))
    states = sorted(set(a.obs["state"]))
    for stag, ds in STUDIES.items():
        print(f"\n=== {stag} = {ds[:8]} ===", flush=True)
        for s in states:
            mask = ((a.obs.dataset_id == ds) & (a.obs.state == s)).values
            if mask.sum() < 50:
                print(f"  skip {s} ({int(mask.sum())} cells)", flush=True); continue
            sub = a[mask].copy()
            df, ndh, ndc, nsig = state_split._de(sub, "Alzheimer disease", "alz")
            genes = expressed(a, mask)
            build = f"microglia_alz_{stag}_{s}"
            (EXPR_DIR / f"{build}.txt").write_text("\n".join(genes) + "\n")
            tag = f"alz_microglia_{s}_{stag}"
            if df is None:
                build_neutral(a, tag, mask, op); continue
            de_tsv = RS / f"microglia_alz_{stag}_states/states/{s}/pseudobulk_de.tsv"
            de_tsv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(de_tsv, sep="\t")
            man["builds"][build] = {
                "cell_type": "microglia", "disease": "Alzheimer disease", "disease_synonyms": ["AD", "dementia"],
                "disease_slug": "alz", "celltype_slug": f"microglia_{s}", "de_table": str(de_tsv),
                "_note": f"AD microglia single-study replicate {stag} ({ds[:8]}); batch test vs the other study.",
                "arms": [{"name": "healthy", "h5ad": H5, "disease_filter": None, "target_label": "healthy_microglia"},
                         {"name": "alz", "h5ad": H5, "disease_filter": "Alzheimer disease", "target_label": "alz_microglia"}]}
            json.dump(man, open(MANIFEST, "w"), indent=1)
            B.main(build, expressed_backbone=True, rank_weight_all=True,
                   net_out=str(NET / tag), expr_genes_path=str(EXPR_DIR / f"{build}.txt"))
            print(f"  {tag:36s} {int(mask.sum()):6d} cells -> DE-weighted (healthy={ndh},alz={ndc},sig={nsig})", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
