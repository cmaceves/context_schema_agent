"""Control-experiment networks for results/crohn_alzheimer_ild_uc/ (built alongside symlinks to the main
pooled networks). ALL neutral (expression-only) so the VARIATION comparisons are apples-to-apples:

  Crohn inflammatory macrophage:
    - donor-split within ONE dataset (a37f857c): crohn_mac_inflammatory_split{A,B}   (donor-resampling floor)
    - per-study singles: crohn_mac_<state>_s1 (a37f857c), _s2 (19053a82)              (between-study variation)
  Alzheimer microglia (repeat the same controls):
    - per-study singles: alz_microglia_<state>_s1/s2/s3 (3 studies)
    - pair-vs-withheld: alz_microglia_<state>_loo{i} = pool of the 2 studies leaving out study i
      (compare loo{i} to the withheld single s{i})
    - donor-split within ONE dataset (ac0c6561), dam state: alz_microglia_dam_split{A,B}

Run: .venv/bin/python mlp_mods/de_ppi/build_control_experiments.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

RS = Path("mlp_mods/rank_shifts")
CTRL = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc/networks")
OMNI = Path("mlp_mods/omnipath_directed/omnipath_global_directed.tsv")
CP10K = 0.5; MIN_CELLS = 50
op = pd.read_csv(OMNI, sep="\t")
ALZ_DS = {"s1": "203025fe-fa99-4d57-81da-458ed8f0c334", "s2": "ac0c6561-7a48-4185-af6f-af799f699172",
          "s3": "cff99df2-4904-44f7-9173-ff837f95606e"}


def expressed(a, mask):
    sub = a[mask]; X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    return set(pd.Index(sub.var_names)[np.asarray((sp.diags(1e4/tot)@X).mean(0)).ravel() >= CP10K])


def neutral(a, tag, mask):
    if mask.sum() < MIN_CELLS:
        print(f"  skip {tag} ({int(mask.sum())} cells)", flush=True); return
    genes = expressed(a, mask)
    o = op[op.src.isin(genes) & op.dst.isin(genes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    inc = genes & (set(o.src) | set(o.dst)); prot = sorted(inc); d = CTRL/tag; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_id":prot,"node_type":"protein","source":"expressed","direction":"","sender_weight":1.0}).to_csv(d/"network_nodes.tsv",sep="\t",index=False)
    e = o[o.src.isin(inc) & o.dst.isin(inc)]
    pd.DataFrame({"source":e.src,"target":e.dst,"edge_origin":"OmniPath","edge_property":"","weight":1.0,"direction":""}).to_csv(d/"network_edges.tsv",sep="\t",index=False)
    print(f"  {tag:36s} {int(mask.sum()):6d} cells -> {len(prot)} proteins", flush=True)


def donor_split(a, state_mask, tag_prefix, seed=0):
    don = a.obs.donor_id.astype(str).values
    vc = pd.Series(don[state_mask]).value_counts(); donors = sorted(vc[vc >= 20].index)
    perm = np.random.default_rng(seed).permutation(donors); h = len(perm)//2
    A, Bs = set(perm[:h]), set(perm[h:])
    print(f"  {tag_prefix}: {len(donors)} donors -> {len(A)}/{len(Bs)}", flush=True)
    neutral(a, f"{tag_prefix}_splitA", state_mask & np.isin(don, list(A)))
    neutral(a, f"{tag_prefix}_splitB", state_mask & np.isin(don, list(Bs)))


def main():
    CTRL.mkdir(parents=True, exist_ok=True)
    # ---- Crohn inflammatory macrophage controls ----
    aA = ad.read_h5ad(RS/"macrophage_crohn_paired/pulled_macrophages.h5ad")
    aA.obs["state"] = pd.read_csv(RS/"macrophage_crohn_states/cell_states.tsv", sep="\t", index_col=0)["state"].astype(str).values
    aB = ad.read_h5ad(RS/"macrophage_crohn_rep_paired/pulled_macrophages.h5ad")
    aB.obs["state"] = pd.read_csv(RS/"macrophage_crohn_rep_states/cell_states.tsv", sep="\t", index_col=0)["state"].astype(str).values
    print("== Crohn per-study singles ==", flush=True)
    for st in ["inflammatory", "resident"]:
        neutral(aA, f"crohn_mac_{st}_s1", (aA.obs.state == st).values)
        neutral(aB, f"crohn_mac_{st}_s2", (aB.obs.state == st).values)
    print("== Crohn donor-split per state (a37f857c) ==", flush=True)
    for s in sorted(set(aA.obs.state)):
        donor_split(aA, (aA.obs.state == s).values, f"crohn_mac_{s}")

    # ---- Alzheimer microglia controls ----
    am = ad.read_h5ad(RS/"microglia_alzheimers_paired/pulled_microglia.h5ad")
    st = pd.read_csv(RS/"microglia_alzheimers_states/cell_states.tsv", sep="\t", index_col=0)
    am.obs["state"] = st["state"].astype(str).values; am.obs["dataset_id"] = st["dataset_id"].astype(str).values
    states = sorted(set(am.obs.state))
    print("== Alz per-study singles + pair-vs-withheld ==", flush=True)
    for s in states:
        sm = (am.obs.state == s).values
        for sk, ds in ALZ_DS.items():
            neutral(am, f"alz_microglia_{s}_{sk}", sm & (am.obs.dataset_id == ds).values)        # single study
        for i, (sk, ds) in enumerate(ALZ_DS.items(), 1):
            neutral(am, f"alz_microglia_{s}_loo{i}", sm & (am.obs.dataset_id != ds).values)       # pool of the other 2
    print("== Alz donor-split per state (ac0c6561, brain) ==", flush=True)
    for s in sorted(set(am.obs.state)):
        donor_split(am, (am.obs.state == s).values & (am.obs.dataset_id == ALZ_DS["s2"]).values, f"alz_microglia_{s}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
