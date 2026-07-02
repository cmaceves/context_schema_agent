"""Within-study / between-study noise vs the real factor contrasts -- ON ONE EMBEDDING, ONE METRIC.

Purpose: put the control floors (donor_split = within-study, between_study, pool_vs_pool) on the SAME
ladder as the real disease-state network contrasts (diff cell state / disease / tissue / cell type), so
"how big is study noise vs how big is a disease effect" is a fair comparison.

Both sides are computed here in the control-containing embedding (default crohn_alzheimer_ild_uc, the
68-network build) with the IDENTICAL unweighted metric used by dump_control_comparison_meanprotcos.py:

  consensus[p]  = mean over MAIN (non-control) networks of Z[.,p]      (control tags excluded)
  r_X[p]        = Z_X[p] - consensus[p]
  mean_prot_cos = mean over proteins present in BOTH (and >=1 main net) of cos(r_A[p], r_B[p])
  shift         = mean ||Z_A[p]-Z_B[p]|| over the same proteins

This guarantees the factor contrasts are directly comparable to the control floor (same encoder space,
same metric, same consensus) -- unlike factor_combinations_pairs.tsv, which is the WEIGHTED cosine in the
30-network _expressed embedding.

control pairs reuse the exact pairing of dump_control_comparison_meanprotcos.py. Factor pairs = all pairs of
MAIN disease-state networks (healthy + control replicates excluded), each labeled by its LOWEST differing
factor (the most-controlled contrast it represents):
  diff_state  (cell type, tissue, disease all same; only state differs)
  diff_disease(cell type, tissue same; disease differs)
  diff_tissue (cell type same; tissue differs)
  diff_celltype(cell type differs)

Output (results/<out_name>/tables/):
  floor_vs_factors_pairs.tsv    one row per pair (control + factor), category, n_proteins, shift, mean_prot_cos
  floor_vs_factors_summary.tsv  the ladder: per category -> n, shift_mean, mean_prot_cos_mean (+/- sd)

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/compare_floor_vs_factors.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_celltype, tag_tissue, tag_disease, tag_state

CROHN_STATES = ["inflammatory", "resident", "proliferating"]
ALZ_STATES = ["dam", "homeostatic", "interferon", "proliferating"]

# ladder order (low -> high expected variation)
CAT_ORDER = ["donor_split", "between_study",
             "diff_state", "diff_disease", "diff_tissue", "diff_celltype"]


def main(out_name, main_build) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    is_prot = d["node_type"] == "protein"
    main_tags = set(np.load(DE_PPI / "results" / main_build / "embeddings.npz", allow_pickle=True)["tags"])
    is_main = np.array([t in main_tags for t in tags])
    print(f"embedding {out_name}: {len(tags)} networks ({int(is_main.sum())} MAIN feed the consensus)", flush=True)

    pi = np.where(is_prot)[0]
    Zp, pres = Z[:, pi, :], present[:, pi]
    contrib = pres & is_main[:, None]
    masked = np.where(contrib[:, :, None], Zp, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmean(masked, axis=0)
    valid = ~np.isnan(consensus).any(axis=1)
    R = Zp - consensus[None]
    Rn = np.linalg.norm(R, axis=2)
    ti = {t: i for i, t in enumerate(tags)}
    have = set(tags)
    rows = []

    def metric(a, b):
        ia, ib = ti[a], ti[b]
        both = pres[ia] & pres[ib] & valid
        if both.sum() == 0:
            return None
        ra, rb = R[ia, both], R[ib, both]
        na, nb = Rn[ia, both], Rn[ib, both]
        cos = (ra * rb).sum(1) / (na * nb + 1e-9)
        shift = np.linalg.norm(Zp[ia, both] - Zp[ib, both], axis=1)
        return int(both.sum()), float(shift.mean()), float(cos.mean())

    def add(category, a, b):
        if a not in have or b not in have:
            return
        m = metric(a, b)
        if m is None:
            return
        n, shift, cos = m
        rows.append(dict(category=category, network_a=a, network_b=b, n_proteins=n,
                         shift=round(shift, 3), mean_prot_cos=round(cos, 3)))

    # ---- control floors (same pairing as dump_control_comparison_meanprotcos) ----
    for s in CROHN_STATES:
        add("donor_split", f"crohn_mac_{s}_splitA", f"crohn_mac_{s}_splitB")
    for s in ALZ_STATES:
        add("donor_split", f"alz_microglia_{s}_splitA", f"alz_microglia_{s}_splitB")
    for s in CROHN_STATES:
        add("between_study", f"crohn_mac_{s}_s1", f"crohn_mac_{s}_s2")
    for s in ALZ_STATES:
        singles = [t for t in (f"alz_microglia_{s}_{k}" for k in ("s1", "s2", "s3")) if t in have]
        for a, b in combinations(singles, 2):
            add("between_study", a, b)
    # (pool_vs_pool control removed; superseded by control m = healthy_loo)

    # ---- factor contrasts: MAIN disease-state networks only (drop healthy + control replicates) ----
    def is_control(t):
        return ("split" in t or t.endswith(("_s1", "_s2", "_s3"))
                or "_loo" in t or t.startswith("healthy_"))
    main_ds = [t for t in tags if is_main[ti[t]] and not is_control(t)]

    def factor_class(a, b):
        if tag_celltype(a) != tag_celltype(b):
            return "diff_celltype"
        if tag_tissue(a) != tag_tissue(b):
            return "diff_tissue"
        if tag_disease(a) != tag_disease(b):
            return "diff_disease"
        if tag_state(a) != tag_state(b):
            return "diff_state"
        return "identical"

    for a, b in combinations(sorted(main_ds), 2):
        add(factor_class(a, b), a, b)

    df = pd.DataFrame(rows)
    out = res / "tables"
    df.to_csv(out / "floor_vs_factors_pairs.tsv", sep="\t", index=False)

    summ = (df.groupby("category")
            .agg(n_pairs=("shift", "size"),
                 shift_mean=("shift", "mean"), shift_sd=("shift", "std"),
                 mean_prot_cos=("mean_prot_cos", "mean"), cos_sd=("mean_prot_cos", "std"))
            .reindex([c for c in CAT_ORDER if c in set(df.category)]).round(3).reset_index())
    summ.to_csv(out / "floor_vs_factors_summary.tsv", sep="\t", index=False)

    print(f"\nwrote {out/'floor_vs_factors_pairs.tsv'} ({len(df)} pairs)")
    print(f"wrote {out/'floor_vs_factors_summary.tsv'}\n")
    print("=== ladder: study noise vs factor contrasts (one embedding, unweighted mean_prot_cos) ===")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc",
                    help="control-containing embedding (has both control + main networks)")
    ap.add_argument("--main-build", default="crohn_alzheimer_ild_uc_embedding_expressed",
                    help="build whose tags define MAIN (non-control) networks for the consensus")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.main_build))
