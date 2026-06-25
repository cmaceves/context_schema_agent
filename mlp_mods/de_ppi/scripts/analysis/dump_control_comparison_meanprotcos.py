"""Control floor recomputed with the UNWEIGHTED mean per-protein cosine metric.

Same control pairs as dump_control_comparison.py (donor_split / between_study / pool_vs_pool), and the
same reference-free consensus as plot_pair_direction_heatmap.py:

  consensus[p] = mean over ALL networks (where present) of Z[.,p]
  r_X[p]       = Z_X[p] - consensus[p]
  mean_prot_cos(A,B) = mean over proteins present in BOTH of  cos(r_A[p], r_B[p])     (UNWEIGHTED)

This differs from dump_control_comparison.py only by dropping the min‖dev‖ weighting, so the number is
directly comparable to tables/celltype_disease_cosine_matrix.tsv (which is also an unweighted mean
per-protein cosine of consensus-centered shifts). shift_mean = mean ‖Z_A-Z_B‖ is carried for reference.

NOTE: unweighted means near-consensus proteins (unstable angle) count equally, so this floor is noisier
/ lower than the weighted heatmap value -- that is the price of matching the cosine-matrix metric.

Output (results/<out_name>/influence_analysis/tables/):
  control_comparison_meanprotcos.tsv          one row per control pair
  control_comparison_meanprotcos_summary.tsv  mean per disease x cell_type x control_type x state

Run: .venv/bin/python mlp_mods/de_ppi/influence_analysis/dump_control_comparison_meanprotcos.py
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
from _layout import tag_tissue

CROHN_STATES = ["inflammatory", "resident", "proliferating"]
ALZ_STATES = ["dam", "homeostatic", "interferon", "proliferating"]


def main(out_name) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    is_prot = d["node_type"] == "protein"

    pi = np.where(is_prot)[0]
    Zp, pres = Z[:, pi, :], present[:, pi]                       # (T, n_prot, dim), (T, n_prot)
    masked = np.where(pres[:, :, None], Zp, np.nan)
    consensus = np.nanmean(masked, axis=0)                       # reference-free origin
    R = Zp - consensus[None]                                     # deviation vectors
    Rn = np.linalg.norm(R, axis=2)
    ti = {t: i for i, t in enumerate(tags)}
    have = set(tags)

    rows = []

    def add(dis, ct, ctype, state, a, b):
        if a not in have or b not in have:
            return
        ta, tb = tag_tissue(a), tag_tissue(b)
        assert ta == tb, f"tissue not controlled: {a}({ta}) vs {b}({tb})"
        ia, ib = ti[a], ti[b]
        both = pres[ia] & pres[ib]
        if both.sum() == 0:
            return
        ra, rb = R[ia, both], R[ib, both]
        na, nb = Rn[ia, both], Rn[ib, both]
        cos = (ra * rb).sum(1) / (na * nb + 1e-9)                # per-protein cosine
        shift = np.linalg.norm(Zp[ia, both] - Zp[ib, both], axis=1)
        rows.append(dict(disease=dis, cell_type=ct, tissue=ta, control_type=ctype, state=state,
                         network_a=a, network_b=b, n_proteins=int(both.sum()),
                         shift=round(float(shift.mean()), 3),
                         mean_prot_cos=round(float(cos.mean()), 3)))

    # donor_split (within-study noise floor)
    for s in CROHN_STATES:
        add("Crohn", "macrophage", "donor_split", s, f"crohn_mac_{s}_splitA", f"crohn_mac_{s}_splitB")
    for s in ALZ_STATES:
        add("Alzheimer", "microglia", "donor_split", s, f"alz_microglia_{s}_splitA", f"alz_microglia_{s}_splitB")
    # between_study (single studies, same state)
    for s in CROHN_STATES:
        add("Crohn", "macrophage", "between_study", s, f"crohn_mac_{s}_s1", f"crohn_mac_{s}_s2")
    for s in ALZ_STATES:
        singles = [t for t in (f"alz_microglia_{s}_{k}" for k in ("s1", "s2", "s3")) if t in have]
        for a, b in combinations(singles, 2):
            add("Alzheimer", "microglia", "between_study", s, a, b)
    # pool_vs_pool (Alz LOO pools)
    for s in ALZ_STATES:
        pools = [t for t in (f"alz_microglia_{s}_loo{i}" for i in (1, 2, 3)) if t in have]
        for a, b in combinations(pools, 2):
            add("Alzheimer", "microglia", "pool_vs_pool", s, a, b)

    df = pd.DataFrame(rows)
    out = res / "tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "control_comparison_meanprotcos.tsv", sep="\t", index=False)
    summ = (df.groupby(["disease", "cell_type", "tissue", "control_type", "state"])
            .agg(n=("shift", "size"), shift_mean=("shift", "mean"), mean_prot_cos=("mean_prot_cos", "mean"))
            .round(3).reset_index().sort_values(["disease", "state", "control_type"]))
    summ.to_csv(out / "control_comparison_meanprotcos_summary.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_comparison_meanprotcos.tsv'} ({len(df)} pairs)")
    print(f"wrote {out/'control_comparison_meanprotcos_summary.tsv'}\n")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
