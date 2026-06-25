"""Tabulate the embedding SHIFT (magnitude) and DIRECTION (cosine) differences between every control
network pair set up in results/crohn_alzheimer_ild_uc/ — so the donor-split floor, between-study batch,
and pair-vs-withheld can be read side by side, per disease/cell type.

Control types:
  donor_split       : same population, donors split in two within ONE study (noise floor)
  between_study     : same state, two/three independent single studies (study batch)
  pool_vs_pool      : two different (n-1)-study pools (sharing one study) vs each other = how stable the
                      pooled signal is to which studies are included (Alz only; the LOO stability test)

Reads pair_shift_heatmap_mean.tsv + pair_direction_heatmap.tsv (run those plots first).
Writes tables/control_comparison.tsv (one row per control pair) and tables/control_comparison_summary.tsv
(mean per disease x control_type).

Run: .venv/bin/python mlp_mods/de_ppi/influence_analysis/dump_control_comparison.py
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
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_tissue
CROHN_STATES = ["inflammatory", "resident", "proliferating"]
ALZ_STATES = ["dam", "homeostatic", "interferon", "proliferating"]


def main(out_name) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(res / "tables" / "pair_shift_heatmap_mean.tsv", sep="\t", index_col=0)
    D = pd.read_csv(res / "tables" / "pair_direction_heatmap.tsv", sep="\t", index_col=0)
    have = set(S.index)
    rows = []

    def add(dis, ct, ctype, state, a, b):
        if a in have and b in have:
            ta, tb = tag_tissue(a), tag_tissue(b)
            assert ta == tb, f"tissue not controlled: {a}({ta}) vs {b}({tb})"   # guard: same tissue only
            rows.append(dict(disease=dis, cell_type=ct, tissue=ta, control_type=ctype, state=state,
                             network_a=a, network_b=b, shift=round(float(S.loc[a, b]), 3),
                             dir_cos=round(float(D.loc[a, b]), 3)))

    # --- donor-split (within one study), every state ---
    for s in CROHN_STATES:
        add("Crohn", "macrophage", "donor_split", s, f"crohn_mac_{s}_splitA", f"crohn_mac_{s}_splitB")
    for s in ALZ_STATES:
        add("Alzheimer", "microglia", "donor_split", s, f"alz_microglia_{s}_splitA", f"alz_microglia_{s}_splitB")

    # --- between-study (single studies, same state) ---
    for s in CROHN_STATES:
        add("Crohn", "macrophage", "between_study", s, f"crohn_mac_{s}_s1", f"crohn_mac_{s}_s2")
    for s in ALZ_STATES:
        singles = [f"alz_microglia_{s}_{k}" for k in ("s1", "s2", "s3")]
        for a, b in combinations([t for t in singles if t in have], 2):
            add("Alzheimer", "microglia", "between_study", s, a, b)

    # --- pool-vs-pool (Alz: two (n-1)-study pools vs each other = pooled-signal stability) ---
    for s in ALZ_STATES:
        pools = [f"alz_microglia_{s}_loo{i}" for i in (1, 2, 3)]
        for a, b in combinations([t for t in pools if t in have], 2):
            add("Alzheimer", "microglia", "pool_vs_pool", s, a, b)

    df = pd.DataFrame(rows)
    out = res / "tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "control_comparison.tsv", sep="\t", index=False)
    summ = (df.groupby(["disease", "cell_type", "tissue", "control_type", "state"])   # state + tissue kept separate
            .agg(n=("shift", "size"), shift_mean=("shift", "mean"), dir_cos_mean=("dir_cos", "mean"))
            .round(3).reset_index().sort_values(["disease", "state", "control_type"]))
    summ.to_csv(out / "control_comparison_summary.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_comparison.tsv'} ({len(df)} pairs)\nwrote {out/'control_comparison_summary.tsv'}\n")
    print("=== per-pair ==="); print(df.to_string(index=False))
    print("\n=== summary (mean per disease x control type) ==="); print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
