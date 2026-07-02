"""Histogram of the per-protein cross-disease direction cosine (disease_cos = cos(r_crohn, r_uc)), colored by
OpenTargets label: SHARED positives (OT-associated with BOTH Crohn and UC) vs 100 negatives (non-associated in
both). Restricted to MOVING proteins (label != static) so the cosine is well-defined (cosine of ~zero shift
vectors is noise). Shared plot_style (whitegrid, no top/right spines, Paul Tol colors).

disease_cos is +1 when Crohn and UC push a protein the same way, -1 when divergent.

Output: results/<main>/disease_axis/disease_cos_hist.png

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/plot_disease_cos_hist.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, TOL

OT_DIR = Path("mlp_mods/opentargets_associations")
OT = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv"}
POS_THR = 0.5                # high-confidence positive: OT score_indirect >= this in BOTH diseases
NEG_THR = 0.1                # negative: OT score_indirect < this in BOTH diseases
N_NEG = 100
SEED = 0


def main(main_name) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    d = pd.read_csv(res / "disease_axis" / "disease_axis_proteins.tsv", sep="\t")
    d = d[d.label != "static"].copy()                                  # cosine only meaningful for moving proteins
    for arm, f in OT.items():
        ot = pd.read_csv(OT_DIR / f, sep="\t")
        d[f"s_{arm}"] = d.protein.map(dict(zip(ot.gene_symbol, ot.score_indirect))).fillna(0.0)

    shared_pos = d[(d.s_crohn >= POS_THR) & (d.s_uc >= POS_THR)]        # high-confidence OT+ in BOTH diseases
    neg_pool = d[(d.s_crohn < NEG_THR) & (d.s_uc < NEG_THR)]           # non-associated in both
    neg = neg_pool.sample(min(N_NEG, len(neg_pool)), random_state=SEED)

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bins = np.linspace(-1, 1, 41)
    ax.hist(shared_pos.disease_cos, bins=bins, density=True, color=TOL["green"], alpha=0.8,
            label=f"shared OT+ (both, n={len(shared_pos)}, med {shared_pos.disease_cos.median():.2f})")
    ax.hist(neg.disease_cos, bins=bins, density=True, histtype="step", lw=2, color=TOL["grey"],
            label=f"OT- (both, n={len(neg)}, med {neg.disease_cos.median():.2f})")
    ax.axvline(shared_pos.disease_cos.median(), color=TOL["green"], ls="--", lw=1)
    ax.axvline(neg.disease_cos.median(), color=TOL["grey"], ls="--", lw=1)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("disease_cos = cos(r$_{crohn}$, r$_{uc}$)   (+1 shared direction, −1 divergent)")
    ax.set_ylabel("density")
    ax.set_title(f"Crohn↔UC per-protein shift-direction agreement (colon macrophage, moving proteins)\n"
                 f"[{main_name.split('_')[-1]}]", fontsize=10)
    ax.legend()
    fig.tight_layout()
    out = res / "disease_axis" / "disease_cos_hist.png"
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"  shared OT+ (both): n={len(shared_pos)}  median disease_cos={shared_pos.disease_cos.median():.3f}")
    print(f"  OT- sample (both): n={len(neg)}  median disease_cos={neg.disease_cos.median():.3f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name))
