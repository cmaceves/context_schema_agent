"""Per-protein embedding-shift magnitude histograms: rows = cell states (top = pooled allstates), cols = diseases.

For each (disease arm, cell state) cell we plot the distribution of per-protein magnitude change
||Z_disease(state) - Z_healthy(allstates)||, centering each disease cell-state on its OWN-STUDY healthy
ALLSTATES arm (state-relaxed but study-matched, so between-study batch is still cancelled), averaged over that
arm's studies. 100 random OpenTargets-NON-associated proteins are overlaid as a negative control. Uses the
shared plot_style (whitegrid, no top/right spines, Paul Tol colors).

The healthy reference is the pooled cell-type baseline, NOT state-matched -- so a cell's magnitude includes the
state-vs-allstates composition difference, not only per-state molecular change. A (disease, state) cell is EMPTY
only when no disease net of that state exists at all (e.g. Crohn colon macrophage has no resident/proliferating
net) -- annotated, not faked.

Output: results/<main>/disease_axis/per_disease_magnitude_hist.png

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/plot_disease_magnitude_hist.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from disease_axis_decompose import parse
from plot_style import apply_style, ARM_COLOR

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv"}
POS_THR = 0.5                # OT score_indirect >= this => high-confidence "true positive"
NEG_THR = 0.1                # OT score_indirect below this (or absent) => "non-associated" negative
N_NEG = 100
SEED = 0


def main(main_name, tissue, ct, arms, states, move_clip_pct, restrict_col=None, restrict_min=0.5) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres = c["Z"][:, pi, :], c["present"][:, pi]
    node_id = np.asarray(c["node_id"])[pi]
    tags = list(c["tags"]); idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}

    def primary(q):
        return q["tissue"] == tissue and q["ct"] == ct and not q["loo"] and q["split"] is None

    # own-study healthy ALLSTATES tag per study: the cell-type healthy baseline each disease state is centered on.
    # State-relaxed (not state-matched) but study-matched, so between-study batch is still cancelled -- this lets
    # every disease-state net be scored even where no same-state healthy net exists (e.g. healthy inflammatory).
    Hall = {P[t]["study"]: t for t in tags if primary(P[t]) and P[t]["arm"] == "healthy" and P[t]["state"] == "allstates"}

    def arm_state(arm, state):
        """(present-mask, per-protein magnitude) for (arm, state) vs own-study healthy allstates; None if no disease net."""
        pairs = [(idx[t], idx[Hall[P[t]["study"]]]) for t in tags
                 if primary(P[t]) and P[t]["arm"] == arm and P[t]["state"] == state and P[t]["study"] in Hall]
        if not pairs:
            return None
        m = np.ones(Z.shape[1], bool)
        for di, hi in pairs:
            m &= pres[di] & pres[hi]
        r = np.mean([Z[di] - Z[hi] for di, hi in pairs], axis=0)     # mean healthy-centered shift over studies
        return m, np.linalg.norm(r, axis=1)

    # per-arm OT-POSITIVE and OT-NEGATIVE protein sets, defined once (same across states) within THIS cell type's
    # node set (arm's allstates present-mask). positives = OT association >= NEG_THR; negatives = < NEG_THR
    # (absent from OT -> score 0 -> negative), sampled to N_NEG.
    # optional protein filter from disease_axis_proteins.tsv (e.g. keep only reproducible proteins uc_xstudy_cos>0.5)
    allow_mask = np.ones(len(node_id), bool)
    if restrict_col:
        dax = pd.read_csv(res / "disease_axis" / "disease_axis_proteins.tsv", sep="\t")
        keep = set(dax.loc[dax[restrict_col] > restrict_min, "protein"])
        allow_mask = np.isin(node_id, list(keep))
        print(f"restrict {restrict_col} > {restrict_min}: {int(allow_mask.sum())}/{len(node_id)} proteins kept")

    rng = np.random.default_rng(SEED)
    pos, negs = {}, {}
    for arm in arms:
        base = arm_state(arm, "allstates")
        ct_present = base[0] if base is not None else np.ones(Z.shape[1], bool)
        ot = pd.read_csv(OT_DIR / OT_FILE[arm], sep="\t")
        score = pd.Series(node_id).map(dict(zip(ot.gene_symbol, ot.score_indirect))).fillna(0.0).to_numpy()
        pos[arm] = set(np.where((score >= POS_THR) & ct_present & allow_mask)[0])
        negpool = np.where((score < NEG_THR) & ct_present & allow_mask)[0]
        negs[arm] = set(rng.choice(negpool, min(N_NEG, len(negpool)), replace=False))

    # gather all magnitudes first (for shared bins + to know which cells are populated)
    data = {}                                                       # (arm, state) -> (mask, mag)
    allvals = []
    for arm in arms:
        for st in states:
            r = arm_state(arm, st)
            data[(arm, st)] = r
            if r is not None:
                allvals.append(r[1][r[0] & allow_mask])
    hi = np.percentile(np.concatenate(allvals), move_clip_pct)
    bins = np.linspace(0, hi, 45)

    apply_style()
    nr, nc = len(states), len(arms)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 2.5 * nr), squeeze=False, sharex=True)
    for ri, st in enumerate(states):
        for ci, arm in enumerate(arms):
            ax = axes[ri][ci]
            r = data[(arm, st)]
            if r is None:
                ax.text(0.5, 0.5, f"no {arm} disease net\nfor {st}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
                ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"{arm}", fontsize=11)
                continue
            m, mag = r
            present_ix = set(np.where(m & allow_mask)[0])
            posmag = mag[np.array(sorted(pos[arm] & present_ix))]
            negmag = mag[np.array(sorted(negs[arm] & present_ix))]
            ax.hist(posmag, bins=bins, density=True, color=ARM_COLOR[arm], alpha=0.85,
                    label=f"OT positives (n={len(posmag)}, med {np.median(posmag):.2f})")
            ax.hist(negmag, bins=bins, density=True, histtype="step", lw=1.8, color=ARM_COLOR["negative"],
                    label=f"OT-neg (n={len(negmag)}, med {np.median(negmag):.2f})")
            ax.axvline(np.median(posmag), color=ARM_COLOR[arm], ls="--", lw=1)
            ax.legend()
            if ri == 0:
                ax.set_title(f"{arm}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{st}\ndensity", fontsize=9)
            if ri == nr - 1:
                ax.set_xlabel("‖healthy-centered shift‖ (embed L2)")
    filt = f";  {restrict_col}>{restrict_min}" if restrict_col else ""
    fig.suptitle(f"{tissue} {ct}  ‖Δ embedding‖  (rows: cell state, cols: disease){filt}\n"
                 f"filled = OT positives (score≥{POS_THR}); grey = {N_NEG} OT-negatives (<{NEG_THR})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    suffix = f"_{restrict_col}_gt{restrict_min}" if restrict_col else ""
    out = res / "disease_axis" / f"per_disease_magnitude_hist{suffix}.png"
    fig.savefig(out)
    print(f"wrote {out}")
    for (arm, st), r in data.items():
        print(f"  {arm:6s} {st:14s} " + ("EMPTY (no disease net for this state)" if r is None
                                          else f"n={int(r[0].sum())}  median_mag={np.median(r[1][r[0]]):.3f}"))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--tissue", default="colon")
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--arms", default="crohn,uc")
    ap.add_argument("--states", default="allstates,inflammatory,resident,proliferating")
    ap.add_argument("--move-clip-pct", type=float, default=99.0, help="x-axis upper limit percentile (tail clip)")
    ap.add_argument("--restrict-col", default=None,
                    help="keep only proteins with disease_axis_proteins.tsv[col] > --restrict-min (e.g. uc_xstudy_cos)")
    ap.add_argument("--restrict-min", type=float, default=0.5)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.tissue, a.celltype,
                          [x for x in a.arms.split(",") if x], [x for x in a.states.split(",") if x],
                          a.move_clip_pct, a.restrict_col, a.restrict_min))
