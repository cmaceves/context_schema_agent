"""Per-protein |CHANGE IN EXPRESSION| histograms: rows = cell states (top = pooled allstates), cols = diseases.

Sibling of plot_disease_magnitude_hist.py, but the x-axis is the change in the raw NODE FEATURE (corrected
log1p CP10k expression), not the embedding shift. Per (disease arm, cell state) we plot the distribution of
    |Delta expr|(p) = | mean_studies( expr_disease(study,state)[p] - expr_healthy(study,allstates)[p] ) |
centering each disease cell-state on its OWN-STUDY healthy ALLSTATES arm (state-relaxed, study-matched, so
between-study batch is cancelled), averaged over that arm's studies. 100 random OpenTargets-NON-associated
proteins are overlaid as a negative control. Shared plot_style (whitegrid, no top/right spines, Paul Tol colors).

Expression is read from each control network's network_nodes.tsv `expression` column; the node universe/order
and presence masks come from controls/control_embeddings.npz (so cells match the embedding-magnitude figure).
The healthy reference is the pooled cell-type baseline (allstates), NOT state-matched, so a cell's |Delta expr|
includes the state-vs-allstates composition difference. A cell is EMPTY only when no disease net of that state
exists (e.g. Crohn colon macrophage has no resident/proliferating net).

Output: results/<main>/disease_axis/per_disease_expr_change_hist.png

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/plot_disease_expr_change_hist.py \
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


def main(main_name, tissue, ct, arms, states, clip_pct) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    pres = c["present"][:, pi]
    node_id = np.asarray(c["node_id"])[pi]
    tags = list(c["tags"]); idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}
    net_dir = res / "controls" / "networks"

    def expr_vec(tag):                                            # per-net expression on the protein universe (NaN if absent)
        d = pd.read_csv(net_dir / tag / "network_nodes.tsv", sep="\t", keep_default_na=False)
        d = d[d["node_type"] == "protein"]
        return pd.Series(d["expression"].astype(float).values, index=d["node_id"].values).reindex(node_id).values

    expr = {t: expr_vec(t) for t in tags if (net_dir / t / "network_nodes.tsv").exists()}

    def primary(q):
        return q["tissue"] == tissue and q["ct"] == ct and not q["loo"] and q["split"] is None

    Hall = {P[t]["study"]: t for t in tags if primary(P[t]) and P[t]["arm"] == "healthy" and P[t]["state"] == "allstates"}

    def arm_state(arm, state):
        """(present-mask, |Delta expr|) for (arm,state) vs own-study healthy allstates; None if no disease net."""
        trip = [(t, Hall[P[t]["study"]]) for t in tags
                if primary(P[t]) and P[t]["arm"] == arm and P[t]["state"] == state and P[t]["study"] in Hall]
        if not trip:
            return None
        m = np.ones(len(node_id), bool)
        deltas = []
        for dt, ht in trip:
            m &= pres[idx[dt]] & pres[idx[ht]]
            deltas.append(expr[dt] - expr[ht])
        d = np.nanmean(deltas, axis=0)                            # mean expression change over studies
        return m, np.abs(d)

    # OT-POSITIVE (score >= NEG_THR) and OT-NEGATIVE (< NEG_THR) protein sets within this cell type's node set.
    rng = np.random.default_rng(SEED)
    pos, negs = {}, {}
    for arm in arms:
        base = arm_state(arm, "allstates")
        ct_present = base[0] if base is not None else np.ones(len(node_id), bool)
        ot = pd.read_csv(OT_DIR / OT_FILE[arm], sep="\t")
        score = pd.Series(node_id).map(dict(zip(ot.gene_symbol, ot.score_indirect))).fillna(0.0).to_numpy()
        pos[arm] = set(np.where((score >= POS_THR) & ct_present)[0])
        negs[arm] = set(rng.choice(np.where((score < NEG_THR) & ct_present)[0], N_NEG, replace=False))

    data, allvals = {}, []
    for arm in arms:
        for st in states:
            r = arm_state(arm, st)
            data[(arm, st)] = r
            if r is not None:
                allvals.append(r[1][r[0]])
    bins = np.linspace(0, np.percentile(np.concatenate(allvals), clip_pct), 45)

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
            present_ix = set(np.where(m)[0])
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
                ax.set_xlabel("|Δ expression|  (corrected log1p CP10k)")
    fig.suptitle(f"{tissue} {ct}  |Δ expression|  (rows: cell state, cols: disease)\n"
                 f"filled = OT positives (score≥{POS_THR}); grey = {N_NEG} OT-negatives (<{NEG_THR})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = res / "disease_axis" / "per_disease_expr_change_hist.png"
    fig.savefig(out)
    print(f"wrote {out}")
    for (arm, st), r in data.items():
        print(f"  {arm:6s} {st:14s} " + ("EMPTY (no disease net for this state)" if r is None
                                          else f"n={int(r[0].sum())}  median|Δexpr|={np.median(r[1][r[0]]):.3f}"))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--tissue", default="colon")
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--arms", default="crohn,uc")
    ap.add_argument("--states", default="allstates,inflammatory,resident,proliferating")
    ap.add_argument("--clip-pct", type=float, default=99.0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.tissue, a.celltype,
                          [x for x in a.arms.split(",") if x], [x for x in a.states.split(",") if x], a.clip_pct))
