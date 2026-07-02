"""Per-protein test of how much of the LEARNED embedding shift is explained by the node's OWN expression change.

For each disease-vs-matched-healthy contrast (same tissue/celltype/state, disease arm vs healthy arm) we plot,
over proteins present in BOTH networks:
    x = |Δexpression|   = |expr_disease - expr_healthy|   (absolute change in the corrected pseudobulk node feature)
    y = ||ΔZ||          = learned embedding shift magnitude between the two networks (from embedding_shift.tsv)
and fit y ~ x (OLS), annotating each panel with R^2 and n.

Interpretation: HIGH R^2 -> the embedding shift just tracks the node's own expression change (embedding is
redundant with raw). LOW R^2 -> the embedding moves for reasons NOT explained by own expression = neighbour /
topology contribution (the part of the embedding that raw expression cannot capture). Low R^2 is necessary but
not sufficient for "useful topology" (residual could be noise) -- it is a clean, interpretable first cut.

One panel per contrast (NOT pooled across cell types: different topology -> different slope, pooling would give a
misleading aggregate R^2). Uses the main pooled-per-context networks, which is what embedding_shift.tsv is built on.

Output (results/<main>/images/expr_vs_embshift/):
    expr_vs_embshift_facets.png   faceted scatter, R^2 per panel
    expr_vs_embshift_r2.tsv       per-contrast n, slope, R^2, pearson_r

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/expr_change_vs_embshift.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISEASES = ("alz", "crohn", "uc", "ild")
MIN_POINTS = 20          # skip a contrast with too few shared proteins to fit


def load_expr(nets_dir: Path, net: str) -> pd.Series:
    """Protein-only expression feature for one network, indexed by node_id."""
    d = pd.read_csv(nets_dir / net / "network_nodes.tsv", sep="\t", keep_default_na=False)
    d = d[d["node_type"] == "protein"]
    return pd.Series(d["expression"].astype(float).values, index=d["node_id"].values)


def find_shift_col(cols: set[str], a: str, b: str) -> str | None:
    """||ΔZ|| is symmetric but stored under one column order; try both."""
    for c in (f"shift_{a}_{b}", f"shift_{b}_{a}"):
        if c in cols:
            return c
    return None


def r2_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS y~x: returns (r2, slope, pearson_r). r2 == pearson_r**2 for a linear fit."""
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")
    return r2, float(slope), r


def main(main_name: str) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    nets_dir = res / "networks"
    sdf = pd.read_csv(res / "embedding_shift.tsv", sep="\t", keep_default_na=False,
                      low_memory=False).set_index("node_id")
    shift_cols = set(c for c in sdf.columns if c.startswith("shift_"))
    present = {c[len("present_"):]: sdf[c].astype(str).str.lower().isin(("true", "1"))
               for c in sdf.columns if c.startswith("present_")}

    all_nets = sorted(p.name for p in nets_dir.iterdir() if (p / "network_nodes.tsv").exists())

    # build disease -> matched-healthy contrasts that exist as nets AND have a shift column
    panels = []
    for dnet in all_nets:
        arm = dnet.split("_", 1)[0]
        if arm not in DISEASES:
            continue
        hnet = "healthy_" + dnet.split("_", 1)[1]      # same tissue_celltype_state, healthy arm
        if hnet not in all_nets:
            continue
        col = find_shift_col(shift_cols, dnet, hnet)
        if col is None:
            continue
        ed, eh = load_expr(nets_dir, dnet), load_expr(nets_dir, hnet)
        common = ed.index.intersection(eh.index)
        if dnet in present and hnet in present:        # restrict to proteins flagged present in both
            pmask = present[dnet] & present[hnet]
            common = common.intersection(pmask.index[pmask])
        common = common.intersection(sdf.index[sdf[col].astype(str) != ""])
        if len(common) < MIN_POINTS:
            continue
        x = np.abs(ed.reindex(common).values - eh.reindex(common).values)
        y = pd.to_numeric(sdf[col].reindex(common), errors="coerce").values
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < MIN_POINTS:
            continue
        r2, slope, r = r2_slope(x, y)
        panels.append(dict(contrast=dnet.split("_", 1)[1], disease=arm, disease_net=dnet, healthy_net=hnet,
                           n=int(x.size), r2=r2, slope=slope, pearson_r=r, x=x, y=y))

    if not panels:
        print("no contrasts with a matched-healthy net + shift column found")
        return 1

    panels.sort(key=lambda p: (p["disease"], p["contrast"]))
    out_dir = res / "images" / "expr_vs_embshift"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-contrast R^2 table ----
    tab = pd.DataFrame([{k: p[k] for k in ("disease", "contrast", "n", "slope", "r2", "pearson_r")} for p in panels])
    tab = tab.round({"slope": 4, "r2": 4, "pearson_r": 4})
    tab.to_csv(out_dir / "expr_vs_embshift_r2.tsv", sep="\t", index=False)

    # ---- faceted scatter ----
    n = len(panels)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for p, ax in zip(panels, axes.flat):
        ax.axis("on")
        ax.scatter(p["x"], p["y"], s=6, alpha=0.35, edgecolors="none", color="#3b6fb0")
        xs = np.array([p["x"].min(), p["x"].max()])
        ax.plot(xs, p["slope"] * xs + (p["y"].mean() - p["slope"] * p["x"].mean()), color="#c0392b", lw=1.3)
        ax.set_title(f"{p['disease']} · {p['contrast']}", fontsize=8)
        ax.text(0.04, 0.95, f"$R^2$={p['r2']:.2f}\nn={p['n']}", transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
        ax.tick_params(labelsize=6)
    fig.supxlabel("|Δ expression|  (disease − healthy, log1p CP10k)", fontsize=10)
    fig.supylabel("‖Δ embedding‖  (disease vs healthy, L2 64d)", fontsize=10)
    fig.suptitle("Per-protein: does own expression change explain the learned embedding shift?", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out_dir / "expr_vs_embshift_facets.png", dpi=150)

    print(f"contrasts plotted: {n}")
    print(tab.to_string(index=False))
    print(f"\nR^2 across contrasts: median={tab.r2.median():.3f}  min={tab.r2.min():.3f}  max={tab.r2.max():.3f}")
    print(f"wrote {out_dir/'expr_vs_embshift_facets.png'}")
    print(f"wrote {out_dir/'expr_vs_embshift_r2.tsv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name))
