"""Per-drug scatter of INFLUENCE percentile in the INFLAMMATORY vs RESIDENT macrophage
state networks, for a disease's drug panel. One point per (drug, mechanism-of-action) =
that mechanism's best in-network target; x = inflammatory percentile, y = resident.

  below the y=x diagonal -> inflammatory-leaning (target reaches the inflammatory
                            dysregulated set more than the resident one)
  above                  -> resident-leaning
  on the diagonal        -> state-agnostic (hub-like across states)

Color = signal magnitude (max raw influence across states); near-zero-reach targets stay
dark so the percentile transform doesn't over-confidently place low-signal points.
Uses UNSIGNED influence (signed influence / Effect were removed — too fragile to network gaps).

Output: de_ppi/results/macrophage_crohn_inflammatory/influence_analysis/state_influence_scatter.png

Run with .venv:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_state_influence_scatter.py [--metric influence|specificity]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
sys.path.insert(0, str(DE_PPI))                                   # de_ppi on path
from config import load_build

DEFAULT_JOINT = DE_PPI / "results" / "macrophage_crohn_embedding" / "joint_influence.tsv"


def pct_map(build: str, metric: str, joint_path: Path | None = None):
    """Return (percentile dict, raw-value dict) for the build's proteins, ranked by `metric`.

    `joint_influence` reads the JOINT table (joint_influence.tsv) and uses that build's own
    influence_<state> column (shared-space, cross-state-comparable). `embed_influence` reads the
    per-build embedding_influence.tsv. Anything else is a column of the analytic P3_influence.tsv.
    """
    if metric == "joint_influence":
        df = (pd.read_csv(joint_path or DEFAULT_JOINT, sep="\t")
              .rename(columns={"node_id": "protein"}).query("node_type == 'protein'"))
        col = f"influence_{_slabel(build)}"
        df = df.dropna(subset=[col])                              # nodes absent from this state
    elif metric == "embed_influence":
        df = (pd.read_csv(load_build(build).embed_influence, sep="\t")
              .rename(columns={"node_id": "protein"}).query("node_type == 'protein'"))
        col = "embed_influence"
    else:
        df = pd.read_csv(load_build(build).p3_influence, sep="\t")
        col = metric
    dfs = df.sort_values(col, ascending=False).reset_index(drop=True)
    n = len(dfs)
    pct = {g: 100.0 * (1.0 - i / n) for i, g in enumerate(dfs.protein)}
    raw = dict(zip(df.protein, df[col]))
    return pct, raw


def _slabel(build: str) -> str:
    return build.replace("macrophage_crohn_", "")


def main(metric: str = "influence", raw: bool = False,
         x_build: str = "macrophage_crohn_inflammatory",
         y_build: str = "macrophage_crohn_resident", joint_path: Path | None = None) -> int:
    xlab, ylab = _slabel(x_build), _slabel(y_build)
    inf_pct, inf_raw = pct_map(x_build, metric, joint_path)
    res_pct, res_raw = pct_map(y_build, metric, joint_path)
    innet = set(inf_pct) & set(res_pct)

    cfg = load_build(x_build)                                # Crohn drug panel + scope (shared)
    kd = pd.read_parquet(cfg.known_drugs_parquet)
    scope = cfg.drug_scope
    anc = lambda a: isinstance(a, (list, np.ndarray)) and any(s in set(a) for s in scope)
    insc = kd[kd["diseaseId"].isin(scope) | kd["ancestors"].apply(anc)]

    xv = inf_raw if raw else inf_pct                         # axes = raw value or percentile
    yv = res_raw if raw else res_pct
    pts = []                                                 # (x_inf, y_res, target, magnitude)
    for _, dg in insc.groupby("drugId"):
        for _, mg in dg.groupby("mechanismOfAction", dropna=False):
            tg = [t for t in {s for s in mg.approvedSymbol.dropna() if s} if t in innet]
            if not tg:
                continue
            best = max(tg, key=lambda t: max(inf_pct[t], res_pct[t]))   # consistent anchor target
            mag = max(abs(inf_raw[best]), abs(res_raw[best]))
            pts.append((xv[best], yv[best], best, mag))
    df = pd.DataFrame(pts, columns=["inf", "res", "target", "mag"]).drop_duplicates("target")
    unit = "raw value" if raw else "percentile"

    fig, ax = plt.subplots(figsize=(13, 11))
    hi = max(df.inf.max(), df.res.max()) if raw else 100
    ax.plot([0, hi], [0, hi], ls="--", color="#999999", lw=1, zorder=0)
    sc = ax.scatter(df.inf, df.res, c=df.mag, cmap="viridis", vmin=0,
                    s=80, edgecolor="black", linewidth=0.4, zorder=3)
    for r in df.itertuples():                                # label every target
        ax.annotate(r.target, (r.inf, r.res), fontsize=8.5, xytext=(2.5, 2.5), textcoords="offset points")
    ax.text(0.82, 0.05, f"{xlab}-leaning", transform=ax.transAxes, fontsize=9, color="#444444", ha="center")
    ax.text(0.18, 0.93, f"{ylab}-leaning", transform=ax.transAxes, fontsize=9, color="#444444", ha="center")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(f"signal strength = max {metric} across states (dark = near-zero reach)")
    ax.set_xlabel(f"{xlab}-state {metric} ({unit})")
    ax.set_ylabel(f"{ylab}-state {metric} ({unit})")
    ax.set_title(f"Crohn drug-mechanism targets: {xlab} vs {ylab} macrophage {metric}\n"
                 f"(position = {unit}; color = magnitude)")
    if raw:
        ax.set_xlim(-0.02 * hi, 1.05 * hi); ax.set_ylim(-0.02 * hi, 1.05 * hi)
    else:
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.grid(alpha=0.25)

    # default inflammatory-vs-resident keeps its legacy filename; other pairs are pair-named
    default_pair = (x_build == "macrophage_crohn_inflammatory" and y_build == "macrophage_crohn_resident")
    stem = f"state_{metric}{'_raw' if raw else ''}_scatter.png" if default_pair else \
           f"state_{xlab}_vs_{ylab}_{metric}{'_raw' if raw else ''}_scatter.png"
    # joint-influence scatter lives next to the joint table; others in the x-build's influence dir
    out_dir = (joint_path or DEFAULT_JOINT).parent / "images" \
        if metric == "joint_influence" else cfg.influence_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / stem
    fig.tight_layout(); fig.savefig(out, dpi=150)
    lean = df.inf - df.res
    print(f"points: {len(df)} | {xlab}-leaning: {(lean>0).sum()} | {ylab}-leaning: {(lean<0).sum()}")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="state-vs-state influence scatter (drug-mechanism targets)")
    ap.add_argument("--metric", default="influence",
                    choices=["influence", "specificity", "effect", "effect_propagated",
                             "embed_influence", "joint_influence"])
    ap.add_argument("--raw", action="store_true", help="axes = raw metric value, not percentile")
    ap.add_argument("--x-build", default="macrophage_crohn_inflammatory")
    ap.add_argument("--y-build", default="macrophage_crohn_resident")
    ap.add_argument("--joint-path", default=None, type=Path,
                    help="path to joint_influence.tsv (for --metric joint_influence)")
    a = ap.parse_args()
    raise SystemExit(main(a.metric, a.raw, a.x_build, a.y_build, a.joint_path))
