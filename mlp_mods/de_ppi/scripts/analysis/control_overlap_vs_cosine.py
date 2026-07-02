"""Node-overlap vs mean_prot_cos for the control pairs in control_comparison_meanprotcos_summary.tsv.

Question: for each control pair (donor_split / between_study / pool_vs_pool), how many protein nodes do
the two networks share, and does that overlap track the agreement metric (mean_prot_cos)? If agreement is
just "the two networks contain the same proteins," overlap and cosine will be tightly correlated.

For each pair we read the published per-pair table (results/<out_name>/tables/control_comparison_meanprotcos.tsv,
written by dump_control_comparison_meanprotcos.py) so mean_prot_cos is IDENTICAL to the summary, and we
compute node overlap fresh from the `present` mask in embeddings.npz (protein nodes only):

  intersection = |present_a & present_b|
  union        = |present_a | present_b|
  jaccard      = intersection / union
  min_frac     = intersection / min(|a|, |b|)        # overlap relative to the smaller network

Then correlate overlap (jaccard, intersection, min_frac) with mean_prot_cos, overall and per control_type
(Pearson + Spearman; Spearman = Pearson on ranks, no scipy dependency).

Output (results/<out_name>/tables/ and images/):
  control_overlap_vs_cosine.tsv   one row per control pair (overlap columns + mean_prot_cos)
  control_overlap_vs_cosine.png   scatter, colored by control_type

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/control_overlap_vs_cosine.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")


def _corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Return (pearson, spearman) between two series; nan if <3 points or zero variance."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(x, y)[0, 1])
    spear = float(np.corrcoef(x.rank(), y.rank())[0, 1])
    return round(pear, 3), round(spear, 3)


def main(out_name: str) -> int:
    res = DE_PPI / "results" / out_name
    pair_tsv = res / "tables" / "control_comparison_meanprotcos.tsv"
    if not pair_tsv.exists():
        raise SystemExit(f"missing {pair_tsv} -- run dump_control_comparison_meanprotcos.py first")
    pairs = pd.read_csv(pair_tsv, sep="\t")

    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags = list(d["tags"])
    present = d["present"]
    is_prot = d["node_type"] == "protein"
    ti = {t: i for i, t in enumerate(tags)}
    # protein-node presence set per network tag
    pres_prot = {t: set(np.where(present[ti[t]] & is_prot)[0]) for t in tags}

    rows = []
    for r in pairs.itertuples(index=False):
        a, b = r.network_a, r.network_b
        sa, sb = pres_prot.get(a), pres_prot.get(b)
        if sa is None or sb is None:
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        smaller = min(len(sa), len(sb))
        rows.append({
            "disease": r.disease, "cell_type": r.cell_type, "control_type": r.control_type,
            "state": r.state, "network_a": a, "network_b": b,
            "n_a": len(sa), "n_b": len(sb),
            "intersection": inter, "union": union,
            "jaccard": round(inter / union, 4) if union else float("nan"),
            "min_frac": round(inter / smaller, 4) if smaller else float("nan"),
            "mean_prot_cos": r.mean_prot_cos,
        })

    df = pd.DataFrame(rows)
    out = res / "tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "control_overlap_vs_cosine.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_overlap_vs_cosine.tsv'} ({len(df)} pairs)\n")
    print(df.to_string(index=False))

    # ---- correlations: overlap vs mean_prot_cos ----
    print("\n=== correlation of overlap with mean_prot_cos (pearson / spearman) ===")
    corr_rows = []
    for measure in ("jaccard", "intersection", "min_frac"):
        for label, sub in [("ALL", df)] + [(f"  {ct}", g) for ct, g in df.groupby("control_type")]:
            p, s = _corr(sub[measure], sub["mean_prot_cos"])
            corr_rows.append({"measure": measure, "group": label.strip(), "n": len(sub),
                              "pearson": p, "spearman": s})
            print(f"{measure:>12}  {label:<16} n={len(sub):<3} pearson={p}  spearman={s}")
    pd.DataFrame(corr_rows).to_csv(out / "control_overlap_vs_cosine_corr.tsv", sep="\t", index=False)
    print(f"\nwrote {out/'control_overlap_vs_cosine_corr.tsv'}")

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ct, g in df.groupby("control_type"):
        ax.scatter(g["jaccard"], g["mean_prot_cos"], label=ct, s=60, alpha=0.8, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("node overlap (Jaccard of present protein nodes)")
    ax.set_ylabel("mean_prot_cos")
    p_all, s_all = _corr(df["jaccard"], df["mean_prot_cos"])
    ax.set_title(f"Control-pair node overlap vs agreement\nall pairs: pearson={p_all}, spearman={s_all}")
    ax.legend(title="control_type", fontsize=8)
    fig.tight_layout()
    img = res / "images" / "control_overlap_vs_cosine.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img, dpi=150)
    print(f"wrote {img}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
