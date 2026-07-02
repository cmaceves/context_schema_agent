"""Plots for the OOF target-prediction results (reads predictions_per_instance.tsv from --out-dir):
  oof_prob_boxplot.png                 OOF prob by disease, hue = OpenTargets max clinical phase
  oof_disease_vs_healthy_scatter.png   OOF disease vs healthy-macrophage prob, targets ringed by phase

Phase coloring now includes PHASE 1 (the trainer's phase floor is 1). Non-target = no drug evidence.
Run: .venv/bin/python mlp_mods/de_ppi/scripts/test_mlp/plot_oof.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, tol_colors as tc

OT = Path("mlp_mods/03_opentargets_rebuild")
DISEASE_EFO = {"alz": "MONDO_0004975", "crohn": "EFO_0000384", "uc": "EFO_0000729", "ild": "EFO_0004244"}
ORDER = ["non-target (no drug evidence)", "phase 1", "phase 2", "phase 3", "phase 4"]


def phase_cats(df: pd.DataFrame) -> pd.DataFrame:
    # disease-level (collapsed) phase sets: union across cell types, matching the trainer's labeling
    def union(code, ph):
        d = json.load(open(OT / f"positive_proteins_{code}_new_phase{ph}.json"))[code]
        return set().union(*d.values()) if d else set()
    p = {ph: {dis: union(code, ph) for dis, code in DISEASE_EFO.items()} for ph in (2, 3, 4)}

    def cat(r):
        if r.label == 0:
            return ORDER[0]
        for ph, name in [(4, "phase 4"), (3, "phase 3"), (2, "phase 2")]:
            if r.protein in p[ph][r.disease]:
                return name
        return "phase 1"                       # label==1 but below phase 2 = phase 1
    df = df.copy(); df["phase_cat"] = df.apply(cat, axis=1)
    return df


def main(out_dir: str) -> int:
    OUT = Path(out_dir)
    sns.set_style("whitegrid"); c = tc.colorsets["bright"]   # blue red green yellow cyan purple grey
    pal = {ORDER[0]: c.grey, "phase 1": c.cyan, "phase 2": c.blue, "phase 3": c.yellow, "phase 4": c.red}
    df = phase_cats(pd.read_csv(OUT / "predictions_per_instance.tsv", sep="\t"))

    # --- boxplot: OOF prob by disease, hue = phase
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=df, x="disease", y="oof_prob", hue="phase_cat", order=sorted(df.disease.unique()),
                hue_order=ORDER, palette=pal, fliersize=2, linewidth=1, ax=ax)
    ax.set_xlabel("disease"); ax.set_ylabel("out-of-fold probability")
    ax.set_title("OOF target probability by disease and OpenTargets max clinical phase")
    ax.legend(title="OpenTargets label", loc="upper right", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.1)
    plt.tight_layout(); plt.savefig(OUT / "oof_prob_boxplot.png", dpi=150); plt.close()

    # --- scatter: disease vs healthy-macrophage prob (macrophage rows only), targets ringed by phase
    d = df[df.oof_healthy.notna()]
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, zorder=0)
    nt = d[d.phase_cat == ORDER[0]]
    ax.scatter(nt.oof_healthy, nt.oof_prob, s=14, alpha=.45, c=c.grey, edgecolor="none",
               label=f"{ORDER[0]} (n={len(nt)})", zorder=1)
    for l in ORDER[1:]:
        s = d[d.phase_cat == l]
        ax.scatter(s.oof_healthy, s.oof_prob, s=42, alpha=.95, c=pal[l], edgecolor="black",
                   linewidth=0.7, label=f"{l} (n={len(s)})", zorder=3)
    ax.set_xlabel("OOF healthy-macrophage probability"); ax.set_ylabel("OOF disease probability")
    ax.set_title("OOF disease vs healthy-macrophage probability (macrophage contexts)")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.set_aspect("equal")
    ax.text(.04, .96, "above line = disease-specific", fontsize=8, c="0.4", va="top")
    ax.legend(loc="lower right", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.1)
    plt.tight_layout(); plt.savefig(OUT / "oof_disease_vs_healthy_scatter.png", dpi=150); plt.close()

    print("wrote oof_prob_boxplot.png + oof_disease_vs_healthy_scatter.png to", OUT)
    print(df.phase_cat.value_counts().reindex(ORDER).to_string())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="mlp_mods/de_ppi/results/_tmp_mlp")
    a = ap.parse_args()
    raise SystemExit(main(a.out_dir))
