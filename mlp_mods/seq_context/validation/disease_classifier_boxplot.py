"""Disease-colored version of the pooled classifier boxplot. Reuses the per-protein probabilities already in
<tag>_classifier_table.tsv (EMB pooled classifier). For each disease, shows P(drug target) [EMB] over that
disease's phase-3/4 OT targets, one box per disease colored by disease, + a Control box. LogReg (top) and
MLP (bottom) panels stacked. No retraining.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/disease_classifier_boxplot.py --run link_v9
Out: images/<tag>_disease_classifier_boxplot.png
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEQ = Path("mlp_mods/seq_context")
DISNAME = {"EFO_0000384": "crohns_disease", "EFO_0000729": "ulcerative_colitis", "EFO_0003767": "ibd",
           "EFO_0004244": "bronchiolitis_obliterans", "MONDO_0004975": "alzheimer", "MONDO_0100096": "covid19",
           "EFO_0003914": "atherosclerosis", "EFO_0009940": "heart_valve_disease", "MONDO_0004985": "bipolar_disorder"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v10")
    ap.add_argument("--n-control", type=int, default=30)
    ap.add_argument("--min-targets", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tag = args.run.replace("link_", "")
    tab = pd.read_csv(SEQ / "results" / args.run / f"{tag}_classifier_table.tsv", sep="\t")

    # max clinical phase per gene (OT files are already phase>=3, so phase 3 or 4)
    phase = {}
    for f in sorted(glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv")):
        for g, p in pd.read_csv(f, sep="\t").groupby("gene_symbol").phase.max().items():
            phase[str(g)] = max(phase.get(str(g), 0), p)
    tab["phase34"] = tab.protein.map(lambda g: int(round(phase.get(g, 0))) in (3, 4))

    diseases = [d for d in DISNAME.values() if d in tab.columns]
    diseases = sorted(diseases, key=lambda d: -int(tab[d].sum()))
    nonpos = tab[(tab[diseases].sum(axis=1) == 0)]
    controls = nonpos.sample(min(args.n_control, len(nonpos)), random_state=args.seed)

    sns.set_style("whitegrid")
    pal = dict(zip(diseases, sns.color_palette("tab10", len(diseases))))
    pal["Control"] = "#999999"
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * (len(diseases) + 1)), 5.5))
    col = "prob_model_mlp"                                 # MLP EMB pooled classifier probability (LogReg dropped)
    rows, order, counts = [], [], {}
    for d in diseases:
        sub = tab[(tab[d] == 1) & tab.phase34]
        if len(sub) < args.min_targets:
            continue
        order.append(d); counts[d] = len(sub)
        for p in sub[col]:
            rows.append({"group": d, "P": p})
    for p in controls[col]:
        rows.append({"group": "Control", "P": p})
    order2 = order + ["Control"]; counts["Control"] = len(controls)
    df = pd.DataFrame(rows)
    sns.boxplot(data=df, x="group", y="P", order=order2, palette=pal, showfliers=False, ax=ax)
    sns.stripplot(data=df, x="group", y="P", order=order2, color="black", size=3, alpha=0.5, ax=ax)
    ax.set_xticklabels([f"{g}\n(n={counts[g]})" for g in order2], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="P(drug target) [MLP EMB, OOF]", xlabel="",
           title=f"Per-disease target probability (MLP EMB pooled) · {args.run} · phase 3-4 targets, colored by disease")
    fig.tight_layout()
    out = SEQ / "images" / f"{tag}_disease_classifier_boxplot.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
