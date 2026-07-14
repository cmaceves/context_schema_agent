"""Context-factor ablation: retrain zeroing each factor one at a time (+ full + blind), report held-out
link-prediction AUC, and plot the training-loss / val-AUC curves colored per ablation. See SEQ_CONTEXT_EMBED.md.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/ablation_context.py [--epochs 60]
Out: images/ablation_curve.png , validation/ablation_<run>.tsv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "mlp_mods/seq_context/scripts")
from train_link_context import build_data, run_split, SEQ

ABLATIONS = [("full", None), ("no_cell_type", "cell_type"), ("no_disease", "disease"),
             ("no_tissue", "tissue"), ("no_state", "state"), ("blind (all)", "all")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v6_esmproj256")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = build_data(args.seed, neg_mode="hard", labels="cistarget")
    print(f"data: {len(d['tags'])} contexts | device={device}", flush=True)

    rows, hists = [], {}
    for name, z in ABLATIONS:
        _, m, hist = run_split(d, z, args.epochs, device, args.seed)
        rows.append({"ablation": name, "zeroed": z or "-", "auc": m["auc"], "ap": m["ap"]})
        hists[name] = hist
        print(f"  {name:14s} zeroed={str(z):10s} AUC={m['auc']:.3f} AP={m['ap']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    full_auc = float(df.loc[df.ablation == "full", "auc"].iloc[0])
    df["drop_vs_full"] = (full_auc - df.auc).round(3)
    print("\n=== context-factor ablation (held-out link-prediction AUC) ===", flush=True)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    df.to_csv(SEQ / "validation" / f"ablation_{args.tag}.tsv", sep="\t", index=False)

    sns.set_style("whitegrid")
    pal = dict(zip([a[0] for a in ABLATIONS], sns.color_palette("tab10", len(ABLATIONS))))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for name in hists:
        ep = hists[name]["epoch"]
        ax[0].plot(ep, hists[name]["train_loss"], color=pal[name], label=name)
        ax[1].plot(ep, hists[name]["val_auc"], color=pal[name], label=name)
    ax[0].set(xlabel="epoch", ylabel="train BCE loss", title="Training loss per ablation")
    ax[1].set(xlabel="epoch", ylabel="val link-pred AUC", title="Validation AUC per ablation")
    ax[1].axhline(0.5, color="grey", lw=0.8, ls=":")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle(f"Context-factor ablation ({args.tag}, ESM→{__import__('train_link_context').ESM_PROJ})", fontsize=12)
    fig.tight_layout()
    out = SEQ / "images" / "ablation_curve.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
