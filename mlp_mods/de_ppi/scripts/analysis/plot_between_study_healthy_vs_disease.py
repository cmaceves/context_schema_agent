"""Scatter: between-study agreement for HEALTHY (control i) vs DISEASE (control b) arms.

Both controls hold cell type, disease, tissue, and cell state constant and vary STUDY, so each point is a
same-context pair of independent studies. x = mean per-protein cosine similarity, y = % overlapped proteins.
Reads controls/control_pairs.tsv (compare_controls output) for the given build.

Output: images/between_study_healthy_vs_disease.png
Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/plot_between_study_healthy_vs_disease.py --main-name <build>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARM = {"b": ("disease (control b)", "#d62728"), "i": ("healthy (control i)", "#1f77b4")}


def main(main_name) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    df = pd.read_csv(res / "controls" / "control_pairs.tsv", sep="\t")
    df = df[df.control.isin(["b", "i"])].copy()
    df["pct_overlap"] = df.jaccard * 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    for ctrl, (label, color) in ARM.items():
        g = df[df.control == ctrl]
        ax.scatter(g.average_cosine_similarity, g.pct_overlap, s=70, alpha=0.8,
                   edgecolor="k", linewidth=0.4, color=color, label=f"{label}, n={len(g)}")
    ax.set_xlabel("mean per-protein cosine similarity (between studies)")
    ax.set_ylabel("% overlapped proteins (Jaccard)")
    ax.set_title("Between-study agreement: healthy vs disease arms\n"
                 "(cell type, disease, tissue, cell state held constant; vary study)")
    ax.legend(title="arm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = res / "images" / "between_study_healthy_vs_disease.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img, dpi=150)
    print(f"wrote {img}\n")
    print(df.groupby("control_name").agg(n=("jaccard", "size"),
          mean_cos=("average_cosine_similarity", "mean"),
          pct_overlap=("pct_overlap", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_noselflin_detectfrac")
    raise SystemExit(main(ap.parse_args().main_name))
