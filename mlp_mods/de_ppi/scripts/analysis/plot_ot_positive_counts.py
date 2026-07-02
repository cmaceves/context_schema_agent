"""Count OpenTargets-associated proteins present in each Crohn / UC disease network, at two score thresholds.

For every disease-arm MAIN network (arm in {crohn, uc}; one per tissue × cell type × cell state), count how many
of its protein nodes are OT-associated with THAT disease at score_indirect > 0.1 and > 0.5 (hue). Bars grouped
per network; a network's OT label uses the matching disease file (Crohn=EFO_0000384, UC=EFO_0000729).

Outputs (results/<main>/images/):
  ot_positive_counts_by_network.png   grouped barplot (network × threshold)
  ot_positive_counts_by_network.tsv   the underlying counts (+ n_proteins per network)

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/plot_ot_positive_counts.py \
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
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv"}
THRESHOLDS = [(">0.1", 0.1), (">0.5", 0.5)]


def main(main_name, arms) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    net_dir = res / "networks"
    ot_score = {}
    for arm in arms:
        ot = pd.read_csv(OT_DIR / OT_FILE[arm], sep="\t")
        ot_score[arm] = dict(zip(ot.gene_symbol, ot.score_indirect))

    rows = []
    for nd in sorted(net_dir.iterdir()):
        arm = nd.name.split("_")[0]
        if arm not in arms or not (nd / "network_nodes.tsv").exists():
            continue
        n = pd.read_csv(nd / "network_nodes.tsv", sep="\t", keep_default_na=False)
        genes = n.loc[n["node_type"] == "protein", "node_id"]
        s = genes.map(ot_score[arm]).fillna(0.0).to_numpy()
        for lbl, thr in THRESHOLDS:
            rows.append(dict(network=nd.name, arm=arm, n_proteins=len(genes),
                             threshold=lbl, count=int((s > thr).sum())))
    df = pd.DataFrame(rows).sort_values(["arm", "network"])
    out = res / "images"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ot_positive_counts_by_network.tsv", sep="\t", index=False)

    apply_style()
    nets = df["network"].drop_duplicates().tolist()
    x = np.arange(len(nets)); w = 0.4
    colors = {">0.1": TOL["blue"], ">0.5": TOL["red"]}
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(nets)), 4.8))
    for i, (lbl, _) in enumerate(THRESHOLDS):
        sub = df[df.threshold == lbl].set_index("network").reindex(nets)
        bars = ax.bar(x + (i - 0.5) * w, sub["count"].values, w, label=f"OT score {lbl}", color=colors[lbl])
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(nets, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("OT-associated proteins in network")
    ax.set_title(f"OpenTargets-associated proteins per Crohn/UC network (score_indirect thresholds)  "
                 f"[{main_name.split('_')[-1]}]", fontsize=10)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out / "ot_positive_counts_by_network.png")
    print(f"wrote {out/'ot_positive_counts_by_network.png'}")
    print(df.pivot_table(index="network", columns="threshold", values="count").to_string())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--arms", default="crohn,uc")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, [x for x in a.arms.split(",") if x]))
