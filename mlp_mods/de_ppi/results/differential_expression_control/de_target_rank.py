"""DE-alone baseline for target prediction, and rank comparison vs the in-silico perturbation.

Question: does the perturbation ranking beat differential expression alone at putting OpenTargets targets near
the top? This builds the DE baseline and compares ranks head-to-head for the OT>0.5 targets in Crohn colon
macrophage (inflammatory).

(1) DE model: instance = (protein, disease network) over the scvi build's disease networks that have a matched
    healthy net (same tissue+cell type+state). Feature = |disease_expr − healthy_expr| (log1p CP10k, the DE
    magnitude). Label = 1 if that network's disease has OpenTargets score_indirect > 0.5 for the protein.
    Logistic regression (standardized feature, class-weighted). With one feature the model's rank order equals
    the DE-sorted order — the logistic is the extensible wrapper, not extra ranking signal.
(2) Rank proteins per network by predicted P(target).
(3) For Crohn colon macrophage inflammatory, take the OT>0.5 targets and plot their DE-model rank (x) vs their
    perturbation rank (y). images/de_vs_perturbation_ranks.png.

Run: .venv/bin/python mlp_mods/de_ppi/results/differential_expression_control/de_target_rank.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
try:
    from plot_style import apply_style
except Exception:
    def apply_style(): pass

BUILD = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi")
HERE = Path("mlp_mods/de_ppi/results/differential_expression_control")
OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
           "ild": "ild_target_association_EFO_0004244.tsv"}
EVAL_TAG = "crohn_colon_macrophage_inflammatory"
POS_THR = 0.5


def expr(tag):
    df = pd.read_csv(BUILD / "networks" / tag / "network_nodes.tsv", sep="\t", keep_default_na=False)
    return pd.Series(df.expression.astype(float).values, index=df.node_id.values)


def ot_map(disease):
    d = pd.read_csv(OT_DIR / OT_FILE[disease], sep="\t")
    return dict(zip(d.gene_symbol, d.score_indirect))


def de_feature(tag):
    """|disease − matched-healthy| expression for a disease network; None if no matched healthy."""
    arm, rest = tag.split("_", 1)
    htag = "healthy_" + rest
    if not (BUILD / "networks" / htag / "network_nodes.tsv").exists():
        return None, None
    d, h = expr(tag), expr(htag)
    common = d.index                                     # proteins present in the disease net
    de = (d.reindex(common).fillna(0.0) - h.reindex(common).fillna(0.0)).abs()
    return de, arm


def main():
    tags = sorted(p.name for p in (BUILD / "networks").iterdir() if (p / "network_nodes.tsv").exists())
    dis_tags = [t for t in tags if t.split("_")[0] in OT_FILE]
    ot = {dz: ot_map(dz) for dz in OT_FILE}

    # ---- (1) build training instances over disease networks with a matched healthy ----
    rows = []
    for t in dis_tags:
        de, arm = de_feature(t)
        if de is None:
            continue
        m = ot[arm]
        for g, v in de.items():
            rows.append((t, arm, g, float(v), int(m.get(g, 0) > POS_THR)))
    tr = pd.DataFrame(rows, columns=["tag", "disease", "protein", "de", "label"])
    print(f"train instances: {len(tr)} over {tr.tag.nunique()} disease nets | positives (OT>{POS_THR}): {tr.label.sum()}")

    sc = StandardScaler()
    X = sc.fit_transform(tr[["de"]].values)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, tr.label.values)
    print(f"logistic coef(DE)={clf.coef_[0,0]:+.3f}  (>0 => higher DE -> more target-like)")

    # ---- (2) rank proteins per network by predicted P(target) ----
    def de_rank(tag):
        de, _ = de_feature(tag)
        p = clf.predict_proba(sc.transform(de.values.reshape(-1, 1)))[:, 1]
        r = pd.DataFrame({"protein": de.index, "de": de.values, "p_target": p})
        r = r.sort_values("p_target", ascending=False).reset_index(drop=True)
        r["de_rank"] = np.arange(1, len(r) + 1)
        return r

    de_r = de_rank(EVAL_TAG)
    de_r.to_csv(HERE / "de_target_rank_crohn_colon_inflammatory.tsv", sep="\t", index=False)

    # ---- (3) compare to perturbation ranks for OT>0.5 targets ----
    pert = pd.read_csv(BUILD / "insilico_perturb" / f"{EVAL_TAG}_perturbation_results.tsv", sep="\t")
    pert = pert.sort_values("projection", ascending=False).reset_index(drop=True)
    pert["pert_rank"] = np.arange(1, len(pert) + 1)
    N = len(pert)

    m = ot["crohn"]
    pos = sorted([g for g in de_r.protein if m.get(g, 0) > POS_THR], key=lambda g: -m.get(g, 0))
    comp = pd.DataFrame({"protein": pos})
    comp["ot"] = comp.protein.map(m).round(3)
    comp["de_rank"] = comp.protein.map(dict(zip(de_r.protein, de_r.de_rank)))
    comp["pert_rank"] = comp.protein.map(dict(zip(pert.protein, pert.pert_rank)))
    comp = comp.sort_values("ot", ascending=False)
    comp.to_csv(HERE / "rank_comparison_OT0.5.tsv", sep="\t", index=False)
    print(f"\nOT>0.5 targets in {EVAL_TAG} (N={N} proteins ranked):")
    print(comp.to_string(index=False))
    from scipy.stats import spearmanr
    rho, p = spearmanr(comp.de_rank, comp.pert_rank)
    print(f"\nSpearman(de_rank, pert_rank) over the {len(comp)} targets = {rho:+.2f} (p={p:.2f})")
    print(f"median rank — DE: {comp.de_rank.median():.0f}  perturbation: {comp.pert_rank.median():.0f}  (of {N})")

    # ---- plot ----
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim = N + 30
    ax.plot([1, lim], [1, lim], ls="--", lw=1, color="0.6", zorder=0)
    ax.scatter(comp.de_rank, comp.pert_rank, s=70, color="#4C72B0", edgecolors="0.2", linewidths=0.6, zorder=3)
    for _, r in comp.iterrows():
        ax.annotate(f"{r.protein}", (r.de_rank, r.pert_rank), fontsize=8, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("rank by DE model  (1 = best)")
    ax.set_ylabel("rank by perturbation  (1 = best)")
    ax.set_title(f"OT>0.5 target ranks: DE model vs perturbation\n{EVAL_TAG} (N={N}); "
                 f"below diagonal = perturbation ranks it better", fontsize=9.5)
    fig.tight_layout()
    out = HERE / "images" / "de_vs_perturbation_ranks.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")
    print(f"wrote {HERE/'rank_comparison_OT0.5.tsv'}, {HERE/'de_target_rank_crohn_colon_inflammatory.tsv'}")


if __name__ == "__main__":
    main()
