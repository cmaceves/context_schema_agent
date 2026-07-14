"""Boxplot of P(Crohn target) by drug phase vs random controls, across three conditions (color):
  EMB (Crohn)   = context embedding in crohn_colon_inflammatory
  EMB (healthy) = context embedding in the matched healthy context (healthy_colon_inflammatory)
  ESM (Crohn)   = raw sequence, crohn context (context-invariant reference)
Same Crohn drug-target labels + same control genes across conditions. Out-of-fold P (5-fold cross_val_predict),
L2 LogReg (balanced). See SEQ_CONTEXT_EMBED.md.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/crohn_phase_boxplot.py
Out: images/crohn_phase_boxplot.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

SEQ = Path("mlp_mods/seq_context")
DRUGFILE = "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0000384.tsv"
OTFILE = "mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv"
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())
CROHN = "crohn_colon_macrophage_inflammatory"
HEALTHY = "healthy_colon_macrophage_inflammatory"


def make_clf(kind):
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        # small + strongly regularized given tiny positive count (no class_weight in MLPClassifier)
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))


def rank_metrics(score_dict, pos):
    """Rank all proteins in the context by P; positives = drug targets. hits@k over the full ranking,
    MRR = mean reciprocal rank over all positives (1/rank averaged)."""
    ranked = sorted(score_dict, key=lambda g: -score_dict[g])
    rank = {g: i + 1 for i, g in enumerate(ranked)}
    P = [g for g in ranked if g in pos]
    if not P:
        return 0, 0, float("nan"), 0
    h10 = sum(rank[g] <= 10 for g in P)
    h100 = sum(rank[g] <= 100 for g in P)
    mrr = float(np.mean([1 / rank[g] for g in P]))
    return h10, h100, mrr, len(P)


def score(d, context, feature, pos, clf_kind):
    m = d["context"] == context
    genes = np.array([PROT[i] for i in d["prot_idx"][m]])
    X = d["emb"][m].astype(np.float64) if feature == "EMB" else \
        np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
    y = np.array([1 if g in pos else 0 for g in genes])
    P = cross_val_predict(make_clf(clf_kind), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return dict(zip(genes, P)), set(genes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v4_cistarget")
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    kd = pd.read_csv(DRUGFILE, sep="\t")
    phase = kd.groupby("gene_symbol").phase.max()
    drug_genes = set(phase.index)

    conds = [("EMB (Crohn)", CROHN, "EMB"), ("EMB (healthy)", HEALTHY, "EMB"), ("ESM (Crohn)", CROHN, "ESM")]
    scores, present = {}, {}
    for name, ctx, feat in conds:
        scores[name], present[name] = score(d, ctx, feat, drug_genes, args.classifier)

    # full per-protein probabilities across every condition/context (superset of the plotted subset)
    ot = pd.read_csv(OTFILE, sep="\t").set_index("gene_symbol").score_indirect.to_dict()
    allrows = [{"gene": g, "condition": name, "context": ctx, "feature": feat, "P": p,
                "ot_score": ot.get(g, np.nan),
                "drug_target": g in drug_genes,
                "phase": int(round(phase[g])) if g in drug_genes else np.nan}
               for name, ctx, feat in conds for g, p in scores[name].items()]
    csv = SEQ / "results" / args.run / f"crohn_phase_scores{'_mlp' if args.classifier == 'mlp' else ''}.csv"
    pd.DataFrame(allrows).sort_values(["condition", "P"], ascending=[True, False]).to_csv(csv, index=False)
    print(f"wrote {csv}  ({len(allrows)} rows)", flush=True)

    common = set.intersection(*present.values())                 # genes present in all conditions
    targets = sorted(drug_genes & common)
    controls = rng.choice(sorted(common - drug_genes), size=args.n_control, replace=False)

    rows = []
    for name, _, _ in conds:
        for g in targets:
            rows.append({"group": f"Phase {int(round(phase[g]))}", "P": scores[name][g], "condition": name})
        for g in controls:
            rows.append({"group": f"Control (n={args.n_control})", "P": scores[name][g], "condition": name})
    df = pd.DataFrame(rows)
    order = sorted([x for x in df.group.unique() if x.startswith("Phase")]) + \
        [x for x in df.group.unique() if x.startswith("Control")]
    hues = [c[0] for c in conds]
    print(df.groupby(["condition", "group"]).P.median().unstack().to_string(), flush=True)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    pal = {"EMB (Crohn)": "#1b9e77", "EMB (healthy)": "#a6d8c5", "ESM (Crohn)": "#d95f02"}
    sns.boxplot(data=df, x="group", y="P", hue="condition", order=order, hue_order=hues,
                palette=pal, showfliers=False, ax=ax)
    sns.stripplot(data=df, x="group", y="P", hue="condition", order=order, hue_order=hues,
                  dodge=True, color="black", size=3, alpha=0.5, ax=ax, legend=False)
    k = len(hues); w = 0.8 / k
    for i, g in enumerate(order):
        for j, h in enumerate(hues):
            sub = df[(df.group == g) & (df.condition == h)]
            if len(sub):
                x = i + (j - (k - 1) / 2) * w
                ax.text(x, sub.P.quantile(0.75) + 0.015 * df.P.max(), f"{sub.P.median():.3f}",
                        ha="center", va="bottom", fontsize=7, color="#333333")
    n_per = {g: df[(df.group == g) & (df.condition == hues[0])].gene.nunique() if "gene" in df
             else len(df[(df.group == g) & (df.condition == hues[0])]) for g in order}
    ax.set_xticklabels([f"{g.split(' (')[0]}\n(n={n_per[g]})" for g in order])
    ax.set(xlabel="", ylabel="P(Crohn target)  [out-of-fold]",
           title=f"Crohn drug targets by clinical phase — EMB(Crohn) vs EMB(healthy) vs ESM\n"
                 f"{args.run} · {args.classifier.upper()} classifier, 5-fold CV")
    ax.legend(title="", loc="upper left", framealpha=0.9)
    mlines = ["ranking over all proteins (positives = drug targets):"]
    for name, _, _ in conds:
        h10, h100, mrr, npos = rank_metrics(scores[name], drug_genes)
        mlines.append(f"{name}:  H@10={h10}  H@100={h100}  MRR={mrr:.3f}")
    print("\n".join(mlines), flush=True)
    ax.text(0.985, 0.985, "\n".join(mlines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))
    fig.tight_layout()
    (SEQ / "images").mkdir(exist_ok=True)
    suffix = "_mlp" if args.classifier == "mlp" else ""
    out = SEQ / "images" / f"crohn_phase_boxplot{suffix}.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
