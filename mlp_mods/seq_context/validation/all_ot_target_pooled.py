"""Global protein mean-pool -> drug-target prediction, trained on ALL our OT drug-target labels.

Pipeline:
    protein across ALL contexts  ->  mean pooling  ->  one protein embedding  ->  target prediction
Each protein's embedding is averaged over every context (all diseases x tissues x cell types x states)
to give a single context-agnostic vector. Positives = union of drug targets across all known_drugs_*.tsv
(Crohn/UC/IBD/EFO_0004244/Alzheimer); phase = max clinical phase across all. Compared to raw ESM.

Replaces images/crohn_phase_boxplot_pooled.png (per request).
Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/all_ot_target_pooled.py --run link_v7_4ct
"""
from __future__ import annotations
import argparse, glob
from collections import defaultdict
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
DRUGFILES = sorted(glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"))
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())


def make_clf(kind):
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))


def pool_all(d):
    """mean-pool each protein's embedding over ALL contexts."""
    genes = [PROT[i] for i in d["prot_idx"]]
    E = d["emb"].astype(np.float64)
    acc = defaultdict(list)
    for g, v in zip(genes, E):
        acc[g].append(v)
    return {g: np.mean(vs, 0) for g, vs in acc.items()}, len(set(d["context"]))


def rank_metrics(score_dict, pos):
    ranked = sorted(score_dict, key=lambda g: -score_dict[g])
    rank = {g: i + 1 for i, g in enumerate(ranked)}
    P = [g for g in ranked if g in pos]
    if not P:
        return 0, 0, float("nan")
    return (sum(rank[g] <= 10 for g in P), sum(rank[g] <= 100 for g in P),
            float(np.mean([1 / rank[g] for g in P])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v7_4ct")
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)

    phase = {}
    for f in DRUGFILES:
        for g, p in pd.read_csv(f, sep="\t").groupby("gene_symbol").phase.max().items():
            phase[g] = max(phase.get(g, 0), p)
    drug_genes = set(phase)
    print(f"drug-target labels: {len(drug_genes)} genes from {len(DRUGFILES)} disease files", flush=True)

    gpool, n_ctx = pool_all(d)
    genes = np.array(sorted(set(gpool) & set(PROT)))
    print(f"global mean-pool over {n_ctx} contexts; proteins={len(genes)}", flush=True)

    conds = [("EMB (all-context pooled)", "emb"), ("ESM", "esm")]
    feat = {"emb": np.stack([gpool[g] for g in genes]),
            "esm": np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)}
    y = np.array([1 if g in drug_genes else 0 for g in genes])
    scores = {}
    for name, key in conds:
        P = cross_val_predict(make_clf(args.classifier), feat[key], y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        scores[name] = dict(zip(genes, P))

    targets = sorted(drug_genes & set(genes))
    controls = rng.choice(sorted(set(genes) - drug_genes), size=args.n_control, replace=False)
    rows = []
    for name, _ in conds:
        for g in targets:
            rows.append({"group": f"Phase {int(round(phase[g]))}", "P": scores[name][g], "condition": name, "gene": g})
        for g in controls:
            rows.append({"group": f"Control (n={args.n_control})", "P": scores[name][g], "condition": name, "gene": g})
    df = pd.DataFrame(rows)
    order = sorted([x for x in df.group.unique() if x.startswith("Phase")]) + \
        [x for x in df.group.unique() if x.startswith("Control")]
    hues = [c[0] for c in conds]
    print(df.groupby(["condition", "group"]).P.median().unstack().to_string(), flush=True)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    pal = {"EMB (all-context pooled)": "#1b9e77", "ESM": "#d95f02"}
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
    n_per = {g: df[(df.group == g) & (df.condition == hues[0])].gene.nunique() for g in order}
    ax.set_xticklabels([f"{g.split(' (')[0]}\n(n={n_per[g]})" for g in order])
    ax.set(xlabel="", ylabel="P(drug target)  [out-of-fold]",
           title=f"Global protein mean-pool -> drug-target prediction (all OT diseases)\n"
                 f"pool over {n_ctx} contexts vs ESM · {args.run} · {args.classifier.upper()}, 5-fold CV")
    ax.legend(title="", loc="upper left", framealpha=0.9)
    mlines = ["ranking over all proteins (positives = all drug targets):"]
    for name, _ in conds:
        h10, h100, mrr = rank_metrics(scores[name], drug_genes)
        mlines.append(f"{name}:  H@10={h10}  H@100={h100}  MRR={mrr:.3f}")
    print("\n".join(mlines), flush=True)
    ax.text(0.985, 0.985, "\n".join(mlines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))
    fig.tight_layout()
    (SEQ / "images").mkdir(exist_ok=True)
    suffix = "_mlp" if args.classifier == "mlp" else ""
    out = SEQ / "images" / f"crohn_phase_boxplot_pooled_{args.run}{suffix}.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
