"""Combined ESM + pooled-disease-embedding -> drug-target prediction.

Three feature conditions, same all-OT drug-target labels (union across known_drugs_*.tsv), global
mean-pool over all contexts, out-of-fold P (5-fold), per classifier:
    ESM            : raw sequence vector (1280-d)
    EMB (pooled)   : protein mean-pooled over all contexts (128-d)
    ESM + pooled   : concatenation (1408-d)  <- the new combined condition
Writes a boxplot per classifier and one MRR/hits table across both classifiers.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/combined_esm_pooled_target.py --run link_v7_4ct
Out: images/esm_plus_pooled_target_{logreg,mlp}.png ; results/<run>/esm_plus_pooled_mrr.tsv
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
CONDS = [("ESM", "esm"), ("EMB (pooled)", "emb"), ("ESM + pooled", "concat")]
PAL = {"ESM": "#d95f02", "EMB (pooled)": "#1b9e77", "ESM + pooled": "#7570b3"}


def make_clf(kind):
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))


def pool_all(d):
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


def boxplot(df, order, hues, scores, drug_genes, run, clf, n_ctx):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="group", y="P", hue="condition", order=order, hue_order=hues,
                palette=PAL, showfliers=False, ax=ax)
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
           title=f"ESM vs pooled-embedding vs ESM+pooled -> drug-target prediction (all OT diseases)\n"
                 f"pool over {n_ctx} contexts · {run} · {clf.upper()}, 5-fold CV")
    ax.legend(title="", loc="upper left", framealpha=0.9)
    mlines = ["ranking over all proteins (positives = all drug targets):"]
    for name, _ in CONDS:
        h10, h100, mrr = rank_metrics(scores[name], drug_genes)
        mlines.append(f"{name:14s} H@10={h10:2d} H@100={h100:3d} MRR={mrr:.3f}")
    ax.text(0.985, 0.985, "\n".join(mlines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))
    fig.tight_layout()
    out = SEQ / "images" / f"esm_plus_pooled_target_{run}_{clf}.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v7_4ct")
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)

    phase = {}
    for f in DRUGFILES:
        for g, p in pd.read_csv(f, sep="\t").groupby("gene_symbol").phase.max().items():
            phase[g] = max(phase.get(g, 0), p)
    drug_genes = set(phase)
    gpool, n_ctx = pool_all(d)
    genes = np.array(sorted(set(gpool) & set(PROT)))
    print(f"{len(drug_genes)} drug-target genes | global pool over {n_ctx} contexts | proteins={len(genes)}", flush=True)

    esm = np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
    emb = np.stack([gpool[g] for g in genes])
    feat = {"esm": esm, "emb": emb, "concat": np.hstack([esm, emb])}
    y = np.array([1 if g in drug_genes else 0 for g in genes])
    targets = sorted(drug_genes & set(genes))
    controls = rng.choice(sorted(set(genes) - drug_genes), size=args.n_control, replace=False)

    table = []
    for clf in ["logreg", "mlp"]:
        scores = {}
        for name, key in CONDS:
            P = cross_val_predict(make_clf(clf), feat[key], y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            scores[name] = dict(zip(genes, P))
        rows = []
        for name, _ in CONDS:
            for g in targets:
                rows.append({"group": f"Phase {int(round(phase[g]))}", "P": scores[name][g], "condition": name, "gene": g})
            for g in controls:
                rows.append({"group": f"Control (n={args.n_control})", "P": scores[name][g], "condition": name, "gene": g})
        df = pd.DataFrame(rows)
        order = sorted([x for x in df.group.unique() if x.startswith("Phase")]) + \
            [x for x in df.group.unique() if x.startswith("Control")]
        hues = [c[0] for c in CONDS]
        boxplot(df, order, hues, scores, drug_genes, args.run, clf, n_ctx)
        for name, _ in CONDS:
            h10, h100, mrr = rank_metrics(scores[name], drug_genes)
            table.append({"classifier": clf, "condition": name, "n_pos": len(targets),
                          "hits@10": h10, "hits@100": h100, "MRR": round(mrr, 4)})

    tab = pd.DataFrame(table)
    out = SEQ / "results" / args.run / "esm_plus_pooled_mrr.tsv"
    tab.to_csv(out, sep="\t", index=False)
    print("\n" + tab.to_string(index=False), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
