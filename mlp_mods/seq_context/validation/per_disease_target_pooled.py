"""Per-disease pooled target recovery. For each disease, pool each protein's embedding over THAT disease's
contexts -> one vector/protein, train LogReg + MLP (balanced, 5-fold OOF) on THAT disease's drug targets, and
compare to ESM. Tests disease-level context-specificity (does the crohn-pooled embedding recover crohn targets,
and beat ESM, etc.). Boxplot per classifier (x=disease, target P, EMB-per-disease vs ESM) + MRR/hits table.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/per_disease_target_pooled.py --run link_v9
Out: images/per_disease_target_<run>_{logreg,mlp}.png ; results/<run>/per_disease_target_mrr.tsv
"""
from __future__ import annotations
import argparse
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
DRUGDIR = "mlp_mods/03_opentargets_rebuild"
# disease arm (context prefix) -> known_drugs file id
DISEASES = {"crohn": "EFO_0000384", "uc": "EFO_0000729", "ild": "EFO_0004244", "alz": "MONDO_0004975",
            "covid": "MONDO_0100096", "athero": "EFO_0003914", "hvd": "EFO_0009940"}
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())


def make_clf(kind):
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))


def rank_metrics(score, pos):
    ranked = sorted(score, key=lambda g: -score[g]); rank = {g: i + 1 for i, g in enumerate(ranked)}
    P = [g for g in ranked if g in pos]
    if not P:
        return 0, 0, float("nan")
    return sum(rank[g] <= 10 for g in P), sum(rank[g] <= 100 for g in P), float(np.mean([1 / rank[g] for g in P]))


def pooled(d, prefix):
    m = np.array([c.startswith(prefix + "_") for c in d["context"]])
    genes = [PROT[i] for i in d["prot_idx"][m]]
    acc = defaultdict(list)
    for g, v in zip(genes, d["emb"][m].astype(np.float64)):
        acc[g].append(v)
    return {g: np.mean(v, 0) for g, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v9")
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    arms = {c.split("_")[0] for c in d["context"]}

    table, box = [], []
    for arm, efo in DISEASES.items():
        if arm not in arms:
            print(f"skip {arm}: no contexts in {args.run}", flush=True); continue
        f = Path(DRUGDIR) / f"known_drugs_{efo}.tsv"
        if not f.exists():
            print(f"skip {arm}: no {f.name}", flush=True); continue
        pos = set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))
        gp = pooled(d, arm)
        genes = np.array(sorted(set(gp) & set(PROT)))
        tgt = sorted(pos & set(genes))
        if len(tgt) < args.min_targets:
            print(f"skip {arm}: only {len(tgt)} targets present (<{args.min_targets})", flush=True); continue
        feat = {"EMB": np.stack([gp[g] for g in genes]),
                "ESM": np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)}
        y = np.array([1 if g in pos else 0 for g in genes])
        controls = rng.choice(sorted(set(genes) - pos), size=min(args.n_control, len(genes) - len(tgt)), replace=False)
        for clf in ["logreg", "mlp"]:
            for featname, X in feat.items():
                P = cross_val_predict(make_clf(clf), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
                sc = dict(zip(genes, P))
                h10, h100, mrr = rank_metrics(sc, pos)
                table.append({"disease": arm, "efo": efo, "classifier": clf, "feature": featname,
                              "n_targets": len(tgt), "hits@10": h10, "hits@100": h100, "MRR": round(mrr, 4)})
                for g in tgt:
                    box.append({"disease": f"{arm}\n(n={len(tgt)})", "P": sc[g], "feature": featname, "classifier": clf})
        print(f"{arm}: {len(tgt)} targets, {len(genes)} proteins pooled over {sum(c.startswith(arm+'_') for c in d['context'])} contexts", flush=True)

    tab = pd.DataFrame(table)
    out = SEQ / "results" / args.run / "per_disease_target_mrr.tsv"
    tab.to_csv(out, sep="\t", index=False)
    print("\n" + tab.to_string(index=False) + f"\nwrote {out}", flush=True)

    bx = pd.DataFrame(box)
    sns.set_style("whitegrid")
    for clf in ["logreg", "mlp"]:
        sub = bx[bx.classifier == clf]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * sub.disease.nunique()), 5))
        sns.boxplot(data=sub, x="disease", y="P", hue="feature", hue_order=["EMB", "ESM"],
                    palette={"EMB": "#1b9e77", "ESM": "#d95f02"}, showfliers=False, ax=ax)
        sns.stripplot(data=sub, x="disease", y="P", hue="feature", hue_order=["EMB", "ESM"],
                      dodge=True, color="black", size=3, alpha=0.5, ax=ax, legend=False)
        ax.set(xlabel="", ylabel="P(disease drug target)  [out-of-fold]",
               title=f"Per-disease pooled target recovery — EMB(disease-pooled) vs ESM\n{args.run} · {clf.upper()}, 5-fold CV")
        ax.legend(title="", loc="upper right")
        fig.tight_layout()
        p = SEQ / "images" / f"per_disease_target_{args.run}_{clf}.png"
        fig.savefig(p, dpi=130); print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
