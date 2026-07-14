"""Pooled all-OT-target classifier — PER-DISEASE pooling + MAX aggregation (no longer a global pool across diseases).

For each disease arm D present in the build (crohn/uc/ild/alz/hvd/covid/athero; healthy excluded):
  - features: protein mean-pooled over D's contexts (EMB) / ESM as-is
  - labels: union of all-OT drug targets (binary); LogReg + MLP, 5-fold out-of-fold P
Each protein's final score = MAX predicted P across the diseases it appears in -> one combined ranking.
Boxplot: 2 stacked panels (LogReg top, MLP bottom); x={Phase 3, Phase 4, Control}, hue={EMB, ESM}, y=combined P.
H@10/H@100/MRR (drug-target recovery on the combined ranking) reported per method in the upper corner of each panel.
Table: protein, context, prob_model/esm x logreg/mlp (max-agg), 8 per-disease 0/1 columns.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/v10_classifier_boxplot.py --run link_v10
Out: images/<tag>_classifier_boxplot.png ; results/<run>/<tag>_classifier_table.tsv
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

SEQ = Path("mlp_mods/seq_context")
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())
DISNAME = {"EFO_0000384": "crohns_disease", "EFO_0000729": "ulcerative_colitis", "EFO_0003767": "ibd",
           "EFO_0004244": "bronchiolitis_obliterans", "MONDO_0004975": "alzheimer", "MONDO_0100096": "covid19",
           "EFO_0003914": "atherosclerosis", "EFO_0009940": "heart_valve_disease", "MONDO_0004985": "bipolar_disorder"}


def make_clf(kind):
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))


def pool_disease(d, arm):
    m = np.array([c.startswith(arm + "_") for c in d["context"]])
    acc = defaultdict(list); E = d["emb"].astype(np.float64); idx = d["prot_idx"]
    for j in np.where(m)[0]:
        acc[PROT[idx[j]]].append(E[j])
    return {g: np.mean(v, 0) for g, v in acc.items()}


def rank_metrics(score, pos):
    genes = np.array(list(score)); sc = np.array([score[g] for g in genes])
    ranked = genes[np.argsort(-sc)]; rk = {g: i + 1 for i, g in enumerate(ranked)}
    P = [g for g in genes if g in pos]
    if not P:
        return 0, 0, float("nan"), {}
    return (sum(rk[g] <= 10 for g in P), sum(rk[g] <= 100 for g in P),
            float(np.mean([1 / rk[g] for g in P])), {g: 1.0 / rk[g] for g in genes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v10")
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tag = args.run.replace("link_", "")

    disease_targets, phase = {}, {}
    for f in sorted(glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv")):
        fid = Path(f).stem.replace("known_drugs_", ""); name = DISNAME.get(fid, fid)
        df = pd.read_csv(f, sep="\t"); disease_targets[name] = set(df.gene_symbol.astype(str))
        for g, p in df.groupby("gene_symbol").phase.max().items():
            phase[str(g)] = max(phase.get(str(g), 0), p)
    pos = set(phase); diseases = sorted(disease_targets)

    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    arms = sorted({c.split("_")[0] for c in d["context"]} - {"healthy"})
    print(f"{args.run}: disease arms pooled = {arms} | all-OT positives={len(pos)}", flush=True)

    # per (clf, feat): per-disease OOF P, then MAX across diseases
    combined = {}
    for clf in ["mlp"]:                                   # LogReg dropped — MLP is the working readout
        for feat in ["EMB", "ESM"]:
            cmax = {}
            for arm in arms:
                gp = pool_disease(d, arm)
                genes = np.array(sorted(set(gp) & set(PROT)))
                if len(genes) < 50:
                    continue
                X = np.stack([gp[g] for g in genes]) if feat == "EMB" else \
                    np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
                y = np.array([1 if g in pos else 0 for g in genes])
                if y.sum() < 3:
                    continue
                P = cross_val_predict(make_clf(clf), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
                for g, p in zip(genes, P):
                    cmax[g] = max(cmax.get(g, -1.0), float(p))
            combined[(clf, feat)] = cmax
            h10, h100, mrr, _ = rank_metrics(cmax, pos)
            print(f"  {clf:6s} {feat}: H@10={h10} H@100={h100} MRR={mrr:.4f} (over {len(cmax)} proteins)", flush=True)

    # ---- table (max-agg probabilities + per-disease 0/1)
    allg = sorted(set().union(*[set(v) for v in combined.values()]))
    rows = []
    for g in allg:
        r = {"protein": g, "context": "per_disease_max",
             "prob_model_mlp": round(combined[("mlp", "EMB")].get(g, np.nan), 5) if g in combined[("mlp", "EMB")] else np.nan,
             "prob_esm_mlp": round(combined[("mlp", "ESM")].get(g, np.nan), 5) if g in combined[("mlp", "ESM")] else np.nan}
        for dis in diseases:
            r[dis] = int(g in disease_targets[dis])
        rows.append(r)
    pd.DataFrame(rows).sort_values("prob_model_mlp", ascending=False).to_csv(
        SEQ / "results" / args.run / f"{tag}_classifier_table.tsv", sep="\t", index=False)

    # ---- boxplot: 2 panels (logreg, mlp); x=Phase3/4/Control, hue=EMB/ESM, y=combined P; corner H@/MRR box
    targets = sorted(g for g in pos if int(round(phase[g])) in (3, 4))
    controls = rng.choice(sorted(set(allg) - pos), size=args.n_control, replace=False)
    order = ["Phase 3", "Phase 4", f"Control (n={args.n_control})"]
    n3 = sum(int(round(phase[g])) == 3 for g in targets); n4 = sum(int(round(phase[g])) == 4 for g in targets)
    xlabels = [f"Phase 3\n(n={n3})", f"Phase 4\n(n={n4})", f"Control\n(n={len(controls)})"]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    pal = {"EMB": "#1b9e77", "ESM": "#d95f02"}
    clf = "mlp"
    rr = []
    for feat in ["EMB", "ESM"]:
        cm = combined[(clf, feat)]
        for g in targets:
            if g in cm:
                rr.append({"group": f"Phase {int(round(phase[g]))}", "P": cm[g], "feature": feat})
        for g in controls:
            if g in cm:
                rr.append({"group": order[-1], "P": cm[g], "feature": feat})
    dfb = pd.DataFrame(rr)
    sns.boxplot(data=dfb, x="group", y="P", hue="feature", order=order, hue_order=["EMB", "ESM"],
                palette=pal, showfliers=False, ax=ax)
    sns.stripplot(data=dfb, x="group", y="P", hue="feature", order=order, hue_order=["EMB", "ESM"],
                  dodge=True, color="black", size=3, alpha=0.5, ax=ax, legend=False)
    lines = []
    for feat in ["EMB", "ESM"]:
        h10, h100, mrr, _ = rank_metrics(combined[(clf, feat)], pos)
        lines.append(f"{feat}:  H@10={h10}  H@100={h100}  MRR={mrr:.4f}")
    ax.text(0.985, 0.985, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace", bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))
    ax.set(ylabel="P(drug target) [max across diseases, OOF]", xlabel="", title=f"MLP — per-disease pooled + max-agg all-OT — {args.run}")
    ax.set_xticklabels(xlabels)
    ax.legend(title="", loc="upper left")
    fig.tight_layout()
    out = SEQ / "images" / f"{tag}_classifier_boxplot.png"
    fig.savefig(out, dpi=130); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
