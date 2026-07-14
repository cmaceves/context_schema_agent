"""All-OT-target global-pooled LogReg, across ALL embedding builds. For each results/link_*/embeddings.npz:
mean-pool each protein over all contexts, train L2-LogReg (balanced, 5-fold OOF) on the union of drug targets
(all known_drugs_*.tsv), and rank all proteins. Reports EMB-pooled vs ESM (both on that run's pooled gene set)
per build -> table + MRR bar plot. ESM is embedding-independent (varies only slightly with gene set).

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/compare_pooled_logreg_all.py
Out: results/pooled_logreg_all_embeddings.tsv ; images/pooled_logreg_all_embeddings.png
"""
from __future__ import annotations
import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

SEQ = Path("mlp_mods/seq_context")
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000))


def rank_metrics(score, pos):
    ranked = sorted(score, key=lambda g: -score[g]); rank = {g: i + 1 for i, g in enumerate(ranked)}
    P = [g for g in ranked if g in pos]
    if not P:
        return 0, 0, float("nan")
    return sum(rank[g] <= 10 for g in P), sum(rank[g] <= 100 for g in P), float(np.mean([1 / rank[g] for g in P]))


def pool_all(d):
    E = d["emb"].astype(np.float64); idx = d["prot_idx"]   # load emb ONCE (npz re-reads on every access)
    acc = defaultdict(list)
    for j, i in enumerate(idx):
        acc[PROT[i]].append(E[j])
    return {g: np.mean(v, 0) for g, v in acc.items()}


def score(X, y, genes):
    P = cross_val_predict(clf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return dict(zip(genes, P))


def main():
    pos = set()
    for f in glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"):
        pos |= set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))
    print(f"all-OT drug targets: {len(pos)} genes from {len(glob.glob('mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv'))} files", flush=True)

    runs = sorted(Path(p).parent.name for p in glob.glob(str(SEQ / "results/link_*/embeddings.npz")))
    rows = []
    for run in runs:
        d = np.load(SEQ / "results" / run / "embeddings.npz", allow_pickle=True)
        gp = pool_all(d)
        genes = np.array(sorted(set(gp) & set(PROT)))
        y = np.array([1 if g in pos else 0 for g in genes])
        emb = np.stack([gp[g] for g in genes])
        esm = np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
        for feat, X in [("EMB", emb), ("ESM", esm)]:
            h10, h100, mrr = rank_metrics(score(X, y, genes), pos)
            rows.append({"run": run, "feature": feat, "n_ctx": len(set(d["context"])), "n_pos": int(y.sum()),
                         "hits@10": h10, "hits@100": h100, "MRR": round(mrr, 4)})
        print(f"{run:22s} EMB MRR={rows[-2]['MRR']:.4f}  ESM MRR={rows[-1]['MRR']:.4f}", flush=True)

    tab = pd.DataFrame(rows)
    out = SEQ / "results" / "pooled_logreg_all_embeddings.tsv"
    tab.to_csv(out, sep="\t", index=False)
    print("\n" + tab.to_string(index=False) + f"\nwrote {out}", flush=True)

    piv = tab.pivot(index="run", columns="feature", values="MRR").reindex(runs)
    ax = piv.plot(kind="bar", figsize=(12, 5), color={"EMB": "#1b9e77", "ESM": "#d95f02"})
    ax.set(ylabel="MRR (all-OT targets, global pool)", xlabel="",
           title="Global-pooled LogReg target recovery across embedding builds — EMB(pooled) vs ESM")
    ax.legend(title=""); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    p = SEQ / "images" / "pooled_logreg_all_embeddings.png"
    plt.savefig(p, dpi=130); print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
