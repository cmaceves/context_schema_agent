"""MLP classifier: is a protein 'disease-relevant' (OpenTargets score_indirect > 0.1) from its per-network
embedding position Z? Trained pooled across all four diseases, evaluated on a held-out set of PROTEINS
(grouped split, so no protein leaks between train and test), then used to score EVERY protein in EVERY
healthy/disease × tissue × cell type × cell state MAIN network.

Design (per user):
  - instances = (protein, network) over the 44 MAIN networks (embeddings.npz); feature = that network's
    resting embedding Z[protein] (64-d). Position, not shift, so healthy networks can be scored/trained too.
  - label = 1 if the network's DISEASE has OT score_indirect > pos_thr for that protein, else 0.
            healthy networks -> all proteins labelled 0 (trained as negatives).
  - pooled single MLP; class-weighted BCE (positives are a small minority).
  - held-out = a fraction of PROTEINS held out entirely (grouped); report ROC-AUC / PR-AUC overall + per disease.
  - score all (protein, network) pairs -> output table.

Outputs (results/<main>/mlp/):
  mlp_target_metrics.tsv     held-out ROC-AUC / PR-AUC (overall + per disease) + class balance
  mlp_target_scores.tsv      protein, network, arm, tissue, celltype, state, present, label, split, score
  mlp_target_boxplot.png     held-out P(disease-relevant) for true-pos vs true-neg (per disease)

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/mlp_target_classifier.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from plot_style import apply_style, ARM_COLOR, TOL

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
           "alz": "alzheimer_target_association_MONDO_0004975.tsv", "ild": "ild_target_association_EFO_0004244.tsv"}
DISEASES = set(OT_FILE)


def parse_main(tag: str) -> dict:
    p = tag.split("_")
    return dict(arm=p[0], tissue=p[1], ct=p[2], state="_".join(p[3:]))


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main(main_name, pos_thr, test_frac, hidden, epochs, seed) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    node_id = np.asarray(e["node_id"])[pi]
    tags = list(e["tags"]); P = {t: parse_main(t) for t in tags}

    ot_score = {a: dict(zip(o.gene_symbol, o.score_indirect))
                for a, f in OT_FILE.items() for o in [pd.read_csv(OT_DIR / f, sep="\t")]}

    # build instance table over present (protein, network) pairs
    rows_feat, meta = [], []
    for ti, t in enumerate(tags):
        arm = P[t]["arm"]
        present = np.where(pres[ti])[0]
        for gi in present:
            g = node_id[gi]
            lab = int(ot_score[arm].get(g, 0.0) > pos_thr) if arm in DISEASES else 0   # healthy -> 0
            rows_feat.append(Z[ti, gi])
            meta.append((g, t, arm, P[t]["tissue"], P[t]["ct"], P[t]["state"], lab))
    X = np.asarray(rows_feat, dtype=np.float32)
    M = pd.DataFrame(meta, columns=["protein", "network", "arm", "tissue", "celltype", "state", "label"])

    # grouped split by PROTEIN
    uprot = np.array(sorted(M.protein.unique()))
    test_prot = set(rng.choice(uprot, int(round(test_frac * len(uprot))), replace=False))
    is_test = M.protein.isin(test_prot).to_numpy()
    M["split"] = np.where(is_test, "test", "train")
    tr, te = ~is_test, is_test

    # standardize on train
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    y = M.label.to_numpy().astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(Xs, device=dev); yt = torch.tensor(y, device=dev)
    model = MLP(X.shape[1], hidden).to(dev)
    pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tr_idx = np.where(tr)[0]
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[torch.tensor(b, device=dev)])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(Xt)).cpu().numpy()
    M["score"] = score.round(5)

    # ---- held-out metrics (overall + per disease); healthy has no positives so skip its AUC ----
    def metrics(mask):
        yy, ss = y[mask], score[mask]
        if yy.sum() == 0 or yy.sum() == len(yy):
            return (np.nan, np.nan, int(yy.sum()), int(len(yy)))
        return (roc_auc_score(yy, ss), average_precision_score(yy, ss), int(yy.sum()), int(len(yy)))
    mrows = []
    for name, msk in [("overall_disease", te & M.arm.isin(DISEASES).to_numpy())] + \
                     [(a, te & (M.arm == a).to_numpy()) for a in sorted(DISEASES)]:
        au, pr, npos, n = metrics(msk)
        mrows.append(dict(subset=name, roc_auc=round(au, 4) if au == au else np.nan,
                          pr_auc=round(pr, 4) if pr == pr else np.nan, n_pos=npos, n=n,
                          pos_rate=round(npos / n, 4) if n else np.nan))
    mdf = pd.DataFrame(mrows)

    out = res / "mlp"; out.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out / "mlp_target_metrics.tsv", sep="\t", index=False)
    M.to_csv(out / "mlp_target_scores.tsv", sep="\t", index=False)

    # ---- boxplot: held-out P(relevant) for true pos vs neg, faceted by disease, split by tissue/cell type ----
    import seaborn as sns
    apply_style()
    arms = sorted(DISEASES)
    sub = M[te].copy()
    sub["context"] = sub.tissue + "/" + sub.celltype                   # states pooled within a tissue/celltype
    sub["cls"] = np.where(sub.label == 1, "TP", "TN")
    auc_by = {a: mdf.loc[mdf.subset == a, "roc_auc"].iloc[0] for a in arms}
    g = sns.catplot(data=sub, x="context", y="score", hue="cls", hue_order=["TP", "TN"],
                    col="arm", col_order=arms, kind="box", col_wrap=2, showfliers=False,
                    sharex=False, height=3.2, aspect=1.5, legend_out=False,
                    palette={"TP": TOL["red"], "TN": TOL["grey"]})
    for ax, a in zip(g.axes.flat, arms):
        ax.set_title(f"{a}  (overall AUC {auc_by[a]})", fontsize=10)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    g.set_ylabels("held-out P(disease-relevant)")
    g.figure.suptitle(f"MLP OT-relevant (>{pos_thr}) by disease × tissue/cell type — held-out proteins  "
                      f"[{main_name.split('_')[-1]}]", fontsize=10)
    g.figure.tight_layout()
    g.figure.savefig(out / "mlp_target_boxplot.png")

    print(mdf.to_string(index=False))
    print(f"\ninstances={len(M)}  proteins={len(uprot)} (test {len(test_prot)})  "
          f"disease-positive rate (train)={y[tr & M.arm.isin(DISEASES).to_numpy()].mean():.3f}")
    print(f"wrote {out}/mlp_target_(metrics|scores).tsv, mlp_target_boxplot.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--pos-thr", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.pos_thr, a.test_frac, a.hidden, a.epochs, a.seed))
