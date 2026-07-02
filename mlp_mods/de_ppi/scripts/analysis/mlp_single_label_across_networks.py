"""Train ONE MLP with a SINGLE fixed target label (default: Crohn OT targets, score_indirect > thr) using each
protein's per-network embedding position Z, pooled across ALL main networks. The label is protein-level and
disease-agnostic (a Crohn target is labelled 1 in EVERY network); only the feature Z changes per network.

Held-out by PROTEIN (grouped). Then score every (protein, network) pair and COMPARE the predicted
P(Crohn-target) across disease × tissue × cell type × cell state networks — i.e. do Crohn targets look most
'Crohn-target-like' when embedded in Crohn-relevant contexts (colon/gut) vs other diseases/healthy?

Outputs (results/<main>/mlp/):
  mlp_<label>_label_metrics.tsv   held-out ROC-AUC/PR-AUC per NETWORK (TP=label proteins vs TN), + overall
  mlp_<label>_label_scores.tsv    protein, network, arm, tissue, celltype, state, present, label, split, score
  mlp_<label>_label_boxplot.png   held-out P(target) per network, split TP vs TN (compare across contexts)

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/mlp_single_label_across_networks.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr --label-disease crohn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from plot_style import apply_style, TOL

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
           "alz": "alzheimer_target_association_MONDO_0004975.tsv", "ild": "ild_target_association_EFO_0004244.tsv"}
ARM_ORDER = ["crohn", "uc", "alz", "ild", "healthy"]


def parse_main(tag):
    p = tag.split("_")
    return dict(arm=p[0], tissue=p[1], ct=p[2], state="_".join(p[3:]))


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main(main_name, label_disease, pos_thr, test_frac, hidden, epochs, seed) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    node_id = np.asarray(e["node_id"])[pi]
    tags = list(e["tags"]); P = {t: parse_main(t) for t in tags}

    ot = pd.read_csv(OT_DIR / OT_FILE[label_disease], sep="\t")
    score_map = dict(zip(ot.gene_symbol, ot.score_indirect))
    y_prot = {g: int(score_map.get(g, 0.0) > pos_thr) for g in node_id}    # FIXED per-protein label

    rows_feat, meta = [], []
    for ti, t in enumerate(tags):
        for gi in np.where(pres[ti])[0]:
            g = node_id[gi]
            rows_feat.append(Z[ti, gi])
            meta.append((g, t, P[t]["arm"], P[t]["tissue"], P[t]["ct"], P[t]["state"], y_prot[g]))
    X = np.asarray(rows_feat, dtype=np.float32)
    M = pd.DataFrame(meta, columns=["protein", "network", "arm", "tissue", "celltype", "state", "label"])

    uprot = np.array(sorted(M.protein.unique()))
    test_prot = set(rng.choice(uprot, int(round(test_frac * len(uprot))), replace=False))
    is_test = M.protein.isin(test_prot).to_numpy()
    M["split"] = np.where(is_test, "test", "train")
    tr = ~is_test

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
    for _ in range(epochs):
        model.train(); perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad(); lossf(model(Xt[b]), yt[torch.tensor(b, device=dev)]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        M["score"] = torch.sigmoid(model(Xt)).cpu().numpy().round(5)

    # per-network held-out AUC (label proteins vs not, within each network's held-out proteins)
    te = M[is_test]
    mrows = []
    for t in tags:
        s = te[te.network == t]
        yy = s.label.to_numpy()
        au = roc_auc_score(yy, s.score) if 0 < yy.sum() < len(yy) else np.nan
        pr = average_precision_score(yy, s.score) if yy.sum() else np.nan
        q = P[t]
        mrows.append(dict(network=t, arm=q["arm"], tissue=q["tissue"], celltype=q["ct"], state=q["state"],
                          n=len(yy), n_pos=int(yy.sum()), roc_auc=round(au, 4) if au == au else np.nan,
                          pr_auc=round(pr, 4) if pr == pr else np.nan))
    mdf = pd.DataFrame(mrows)
    ov = roc_auc_score(te.label, te.score)
    mdf = pd.concat([pd.DataFrame([dict(network="OVERALL", arm="all", roc_auc=round(ov, 4),
                     n=len(te), n_pos=int(te.label.sum()))]), mdf], ignore_index=True)

    out = res / "mlp"; out.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out / f"mlp_{label_disease}_label_metrics.tsv", sep="\t", index=False)
    M.to_csv(out / f"mlp_{label_disease}_label_scores.tsv", sep="\t", index=False)

    # boxplot: held-out P(target) per network, TP vs TN, ordered by arm then tissue/ct/state
    apply_style()
    te = te.copy()
    te["cls"] = np.where(te.label == 1, "TP", "TN")
    order = sorted(tags, key=lambda t: (ARM_ORDER.index(P[t]["arm"]), P[t]["tissue"], P[t]["ct"], P[t]["state"]))
    fig, ax = plt.subplots(figsize=(max(12, 0.42 * len(order)), 6))
    sns.boxplot(data=te, x="network", y="score", hue="cls", hue_order=["TP", "TN"], order=order,
                showfliers=False, palette={"TP": TOL["red"], "TN": TOL["grey"]}, ax=ax)
    ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=90, fontsize=6.5)
    ax.set_xlabel(""); ax.set_ylabel(f"held-out P({label_disease}-target)")
    ax.set_title(f"MLP trained on {label_disease.upper()} OT targets (>{pos_thr}), single fixed label — "
                 f"held-out P(target) across every network  [overall AUC {ov:.3f}]", fontsize=9.5)
    ax.legend(title="", loc="upper right")
    fig.tight_layout(); fig.savefig(out / f"mlp_{label_disease}_label_boxplot.png")

    print(mdf[mdf.network != "OVERALL"].sort_values("roc_auc", ascending=False)
          [["network", "n_pos", "roc_auc", "pr_auc"]].to_string(index=False))
    print(f"\nOVERALL held-out AUC={ov:.3f}  label={label_disease} pos_thr={pos_thr}  "
          f"pos proteins={sum(y_prot.values())}/{len(y_prot)}")
    print(f"wrote {out}/mlp_{label_disease}_label_(metrics|scores).tsv, _boxplot.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--label-disease", default="crohn", choices=list(OT_FILE))
    ap.add_argument("--pos-thr", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.label_disease, a.pos_thr, a.test_frac, a.hidden, a.epochs, a.seed))
