"""Rank ALL proteins by predicted P(Crohn OT target) from their embedding position across all networks.

Single fixed label = Crohn OT target (score_indirect > pos_thr). Feature = each network's resting Z[protein].
GROUPED K-fold by protein -> every protein gets an OUT-OF-FOLD (non-leaky) score in each network it's in.
Per protein we aggregate the OOF scores over its networks (mean + max) and rank. Runs on any build (default the
PINNACLE cell-type build, where NOD2/TNF are readmitted).

Output (results/<main>/mlp/): mlp_crohn_rank.tsv  (protein, mean_prob, max_prob, n_nets, crohn_ot_score, label), sorted.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/mlp_crohn_target_rank.py \
        --main-name crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

OT_FILE = Path("mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv")


def parse_main(tag):
    p = tag.split("_")
    return p[0], p[1], p[2]              # arm, tissue, celltype


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main(main_name, pos_thr, folds, epochs, hidden, seed) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    node_id = np.asarray(e["node_id"])[pi]
    tags = list(e["tags"])

    ot = pd.read_csv(OT_FILE, sep="\t")
    score_map = dict(zip(ot.gene_symbol, ot.score_indirect))
    ylab = {g: int(score_map.get(g, 0.0) > pos_thr) for g in node_id}

    feats, prot, lab, net = [], [], [], []
    for ti in range(len(tags)):
        for gi in np.where(pres[ti])[0]:
            feats.append(Z[ti, gi]); prot.append(node_id[gi]); lab.append(ylab[node_id[gi]]); net.append(tags[ti])
    X = np.asarray(feats, np.float32); prot = np.asarray(prot); y = np.asarray(lab, np.float32); net = np.asarray(net)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups=prot):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xt = torch.tensor((X - mu) / sd, device=dev); yt = torch.tensor(y, device=dev)
        model = MLP(X.shape[1], hidden).to(dev)
        pw = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(epochs):
            model.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm), 8192):
                b = perm[i:i + 8192]; opt.zero_grad()
                lossf(model(Xt[b]), yt[torch.tensor(b, device=dev)]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(Xt[torch.tensor(te, device=dev)])).cpu().numpy()

    arm, tissue, ct = zip(*[parse_main(t) for t in net])
    inst = pd.DataFrame({"protein": prot, "arm": arm, "tissue": tissue, "celltype": ct,
                         "label": y.astype(int), "prob": oof.round(4)})
    inst["crohn_ot_score"] = inst.protein.map(score_map).fillna(0.0).round(4)
    auc = roc_auc_score(inst.label, inst.prob); prc = average_precision_score(inst.label, inst.prob)

    # (1) per-(protein, network-context) table sorted by probability -- shows WHERE each score is from
    by_ctx = inst.sort_values("prob", ascending=False).reset_index(drop=True)
    by_ctx.insert(0, "rank", np.arange(1, len(by_ctx) + 1))
    # (2) per-protein aggregate: best-scoring context (max prob) + which context that was
    idxmax = inst.loc[inst.groupby("protein").prob.idxmax()]
    per_prot = (inst.groupby("protein").agg(mean_prob=("prob", "mean"), max_prob=("prob", "max"),
                                            n_nets=("prob", "size"), label=("label", "max")).reset_index())
    best = idxmax.set_index("protein")[["arm", "tissue", "celltype"]].rename(
        columns={"arm": "best_arm", "tissue": "best_tissue", "celltype": "best_celltype"})
    per_prot = per_prot.join(best, on="protein")
    per_prot["crohn_ot_score"] = per_prot.protein.map(score_map).fillna(0.0).round(4)
    per_prot = per_prot.sort_values("mean_prob", ascending=False).reset_index(drop=True)
    per_prot.insert(0, "rank", np.arange(1, len(per_prot) + 1))

    out = res / "mlp"; out.mkdir(parents=True, exist_ok=True)
    by_ctx.to_csv(out / "mlp_crohn_rank_by_context.tsv", sep="\t", index=False)
    per_prot.to_csv(out / "mlp_crohn_rank.tsv", sep="\t", index=False)
    print(f"instance-level OOF ROC-AUC={auc:.3f}  PR-AUC={prc:.3f} (baseline {inst.label.mean():.3f})  "
          f"instances={len(inst)} proteins={per_prot.shape[0]} positives={int(per_prot.label.sum())}\n")
    print("TOP 30 (protein, network context) by P(Crohn target):")
    print(by_ctx.head(30)[["rank", "protein", "arm", "tissue", "celltype", "prob", "label", "crohn_ot_score"]].to_string(index=False))
    print(f"\nwrote {out/'mlp_crohn_rank_by_context.tsv'} (per-context) and {out/'mlp_crohn_rank.tsv'} (per-protein + best context)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct")
    ap.add_argument("--pos-thr", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.pos_thr, a.folds, a.epochs, a.hidden, a.seed))
