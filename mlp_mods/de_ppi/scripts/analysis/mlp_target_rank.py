"""Rank ALL proteins by predicted P(disease target) from embedding position, using ALL diseases' targets,
with DISEASE-MATCHED labels: a (protein, network) instance is positive only if the protein is an OT target
(score_indirect > pos_thr) for THAT network's disease. So an ILD target is positive in ILD networks but
negative in Crohn networks; healthy networks are all negative.

Feature = each network's resting Z[protein]. GROUPED K-fold by protein -> out-of-fold (non-leaky) score per
(protein, network). Outputs a context-resolved ranked table (protein, arm/tissue/celltype, prob, matched label).

Output (results/<main>/mlp/):
  mlp_alldisease_rank_by_context.tsv   per (protein, network) sorted by prob, with arm/tissue/celltype
  mlp_alldisease_rank.tsv              per-protein: mean/max prob, best context, disease_hits, per-disease OT

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/mlp_target_rank.py \
        --main-name crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct --diseases crohn,uc,alz,ild
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

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
           "alz": "alzheimer_target_association_MONDO_0004975.tsv", "ild": "ild_target_association_EFO_0004244.tsv"}


def parse_main(tag):
    p = tag.split("_"); return p[0], p[1], p[2]


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main(main_name, diseases, pos_thr, folds, epochs, hidden, seed) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    node_id = np.asarray(e["node_id"])[pi]
    tags = list(e["tags"])

    ot = {d: dict(zip(o.gene_symbol, o.score_indirect)) for d in diseases
          for o in [pd.read_csv(OT_DIR / OT_FILE[d], sep="\t")]}
    per_dis = pd.DataFrame({d: [ot[d].get(g, 0.0) for g in node_id] for d in diseases}, index=node_id).round(4)
    dis_set = set(diseases)

    feats, prot, lab, net = [], [], [], []
    for ti in range(len(tags)):
        arm = parse_main(tags[ti])[0]
        for gi in np.where(pres[ti])[0]:
            g = node_id[gi]
            l = int(ot[arm].get(g, 0.0) > pos_thr) if arm in dis_set else 0   # DISEASE-MATCHED: positive only for this net's disease
            feats.append(Z[ti, gi]); prot.append(g); lab.append(l); net.append(tags[ti])
    X = np.asarray(feats, np.float32); prot = np.asarray(prot); y = np.asarray(lab, np.float32); net = np.asarray(net)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=prot):
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
                         "network": net, "label": y.astype(int), "prob": oof.round(4)})
    auc = roc_auc_score(inst.label, inst.prob); prc = average_precision_score(inst.label, inst.prob)

    hits = {g: ",".join(d for d in diseases if per_dis.loc[g, d] > pos_thr) for g in node_id}
    # one row per (protein, context), full columns: matched label + own-disease OT + all per-disease OT + which diseases it targets
    t = inst.sort_values("prob", ascending=False).reset_index(drop=True)
    t["matched_label"] = t["label"]                                            # 1 iff target for THIS network's disease
    t["net_disease_ot"] = [round(per_dis.loc[p, a], 4) if a in dis_set else 0.0 for p, a in zip(t.protein, t.arm)]
    t["disease_hits"] = t.protein.map(hits)
    t = t.join(per_dis.rename(columns={d: f"ot_{d}" for d in diseases}), on="protein")
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    cols = ["rank", "protein", "arm", "tissue", "celltype", "prob", "matched_label", "net_disease_ot",
            "disease_hits"] + [f"ot_{d}" for d in diseases]
    t = t[cols]

    out = res / "mlp"; out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "mlp_alldisease_rank.tsv", sep="\t", index=False)
    print(f"diseases={diseases}  disease-MATCHED positive instances={int(inst.label.sum())}/{len(inst)}  "
          f"proteins={inst.protein.nunique()}\ninstance OOF ROC-AUC={auc:.3f}  PR-AUC={prc:.3f} "
          f"(baseline {inst.label.mean():.3f})\n")
    print("TOP 30 (protein, context) by P(target); matched_label=1 iff target for THAT context's disease:")
    print(t.head(30).to_string(index=False))
    print(f"\nwrote {out/'mlp_alldisease_rank.tsv'}  ({len(t)} protein×context rows)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct")
    ap.add_argument("--diseases", default="crohn,uc,alz,ild")
    ap.add_argument("--pos-thr", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, [d for d in a.diseases.split(",") if d], a.pos_thr, a.folds, a.epochs, a.hidden, a.seed))
