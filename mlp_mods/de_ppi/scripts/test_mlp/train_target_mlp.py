"""PINNACLE-style MULTI-DISEASE target-prediction MLP over the crohn_alzheimer_ild_uc_embedding_expressed
build (Crohn, ulcerative colitis, ILD, Alzheimer).

Each training instance = a (protein, context) pair: the protein's 64-d embedding in one network. Labels
are CELL-TYPE-RESOLVED OpenTargets (phase>=PHASE_FLOOR, default 1 = any clinical-phase drug target), and
each context is labeled by ITS OWN disease — parsed from the tag prefix:
  alz_*    -> Alzheimer   MONDO_0004975
  crohn_*  -> Crohn       EFO_0000384
  uc_*     -> UC          EFO_0000729
  ild_*    -> ILD         EFO_0004244
  healthy_*-> no disease  (unlabeled -> dropped)
Labels are COLLAPSED to disease level (OpenTargets has no cell-type resolution): a protein is labeled 1
in EVERY context of its disease where it is present in our networks if it is a disease target, 0 if it
has no drug evidence, else dropped. Presence in our networks — not PINNACLE PPI membership — gates it.

IBD (EFO_0003767) is NOT used as a positive set. But IBD positives are also NOT allowed to act as
negatives for Crohn or UC: for crohn_/uc_ contexts, any gene in the IBD positive set is removed from the
negative set (dropped, not labeled 0) — we don't penalize the model for nominating a known IBD target.

LABELS are binary {0,1}; the MLP OUTPUT is a probability in [0,1] (sigmoid) you rank targets by.

OUT-OF-FOLD (OOF) scoring: K-fold StratifiedGroupKFold grouped by PROTEIN. Each protein's instances sit
in exactly one held-out fold; that fold's model is trained only on the OTHER proteins, so a protein's
score always comes from a model that never saw it. Metrics pool OOF probs across folds. The ranked
deliverable is per (protein, disease, cell type): the mean OOF prob over that group's contexts.

Class imbalance is handled by subsampling TRAINING negatives to --neg-ratio per positive (default 25);
the held-out set is never subsampled, so OOF metrics and the ranking reflect the full negative base rate.

Outputs (--out-dir):
  ranked_targets_oof.tsv         protein, disease, cell_type, mean_oof_prob, n_contexts, label  (ranked)
  predictions_per_instance.tsv   protein, context, disease, cell_type, label, oof_prob, fold
  metrics.txt                    pooled-OOF AUROC/AUPRC (overall + per disease + per cell type) + counts

Run: .venv/bin/python mlp_mods/de_ppi/scripts/test_mlp/train_target_mlp.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("mlp_mods")
OT = ROOT / "03_opentargets_rebuild"
IBD = "EFO_0003767"                                   # masked-out, never a positive or a Crohn/UC negative
PHASE_FLOOR = 1                                        # positives = OpenTargets targets at maxPhase >= this
DISEASE_EFO = {"alz": "MONDO_0004975", "crohn": "EFO_0000384",
               "uc": "EFO_0000729", "ild": "EFO_0004244"}


def disease_of(tag: str) -> str | None:
    p = tag.split("_")[0]
    return p if p in DISEASE_EFO else None            # "healthy" -> None


def celltype_of(tag: str) -> str | None:
    """Map a context tag to its PINNACLE cell-type key (the json key used by OpenTargets)."""
    if "microglia" in tag:
        return "microglial cell"
    if "fibroblast" in tag:
        return "fibroblast"
    if "stem" in tag:
        return "intestinal crypt stem cell"
    if "mac" in tag:                                  # matches both crohn_mac_* and *_macrophage_*
        return "macrophage"
    return None


def load_efo(code: str):
    """Load OpenTargets labels and COLLAPSE to disease level: positives/negatives are the union across
    cell types. OpenTargets has no cell-type resolution — the per-cell-type split is only PINNACLE-PPI
    node membership, so we drop it and let presence in OUR networks decide where a label applies."""
    pos = json.load(open(OT / f"positive_proteins_{code}_new_phase{PHASE_FLOOR}.json"))[code]
    neg = json.load(open(OT / f"negative_proteins_{code}_new_phase{PHASE_FLOOR}.json"))[code]
    P = set().union(*pos.values()) if pos else set()
    N = set().union(*neg.values()) if neg else set()
    return P, N


class MLP(nn.Module):
    def __init__(self, d: int, h: int = 128, p: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1))            # logit; sigmoid at scoring time

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(Xtr, ytr, dev, epochs, lr, hidden, seed, neg_ratio) -> "MLP":
    """Train a fresh MLP on the fold's training proteins and return it (eval mode).

    Class imbalance is handled by SUBSAMPLING negatives to neg_ratio per positive in the TRAINING set
    (held-out set is never subsampled), so plain BCE is used (no pos_weight)."""
    rng = np.random.default_rng(seed)
    pos_i = np.where(ytr == 1)[0]; neg_i = np.where(ytr == 0)[0]
    keep_neg = neg_i if len(neg_i) <= neg_ratio * len(pos_i) else rng.choice(
        neg_i, neg_ratio * len(pos_i), replace=False)
    sel = np.concatenate([pos_i, keep_neg]); Xtr, ytr = Xtr[sel], ytr[sel]
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    model = MLP(Xtr.shape[1], hidden).to(dev)
    lossf = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t); loss.backward(); opt.step()
    model.eval()
    return model


def predict(model, X, dev) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(np.asarray(X, np.float32), device=dev))).cpu().numpy()


def main(build, epochs, lr, seed, hidden, folds, neg_ratio, out_dir) -> int:
    OUT = Path(out_dir)
    torch.manual_seed(seed)
    labels_by_dis = {dis: load_efo(code) for dis, code in DISEASE_EFO.items()}  # disease-level (collapsed)
    ibd_raw = json.load(open(OT / f"positive_proteins_{IBD}_new_phase{PHASE_FLOOR}.json"))[IBD]
    ibd_pos = set().union(*ibd_raw.values()) if ibd_raw else set()
    d = np.load(ROOT / f"de_ppi/results/{build}/embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    ids = np.array(d["node_id"], dtype=object); isp = d["node_type"] == "protein"

    # build (protein, context) instances; each context labeled by ITS OWN disease's EFO set.
    feats, labels, genes, ctxs, diss, cts = [], [], [], [], [], []
    for ti, t in enumerate(tags):
        dis = disease_of(t); ck = celltype_of(t)
        if dis is None or ck is None:
            continue
        P, N = labels_by_dis[dis]                      # disease-level: applies to every context of dis
        if dis in ("crohn", "uc"):                    # don't let IBD targets be Crohn/UC negatives
            N = N - ibd_pos
        idxp = np.where(present[ti] & isp)[0]
        for pi in idxp:
            g = ids[pi]
            lab = 1 if g in P else (0 if g in N else None)
            if lab is None:
                continue
            feats.append(Z[ti, pi]); labels.append(lab); genes.append(g)
            ctxs.append(t); diss.append(dis); cts.append(ck)
    X = np.asarray(feats, np.float32); y = np.asarray(labels); genes = np.asarray(genes, object)
    ctxs = np.asarray(ctxs, object); diss = np.asarray(diss, object); cts = np.asarray(cts, object)
    print(f"instances={len(y)}  pos={int(y.sum())} neg={int((1-y).sum())}  "
          f"unique genes={len(set(genes))}  contexts={len(set(ctxs))}  diseases={sorted(set(diss))}", flush=True)

    # healthy-macrophage reference embedding per protein (for the disease-vs-healthy contrast)
    HEALTHY = "healthy_pinnacle_macrophage"
    healthy_vec = {}
    if HEALTHY in tags:
        hti = tags.index(HEALTHY)
        for pi in np.where(present[hti] & isp)[0]:
            healthy_vec[ids[pi]] = Z[hti, pi]
    print(f"healthy ref '{HEALTHY}': {len(healthy_vec)} proteins", flush=True)

    # K-fold OOF: group by protein (no gene leakage), stratified on label so positives spread across folds.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.full(len(y), np.nan, np.float32); fold_id = np.full(len(y), -1, int)
    oof_healthy_gene = {}   # held-out protein -> its healthy-macrophage score under the same fold model
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups=genes)):
        model = train_fold(X[tr], y[tr], dev, epochs, lr, hidden, seed + k, neg_ratio)
        oof[te] = predict(model, X[te], dev)
        fold_id[te] = k
        hg = [g for g in np.unique(genes[te]) if g in healthy_vec]
        if hg:
            oof_healthy_gene.update(zip(hg, predict(model, [healthy_vec[g] for g in hg], dev)))
        au = roc_auc_score(y[te], oof[te]) if len(set(y[te])) > 1 else float("nan")
        ap = average_precision_score(y[te], oof[te]) if len(set(y[te])) > 1 else float("nan")
        print(f"  fold {k}: train_inst={len(tr)} heldout_inst={len(te)} "
              f"heldout_genes={len(set(genes[te]))} pos={int(y[te].sum())}  AUROC {au:.3f} AUPRC {ap:.3f}",
              flush=True)
    assert not np.isnan(oof).any(), "some instance never received an OOF prediction"

    def strat(mask_vals):
        out = []
        for v in sorted(set(mask_vals)):
            m = mask_vals == v
            if len(set(y[m])) > 1:
                out.append(f"  {v}: n={int(m.sum())} pos={int(y[m].sum())} "
                           f"AUROC={roc_auc_score(y[m], oof[m]):.3f} AUPRC={average_precision_score(y[m], oof[m]):.3f}")
        return out

    auc = roc_auc_score(y, oof); aup = average_precision_score(y, oof)

    # oof_healthy: the protein's score on the healthy-macrophage embedding (same held-out fold model);
    # constant per protein, NaN if the protein is absent from the healthy reference network. Only the
    # healthy MACROPHAGE reference exists, so it is left NaN for non-macrophage contexts (no matched ref).
    oof_healthy = np.array([oof_healthy_gene.get(g, np.nan) for g in genes], np.float32)
    oof_healthy[cts != "macrophage"] = np.nan

    OUT.mkdir(parents=True, exist_ok=True)
    inst = pd.DataFrame({"protein": genes, "context": ctxs, "disease": diss, "cell_type": cts,
                         "label": y, "oof_prob": np.round(oof, 4),
                         "oof_healthy": np.round(oof_healthy, 4), "fold": fold_id}
                        ).sort_values("oof_prob", ascending=False)
    inst.to_csv(OUT / "predictions_per_instance.tsv", sep="\t", index=False)

    # ranked deliverable: per (protein, disease, cell type), mean OOF prob over that group's contexts.
    # delta_vs_healthy = disease score - healthy-macrophage score (cell-type-matched only for macrophage).
    ranked = (inst.groupby(["protein", "disease", "cell_type"])
              .agg(mean_oof_prob=("oof_prob", "mean"), mean_oof_healthy=("oof_healthy", "mean"),
                   n_contexts=("oof_prob", "size"), label=("label", "max")).reset_index())
    ranked["delta_vs_healthy"] = (ranked.mean_oof_prob - ranked.mean_oof_healthy).round(4)
    ranked["mean_oof_prob"] = ranked.mean_oof_prob.round(4)
    ranked["mean_oof_healthy"] = ranked.mean_oof_healthy.round(4)
    ranked = ranked.sort_values("mean_oof_prob", ascending=False)
    ranked.to_csv(OUT / "ranked_targets_oof.tsv", sep="\t", index=False)

    (OUT / "metrics.txt").write_text(
        f"build={build}  (multi-disease, per-context-disease labels; IBD masked from Crohn/UC negatives)\n"
        f"diseases={sorted(set(diss))} phase>={PHASE_FLOOR}  out-of-fold {folds}-fold StratifiedGroupKFold by protein\n"
        f"instances={len(y)} pos={int(y.sum())} neg={int((1-y).sum())} "
        f"unique_genes={len(set(genes))} (protein,disease,celltype)_rows={len(ranked)}\n"
        f"pooled-OOF AUROC={auc:.3f} AUPRC={aup:.3f}\n"
        "per disease:\n" + "\n".join(strat(diss)) + "\n"
        "per cell type:\n" + "\n".join(strat(cts)) + "\n")
    print(f"\npooled-OOF AUROC={auc:.3f} AUPRC={aup:.3f}\nwrote outputs to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-ratio", type=int, default=25, help="negatives per positive in training")
    ap.add_argument("--out-dir", default="mlp_mods/de_ppi/results/_tmp_mlp")
    a = ap.parse_args()
    raise SystemExit(main(a.build, a.epochs, a.lr, a.seed, a.hidden, a.folds, a.neg_ratio, a.out_dir))
