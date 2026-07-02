"""5-fold CV: can the FULL 64-d disease-shift embedding vector predict OpenTargets target labels?

For each disease arm we build, per colon-macrophage protein, the 64-d healthy-centered shift vector
    ΔZ(p) = mean_studies( Z_disease(study, allstates)[p] - Z_healthy(study, allstates)[p] )   (own-study centered)
and ask whether a plain L2-logistic model on that FULL vector (not its magnitude) separates OT positives
(score_indirect >= --pos-thr) from negatives (< --pos-thr). Repeated stratified 5-fold CV; features are
standardized inside each fold (no leakage); class_weight balanced for the label imbalance.

Positive threshold defaults to 0.1: at 0.5 only ~10 positives survive in the node set (unusable for CV).

Outputs (results/<main>/cv/):
  embedding_target_cv_metrics.tsv  per disease: mean/sd ROC-AUC & PR-AUC over repeats, positive rate (PR-AUC baseline)
  embedding_target_cv_probs.tsv    per protein: disease, true_label, oof predicted probability (mean over repeats)
  embedding_target_cv_boxplot.png  boxplot of predicted P(target) for true-positives vs true-negatives, per disease

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/embedding_target_cv.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from disease_axis_decompose import parse
from plot_style import apply_style, ARM_COLOR, TOL

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv"}


def disease_shift(Z, pres, idx, P, tags, tissue, ct, arm, center=True):
    """(present-mask, mean 64-d matrix) for (arm, tissue, ct) allstates, averaged over the arm's studies.
    center=True  -> healthy-centered shift  ΔZ = Z_disease - Z_own-study-healthy  (disease displacement).
    center=False -> raw disease POSITION     Z_disease  (no healthy subtraction; tests static position)."""
    def prim(q, a):
        return (q["arm"] == a and q["tissue"] == tissue and q["ct"] == ct and q["state"] == "allstates"
                and not q["loo"] and q["split"] is None)
    dis = [t for t in tags if prim(P[t], arm)]
    if not dis:
        raise SystemExit(f"no {arm} {tissue} {ct} allstates net")
    m = np.ones(Z.shape[1], bool)
    if center:
        Hall = {P[t]["study"]: t for t in tags if prim(P[t], "healthy")}
        pairs = [(idx[t], idx[Hall[P[t]["study"]]]) for t in dis if P[t]["study"] in Hall]
        for di, hi in pairs:
            m &= pres[di] & pres[hi]
        R = np.mean([Z[di] - Z[hi] for di, hi in pairs], axis=0)
    else:
        di_list = [idx[t] for t in dis]
        for di in di_list:
            m &= pres[di]
        R = np.mean([Z[di] for di in di_list], axis=0)
    return m, R


def cv_disease(X, y, folds, repeats, seed):
    """Repeated stratified k-fold L2-logistic. Returns (oof_prob mean over repeats, per-repeat aucs, prcs)."""
    n = len(y)
    oof = np.zeros((repeats, n))
    aucs, prcs = [], []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        p = np.zeros(n)
        for tr, te in skf.split(X, y):
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
            model.fit(X[tr], y[tr])
            p[te] = model.predict_proba(X[te])[:, 1]
        oof[rep] = p
        aucs.append(roc_auc_score(y, p))
        prcs.append(average_precision_score(y, p))
    return oof.mean(0), np.array(aucs), np.array(prcs)


def main(main_name, tissue, ct, arms, pos_thr, folds, repeats, seed, feature) -> int:
    center = feature == "shift"
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres = c["Z"][:, pi, :], c["present"][:, pi]
    node_id = np.asarray(c["node_id"])[pi]
    tags = list(c["tags"]); idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}

    metric_rows, prob_rows = [], []
    for arm in arms:
        m, R = disease_shift(Z, pres, idx, P, tags, tissue, ct, arm, center=center)
        genes = node_id[m]
        X = R[m]                                                          # (n_proteins, 64) full vectors (ΔZ or Z)
        ot = pd.read_csv(OT_DIR / OT_FILE[arm], sep="\t")
        score = pd.Series(genes).map(dict(zip(ot.gene_symbol, ot.score_indirect))).fillna(0.0).to_numpy()
        y = (score >= pos_thr).astype(int)
        oof, aucs, prcs = cv_disease(X, y, folds, repeats, seed)
        metric_rows.append(dict(disease=arm, n=len(y), n_pos=int(y.sum()), pos_rate=round(y.mean(), 4),
                                roc_auc_mean=round(aucs.mean(), 4), roc_auc_sd=round(aucs.std(), 4),
                                pr_auc_mean=round(prcs.mean(), 4), pr_auc_sd=round(prcs.std(), 4)))
        for g, lab, pr in zip(genes, y, oof):
            prob_rows.append(dict(disease=arm, protein=g, true_label=int(lab), oof_prob=round(float(pr), 5)))
        print(f"  {arm}: n={len(y)} pos={int(y.sum())} ({y.mean():.1%})  "
              f"ROC-AUC={aucs.mean():.3f}±{aucs.std():.3f}  PR-AUC={prcs.mean():.3f} (baseline {y.mean():.3f})")

    out = res / "cv"; out.mkdir(parents=True, exist_ok=True)
    tag = f"{feature}_pos{pos_thr}"
    mdf = pd.DataFrame(metric_rows); pdf = pd.DataFrame(prob_rows)
    mdf.to_csv(out / f"embedding_target_cv_metrics_{tag}.tsv", sep="\t", index=False)
    pdf.to_csv(out / f"embedding_target_cv_probs_{tag}.tsv", sep="\t", index=False)

    # boxplot: predicted P(target) for true-positives vs true-negatives, per disease
    apply_style()
    fig, ax = plt.subplots(figsize=(1.9 * len(arms) + 2, 4.4))
    positions, ticklabels, colors = [], [], []
    for i, arm in enumerate(arms):
        sub = pdf[pdf.disease == arm]
        for j, lab in enumerate((1, 0)):
            positions.append(i * 2.6 + j)
            ticklabels.append(f"{arm}\n{'TP' if lab else 'TN'}")
            colors.append(ARM_COLOR[arm] if lab else TOL["grey"])
        for j, lab in enumerate((1, 0)):
            vals = sub.loc[sub.true_label == lab, "oof_prob"].values
            bp = ax.boxplot(vals, positions=[i * 2.6 + j], widths=0.7, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(ARM_COLOR[arm] if lab else "white")
            bp["boxes"][0].set_edgecolor(ARM_COLOR[arm] if lab else "0.4")
        au = mdf.loc[mdf.disease == arm, "roc_auc_mean"].iloc[0]
        pr = mdf.loc[mdf.disease == arm, "pr_auc_mean"].iloc[0]
        ax.text(i * 2.6 + 0.5, 0.97, f"AUC {au:.2f}\nPR {pr:.2f}", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xticks(positions); ax.set_xticklabels(ticklabels, fontsize=9)
    ax.set_ylabel("out-of-fold predicted P(OT target)")
    feat_lbl = "ΔZ (healthy-centered shift)" if center else "Z (raw disease position)"
    fig.suptitle(f"{tissue} {ct}: full-{feat_lbl} 5-fold logistic — P(target), true pos vs neg  "
                 f"(pos = OT score≥{pos_thr}; {repeats}×{folds}-fold CV)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / f"embedding_target_cv_boxplot_{tag}.png")
    print(f"\nwrote {out}/embedding_target_cv_(metrics|probs|boxplot)_{tag}.*")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--tissue", default="colon")
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--arms", default="crohn,uc")
    ap.add_argument("--pos-thr", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature", choices=("shift", "position"), default="position",
                    help="shift = Z_disease - Z_healthy (ΔZ); position = raw Z_disease (no healthy subtraction)")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.tissue, a.celltype, [x for x in a.arms.split(",") if x],
                          a.pos_thr, a.folds, a.repeats, a.seed, a.feature))
