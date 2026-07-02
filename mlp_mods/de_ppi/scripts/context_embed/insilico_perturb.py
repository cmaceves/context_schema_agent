"""In-silico perturbation on the frozen contrastive encoder: does knocking out / normalizing a protein move the
Crohn colon-macrophage network toward its healthy counterpart?

For each candidate protein X, edit X's expression feature in the DISEASE network input and re-run the encoder
(frozen forward pass). The 2-layer message passing propagates X's change to its <=2-hop neighbours, so the
network embedding shifts. We score the shift over the WHOLE network (nodes present in both disease & healthy):

  Z_dis   = encoder(disease net)            Z_hlt = encoder(healthy net)
  Z_pert  = encoder(disease net with X edited)
  delta_distance = mean_p ||Z_pert[p]-Z_hlt[p]||  -  mean_p ||Z_dis[p]-Z_hlt[p]||     (<0 => moved TOWARD healthy)
  projection     = sum_p (Z_pert[p]-Z_dis[p]) . unit(Z_hlt[p]-Z_dis[p])               (>0 => moved TOWARD healthy)

Two perturbations: KO (X expr -> 0) and normalize (X expr -> its healthy value).
Scope: EVERY protein present in the disease network is perturbed (both KO and normalize). Node out-degree /
in-degree in the disease network are recorded as columns so the degree<->response confound is inspectable.

Output: results/<main>/insilico_perturb/perturbation_results.tsv  (one row per protein x perturbation, sorted
by |projection| descending), plus projection_distributions.png (full-population OT positives vs negatives).

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/insilico_perturb.py \
        --main-name crohn_alzheimer_ild_uc_context_contrastive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, "mlp_mods/de_ppi/scripts/embed")
sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
from embedding_utils import Encoder
from joint_embed import Net
from plot_style import ARM_COLOR, TOL, apply_style

OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILE = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
           "alz": "alzheimer_target_association_MONDO_0004975.tsv", "ild": "ild_target_association_EFO_0004244.tsv"}


def main(main_name, base, dis_tag, hlt_tag, ot_disease) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    dev = torch.device("cpu")
    ck = torch.load(res / "encoder.pt", map_location=dev, weights_only=False)
    order = ck["node_id"]; idx = {g: i for i, g in enumerate(order)}; cfg = ck["config"]
    model = Encoder(cfg["N"], cfg["dim"], cfg["layers"], use_self_lin=cfg["use_self_lin"],
                    use_expr_feat=cfg["use_expr_feat"]).to(dev)
    model.load_state_dict(ck["encoder"]); model.eval()
    nroot = Path("mlp_mods/de_ppi/results") / base / "networks"
    dn = Net(dis_tag, idx, dev, nroot / dis_tag, self_loops=cfg["self_loops"])
    hn = Net(hlt_tag, idx, dev, nroot / hlt_tag, self_loops=cfg["self_loops"])

    def embed(net, expr):
        with torch.no_grad():
            return model(net.A, w_feat=net.w_feat, node_feat=expr).cpu().numpy()
    Zd = embed(dn, dn.expr); Zh = embed(hn, hn.expr)
    m = dn.present & hn.present                                   # whole network = nodes present in both
    axis = Zh[m] - Zd[m]; ahat = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
    base_dist = float(np.linalg.norm(Zd[m] - Zh[m], axis=1).mean())

    def score(expr):
        Zp = embed(dn, expr)
        dd = float(np.linalg.norm(Zp[m] - Zh[m], axis=1).mean()) - base_dist
        proj = float(((Zp[m] - Zd[m]) * ahat).sum())
        return dd, proj

    ot_map = dict(zip(*[pd.read_csv(OT_DIR / OT_FILE[ot_disease], sep="\t")[c]
                        for c in ["gene_symbol", "score_indirect"]]))
    # node degrees in the DISEASE network (directed: out-degree = how far a perturbation broadcasts)
    edf = pd.read_csv(nroot / dis_tag / "network_edges.tsv", sep="\t", keep_default_na=False)
    outdeg, indeg = edf.source.value_counts().to_dict(), edf.target.value_counts().to_dict()

    # perturb EVERY protein present in the disease network (normalize: expr -> paired-healthy value)
    present_genes = [order[i] for i in np.where(dn.present)[0]]
    rows = []
    for k, g in enumerate(present_genes):
        xi = idx[g]
        expr_dis, expr_hlt = float(dn.expr[xi]), float(hn.expr[xi])   # model-seen values (0 = placeholder)
        expr = dn.expr.clone(); expr[xi] = expr_hlt                   # normalize to healthy
        dd, proj = score(expr)
        rows.append(dict(protein=g, ot_score=round(ot_map.get(g, 0), 3),
                         out_degree=int(outdeg.get(g, 0)), in_degree=int(indeg.get(g, 0)),
                         expr_disease=round(expr_dis, 4), expr_healthy=round(expr_hlt, 4),
                         delta_distance=round(dd, 5), projection=round(proj, 4)))
        if (k + 1) % 250 == 0:
            print(f"  perturbed {k + 1}/{len(present_genes)}", flush=True)
    df = pd.DataFrame(rows)
    # residual test: regress |projection| ~ out_degree + |Δexpr|; residual = moves more/less than wiring+DE predict
    y = df.projection.abs().to_numpy()
    dexpr = (df.expr_disease - df.expr_healthy).abs().to_numpy()
    X = np.column_stack([np.ones(len(df)), df.out_degree.to_numpy(), dexpr])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df["residual_projection"] = (y - X @ beta).round(4)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    df = df.reindex(df.projection.abs().sort_values(ascending=False).index)   # biggest perturbation first
    out = res / "insilico_perturb"; out.mkdir(parents=True, exist_ok=True)
    tsv = out / f"{dis_tag}_perturbation_results.tsv"
    df.to_csv(tsv, sep="\t", index=False)

    apply_style()
    df["cls"] = np.where(df.ot_score > 0.5, "ot_positive",
                         np.where((df.ot_score > 0) & (df.ot_score < 0.1), "ot_negative", None))

    # (1) projection distribution: full-population OT positives vs negatives
    palette = {"ot_positive": ARM_COLOR.get(ot_disease, ARM_COLOR["crohn"]), "ot_negative": ARM_COLOR["negative"]}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cls, color in palette.items():
        v = df.loc[df.cls == cls, "projection"].to_numpy()
        sns.kdeplot(v, ax=ax, color=color, fill=True, alpha=0.3, bw_adjust=0.8,
                    label=f"{cls} (n={len(v)}, med={np.median(v):.2f})")
        sns.rugplot(v, ax=ax, color=color, height=0.06, lw=1.0, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("projection onto disease→healthy axis  (normalize;  >0 = toward healthy)")
    ax.set_title(f"{dis_tag}: normalize-perturbation projection\nOT {ot_disease} positives (>0.5) vs "
                 f"negatives (<0.1), full population", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{dis_tag}_projection_distributions.png", dpi=150)
    plt.close(fig)

    # (2) out-degree vs |projection|
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(df.out_degree, df.projection.abs(), s=10, alpha=0.4, color=TOL["blue"],
               edgecolors="none", rasterized=True)
    for _, r in df.reindex(df.projection.abs().nlargest(6).index).iterrows():
        ax.annotate(r.protein, (r.out_degree, abs(r.projection)), fontsize=7, fontweight="bold",
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("out-degree (disease network)")
    ax.set_ylabel("|projection onto disease→healthy axis|")
    ax.set_title(f"{dis_tag}: out-degree vs |normalize-perturbation projection|", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / f"{dis_tag}_outdeg_vs_projection.png", dpi=150)
    plt.close(fig)

    from scipy.stats import spearmanr
    rho, p = spearmanr(df.ot_score, df.residual_projection)
    pos = df.residual_projection[df.ot_score > 0.5]; rest = df.residual_projection[df.ot_score <= 0.5]
    print(f"\ndisease={dis_tag} vs {hlt_tag}; ot={ot_disease}; whole-network nodes={int(m.sum())}; baseline dist={base_dist:.3f}")
    print(f"perturbed {len(present_genes)} proteins (normalize) = {len(df)} rows")
    print(f"residual model |proj|~out_degree+|Δexpr|  R²={r2:.3f}")
    print(f"TARGET TEST  Spearman(ot_{ot_disease}, residual) = {rho:+.3f} (p={p:.3f});  "
          f"residual median: OT>0.5 (n={len(pos)})={pos.median():+.3f} vs rest={rest.median():+.3f}")
    print("\ntop 12 by |projection|:")
    print(df.drop(columns="cls").head(12).to_string(index=False))
    print(f"\nwrote {tsv}  ({len(df)} rows, sorted by |projection| desc)")
    print(f"wrote {out}/{dis_tag}_projection_distributions.png, {dis_tag}_outdeg_vs_projection.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_context_contrastive")
    ap.add_argument("--base", default="crohn_alzheimer_ild_uc_coexpr_healthyph")
    ap.add_argument("--dis-tag", default="crohn_colon_macrophage_inflammatory")
    ap.add_argument("--hlt-tag", default="healthy_colon_macrophage_inflammatory")
    ap.add_argument("--ot-disease", default="crohn", choices=["crohn", "uc", "alz", "ild"])
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.base, a.dis_tag, a.hlt_tag, a.ot_disease))
