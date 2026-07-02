"""Control: do SHARED IBD OpenTargets targets co-move in Crohn and UC (relative to healthy) more than negatives?

Colon macrophage, matched INFLAMMATORY state, from a given build's embeddings.npz. Per protein present in all
three nets (crohn/uc/healthy colon-macrophage inflammatory):
    r_crohn = Z_crohn - Z_healthy ;  r_uc = Z_uc - Z_healthy ;  cos = cos(r_crohn, r_uc)
Shared set = OT score_indirect > --pos-thr for BOTH Crohn and UC. Negatives = < --pos-thr in both (sampled
--n-neg). Overlays the two cosine distributions as KDEs.

Output: results/<main>/images/shared_target_cosine_kde.png

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/shared_target_cosine.py \
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

sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
from plot_style import apply_style, TOL

OT_DIR = Path("mlp_mods/opentargets_associations")
OT = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
      "alz": "alzheimer_target_association_MONDO_0004975.tsv", "ild": "ild_target_association_EFO_0004244.tsv"}
TISSUE, CT, STATE = "colon", "macrophage", "inflammatory"


def main(main_name, pos_thr, n_neg, seed) -> int:
    rng = np.random.default_rng(seed)
    res = Path("mlp_mods/de_ppi/results") / main_name
    e = np.load(res / "embeddings.npz", allow_pickle=True)
    pi = np.where(e["node_type"] == "protein")[0]
    Z, pres = e["Z"][:, pi, :], e["present"][:, pi]
    node_id = np.asarray(e["node_id"])[pi]
    tags = list(e["tags"]); idx = {t: i for i, t in enumerate(tags)}
    need = {a: f"{a}_{TISSUE}_{CT}_{STATE}" for a in ("crohn", "uc", "healthy")}
    for a, t in need.items():
        if t not in idx:
            raise SystemExit(f"missing network {t} in {main_name}")
    ic, iu, ih = idx[need["crohn"]], idx[need["uc"]], idx[need["healthy"]]

    m = pres[ic] & pres[iu] & pres[ih]
    rc = Z[ic][m] - Z[ih][m]; ru = Z[iu][m] - Z[ih][m]
    cos = (rc * ru).sum(1) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(ru, axis=1) + 1e-9)
    genes = node_id[m]

    sc = {a: dict(zip(o.gene_symbol, o.score_indirect)) for a, f in OT.items()
          for o in [pd.read_csv(OT_DIR / f, sep="\t")]}
    oc = pd.Series(genes).map(sc["crohn"]).fillna(0).to_numpy()
    ou = pd.Series(genes).map(sc["uc"]).fillna(0).to_numpy()
    oa = pd.Series(genes).map(sc["alz"]).fillna(0).to_numpy()
    oi = pd.Series(genes).map(sc["ild"]).fillna(0).to_numpy()
    ibd = (oc > pos_thr) | (ou > pos_thr)                                  # IBD (Crohn/UC) targets > thr
    other = ((oa > pos_thr) | (oi > pos_thr)) & ~ibd                       # Alz/ILD targets > thr, NOT IBD (matched neg)
    cos_shared, cos_neg = cos[ibd], cos[other]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sns.kdeplot(cos_shared, ax=ax, fill=True, color=TOL["red"], alpha=0.45,
                label=f"IBD OT targets (>{pos_thr}), n={ibd.sum()}, med {np.median(cos_shared):.2f}")
    sns.kdeplot(cos_neg, ax=ax, fill=True, color=TOL["grey"], alpha=0.4,
                label=f"other-disease targets (Alz/ILD >{pos_thr}), n={other.sum()}, med {np.median(cos_neg):.2f}")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.axvline(np.median(cos_shared), color=TOL["red"], ls="--", lw=1)
    ax.axvline(np.median(cos_neg), color=TOL["grey"], ls="--", lw=1)
    ax.set_xlabel("cos(r$_{crohn}$, r$_{uc}$)   (r = disease − healthy, colon macrophage inflammatory)")
    ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title(f"Crohn–UC co-movement: IBD targets vs other-disease targets (OT>{pos_thr}) — "
                 f"{main_name.split('_')[-1]}", fontsize=9.5)
    fig.tight_layout()
    out = res / "images"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "shared_target_cosine_kde.png")
    print(f"IBD targets n={int(ibd.sum())} median cos={np.median(cos_shared):.3f} | "
          f"other-disease targets n={int(other.sum())} median cos={np.median(cos_neg):.3f}")
    print(f"wrote {out/'shared_target_cosine_kde.png'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_context_contrastive")
    ap.add_argument("--pos-thr", type=float, default=0.50)
    ap.add_argument("--n-neg", type=int, default=100)   # unused (negatives = all other-disease targets)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.pos_thr, a.n_neg, a.seed))
