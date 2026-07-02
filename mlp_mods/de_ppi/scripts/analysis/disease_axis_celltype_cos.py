"""Cross-disease cosine at the CELL-TYPE level, tissue constraint RELAXED (exploratory).

disease_axis_decompose.py needs >=2 diseases in the SAME (tissue, celltype) -> only colon macrophage (Crohn/UC)
qualifies. This variant relaxes tissue: it pools each disease's nets of one CELL TYPE across whatever tissues it
appears in, and asks how aligned different diseases' perturbations are (pairwise disease cosine).

CONFOUND (state it): each disease sits in a different tissue (Crohn colon/ileum, UC colon, ILD lung), so the
cosine mixes a shared DISEASE response with shared cross-tissue CELL-TYPE identity. Each net is still centered on
its OWN (study, tissue) healthy arm, so per-net tissue baseline is removed -- but the residual disease directions
are compared ACROSS tissues, which own-study centering cannot equalize. Read as exploratory, not a clean result.

Per disease arm:
  r_arm[p] = mean over that arm's (study, tissue) nets of ( Z_disease(study,tissue)[p] - Z_healthy(study,tissue)[p] )
Then pairwise cosine over MOVING proteins (max-magnitude >= move_pct percentile), present in both arms.

Output (results/<main>/disease_axis/): celltype_cross_disease_cos.tsv (pairwise) + prints the arm composition.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/disease_axis_celltype_cos.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr --celltype macrophage --arms crohn,uc,ild
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from disease_axis_decompose import parse                            # same tag parser


def main(main_name, ct, arms, move_pct) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres = c["Z"][:, pi, :], c["present"][:, pi]
    node_id = np.asarray(c["node_id"])[pi]
    tags = list(c["tags"]); idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}

    def is_primary(q, arm):                                         # allstates, non-loo, non-split net of (arm, ct)
        return (q["arm"] == arm and q["ct"] == ct and q["state"] == "allstates"
                and not q["loo"] and q["split"] is None)

    # healthy (study, tissue) -> tag, for own-study+own-tissue centering
    H = {(P[t]["study"], P[t]["tissue"]): t for t in tags if is_primary(P[t], "healthy")}

    # per-arm healthy-centered shift on the FULL universe, one entry per (study, tissue) with a healthy match
    perkey, arm_keys = {}, {a: [] for a in arms}
    for t in tags:
        q = P[t]
        for a in arms:
            if is_primary(q, a) and (q["study"], q["tissue"]) in H:
                hi = idx[H[(q["study"], q["tissue"])]]
                perkey[(a, q["study"], q["tissue"])] = (idx[t], hi)
                arm_keys[a].append((q["study"], q["tissue"]))

    for a in arms:
        if not arm_keys[a]:
            raise SystemExit(f"no {a} {ct} allstates net with a matching own (study,tissue) healthy")
        comp = ", ".join(f"{s}/{ti}" for s, ti in arm_keys[a])
        print(f"  {a:6s} {ct}: {len(arm_keys[a])} net(s)  [{comp}]")

    def arm_r(a, mask):                                             # mean healthy-centered shift for arm a on mask
        rs = [Z[di][mask] - Z[hi][mask] for (di, hi) in (perkey[(a, s, ti)] for s, ti in arm_keys[a])]
        return np.mean(rs, axis=0)

    rows = []
    for a, b in itertools.combinations(arms, 2):
        m = np.ones(Z.shape[1], bool)                              # proteins present in every net of both arms + healthy refs
        for arm in (a, b):
            for s, ti in arm_keys[arm]:
                di, hi = perkey[(arm, s, ti)]
                m &= pres[di] & pres[hi]
        ra, rb = arm_r(a, m), arm_r(b, m)
        na, nb = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1)
        move = np.maximum(na, nb); moving = move >= np.percentile(move, move_pct)
        pcos = (ra * rb).sum(1) / (na * nb + 1e-9)
        rows.append(dict(arm_a=a, arm_b=b, n_proteins=int(m.sum()), n_moving=int(moving.sum()),
                         mean_disease_cos_moving=round(float(pcos[moving].mean()), 4),
                         mean_disease_cos_all=round(float(pcos.mean()), 4)))
    out = res / "disease_axis"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"celltype_cross_disease_cos_{ct}.tsv", sep="\t", index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nwrote {out / f'celltype_cross_disease_cos_{ct}.tsv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--arms", default="crohn,uc,ild")
    ap.add_argument("--move-pct", type=float, default=50.0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.celltype, [x for x in a.arms.split(",") if x], a.move_pct))
