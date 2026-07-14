"""For each OT>0.5 Crohn target, find its OUTLIER context (the context whose embedding is furthest from the
protein's other version-vectors) and ask whether it's the disease-relevant context (proxy: a disease arm, and/or
the inflammatory state) — as it is for TNF (crohn_colon_inflammatory). Compare that tendency to count-matched
control proteins (no OT in any disease). Coupled build only.

Run: .venv/bin/python mlp_mods/de_ppi/results/random_walk_control/ot_outlier_context.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

RES = Path("mlp_mods/de_ppi/results")
OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILES = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
            "ild": "ild_target_association_EFO_0004244.tsv", "alz": "alzheimer_target_association_MONDO_0004975.tsv"}
BUILD = "crohn_alzheimer_ild_uc_embedding_protein_linked"
RNG = np.random.default_rng(0); B = 5000


def main():
    z = np.load(RES / BUILD / "embeddings.npz", allow_pickle=True)
    Z, pres, tags = z["Z"], z["present"], [str(t) for t in z["tags"]]
    order = list(z["node_id"])
    arm = [t.split("_")[0] for t in tags]; state = ["_".join(t.split("_")[3:]) for t in tags]
    sh = lambda t: "_".join([tags[t].split("_")[0], tags[t].split("_")[1]] + tags[t].split("_")[3:])

    # per protein: outlier context = argmax mean distance to its other versions
    outlier = {}      # g -> (n_ctx, outlier_tag_idx, is_disease, is_inflammatory)
    for i, g in enumerate(order):
        tis = np.where(pres[:, i])[0]
        if len(tis) < 2:
            continue
        V = Z[tis, i, :]; D = np.linalg.norm(V[:, None] - V[None], axis=2)
        mean_to_others = D.sum(1) / (len(tis) - 1)
        o = tis[int(mean_to_others.argmax())]
        outlier[g] = (len(tis), o, arm[o] != "healthy", state[o] == "inflammatory")

    any_target = set()
    for f in OT_FILES.values():
        d = pd.read_csv(OT_DIR / f, sep="\t"); any_target |= set(d[d.score_indirect > 0].gene_symbol)
    crohn_ot = dict(zip(*[pd.read_csv(OT_DIR / OT_FILES["crohn"], sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))
    ot_targets = [g for g in outlier if crohn_ot.get(g, 0) > 0.5]
    controls = [g for g in outlier if g not in any_target]
    nver_of = {g: outlier[g][0] for g in outlier}
    by_nver = {}
    for g in controls:
        by_nver.setdefault(nver_of[g], []).append(g)

    print(f"OT>0.5 targets (>=2 ctx): {len(ot_targets)} | control pool: {len(controls)}\n")
    print("per-OT outlier context (the 'far out' one):")
    print(f"  {'gene':8s} {'nctx':>4s} {'outlier context':28s} {'disease?':>8s} {'inflam?':>7s}")
    for g in sorted(ot_targets, key=lambda g: -crohn_ot[g]):
        n, o, dis, inf = outlier[g]
        print(f"  {g:8s} {n:4d} {sh(o):28s} {'yes' if dis else 'no':>8s} {'yes' if inf else 'no':>7s}")

    ot_dis = np.mean([outlier[g][2] for g in ot_targets]); ot_inf = np.mean([outlier[g][3] for g in ot_targets])
    def perm(idx):
        obs = np.mean([outlier[g][idx] for g in ot_targets])
        usable = [g for g in ot_targets if by_nver.get(nver_of[g])]
        fr = [np.mean([outlier[RNG.choice(by_nver[nver_of[g]])][idx] for g in usable]) for _ in range(B)]
        fr = np.array(fr); return obs, fr.mean(), float((fr >= obs).mean())
    od, cd, pd_ = perm(2); oi, ci, pi = perm(3)
    print(f"\noutlier context is a DISEASE arm:      OT {od:.2f}  vs matched control {cd:.2f}  (perm p={pd_:.3f})")
    print(f"outlier context is INFLAMMATORY state: OT {oi:.2f}  vs matched control {ci:.2f}  (perm p={pi:.3f})")
    # base rates for reference: fraction of all contexts that are disease / inflammatory
    print(f"\nbase rate across the {len(tags)} contexts: disease={np.mean([a!='healthy' for a in arm]):.2f}, "
          f"inflammatory={np.mean([s=='inflammatory' for s in state]):.2f}")


if __name__ == "__main__":
    main()
