"""Are OT>0.5 Crohn targets 'further out' across contexts than matched controls?

Per protein: pairwise distances between its embeddings in every context it appears in -> mean_pairwise and
max_pairwise (max = the single furthest-out context, the TNF phenomenon). Compare OT>0.5 targets to control
proteins (no OT association in ANY of the 4 diseases) MATCHED one-to-one on version count (nver), via a matched
permutation test (removes the nver confound). Run on the coupled build and the uncoupled scvi build.

Run: .venv/bin/python mlp_mods/de_ppi/results/random_walk_control/ot_context_spread_vs_control.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

RES = Path("mlp_mods/de_ppi/results")
OT_DIR = Path("mlp_mods/opentargets_associations")
OT_FILES = {"crohn": "crohn_target_association_EFO_0000384.tsv", "uc": "uc_target_association_EFO_0000729.tsv",
            "ild": "ild_target_association_EFO_0004244.tsv", "alz": "alzheimer_target_association_MONDO_0004975.tsv"}
BUILDS = {"coupled": "crohn_alzheimer_ild_uc_embedding_protein_linked",
          "uncoupled_scvi": "crohn_alzheimer_ild_uc_embedding_expressed_scvi"}
RNG = np.random.default_rng(0); B = 5000


def per_protein(build):
    z = np.load(RES / build / "embeddings.npz", allow_pickle=True)
    Z, pres = z["Z"], z["present"]; order = list(z["node_id"])
    rows = {}
    for i, g in enumerate(order):
        tis = np.where(pres[:, i])[0]
        if len(tis) < 2:
            continue
        V = Z[tis, i, :]; D = np.linalg.norm(V[:, None] - V[None], axis=2)
        iu = np.triu_indices(len(tis), 1)
        rows[g] = (len(tis), float(D[iu].mean()), float(D[iu].max()))
    return rows  # g -> (nver, mean_pairwise, max_pairwise)


def matched_test(metric, ot_targets, controls, by_nver):
    ot_vals = np.array([metric[g] for g in ot_targets]); ot_med = np.median(ot_vals)
    usable = [g for g in ot_targets if len(by_nver.get(nver_of[g], [])) >= 1]
    ctrl_meds = []
    for _ in range(B):
        samp = [metric[RNG.choice(by_nver[nver_of[g]])] for g in usable]
        ctrl_meds.append(np.median(samp))
    ctrl_meds = np.array(ctrl_meds)
    p = float((ctrl_meds >= ot_med).mean())            # one-sided: P(matched control median >= OT median)
    return ot_med, float(ctrl_meds.mean()), p, len(usable)


any_target = set()
for f in OT_FILES.values():
    d = pd.read_csv(OT_DIR / f, sep="\t"); any_target |= set(d[d.score_indirect > 0].gene_symbol)
crohn_ot = dict(zip(*[pd.read_csv(OT_DIR / OT_FILES["crohn"], sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))

for label, build in BUILDS.items():
    m = per_protein(build)
    nver_of = {g: v[0] for g, v in m.items()}
    ot_targets = [g for g in m if crohn_ot.get(g, 0) > 0.5]
    controls = [g for g in m if g not in any_target]
    by_nver = {}
    for g in controls:
        by_nver.setdefault(nver_of[g], []).append(g)
    print(f"\n===== build: {label}  ({build}) =====")
    print(f"OT>0.5 targets with >=2 contexts: {len(ot_targets)} | control pool: {len(controls)}")
    for name, idx in [("mean_pairwise", 1), ("max_pairwise", 2)]:
        metric = {g: v[idx] for g, v in m.items()}
        ot_med, ctrl_med, p, nuse = matched_test(metric, ot_targets, controls, by_nver)
        verdict = "OT further out" if ot_med > ctrl_med else "OT NOT further out"
        print(f"  {name:14s}: OT median={ot_med:.3f}  matched-control median={ctrl_med:.3f}  "
              f"perm p={p:.3f}  (n_matched={nuse})  -> {verdict}")
    # per-OT: furthest-out context
    print("  per-target max_pairwise (furthest context pair):")
    Z = np.load(RES / build / "embeddings.npz", allow_pickle=True)
    tags = [str(t) for t in Z["tags"]]; order = list(Z["node_id"]); pres = Z["present"]; Zar = Z["Z"]
    for g in sorted(ot_targets, key=lambda g: -m[g][2])[:9]:
        i = order.index(g); tis = np.where(pres[:, i])[0]; V = Zar[tis, i, :]
        D = np.linalg.norm(V[:, None] - V[None], axis=2); a, b = np.unravel_index(D.argmax(), D.shape)
        sh = lambda t: "_".join([tags[t].split("_")[0], tags[t].split("_")[1]] + tags[t].split("_")[3:])
        print(f"    {g:8s} nver={m[g][0]:2d} max={m[g][2]:.2f}  [{sh(tis[a])} <-> {sh(tis[b])}]")
