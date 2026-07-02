"""Pairwise healthy-centered disease-shift COSINE matrix over every disease context (arm, tissue, celltype).

Each disease context = an (arm, tissue, celltype) allstates net (studies pooled, each centered on its OWN
(study,tissue,celltype) healthy arm). Cell = mean per-protein cos(shift_i, shift_j) over proteins present in
BOTH and MOVING (max ||shift|| >= move_pct percentile, on the pair's common set). Diagonal = 1.

CONFOUND: off-diagonal cells mix disease, tissue, AND cell type -- own-arm centering removes each context's own
baseline but cannot equalize tissue/celltype BETWEEN contexts. Only same-(tissue,celltype) pairs (e.g. colon
macrophage crohn vs uc) are clean disease contrasts; the rest are exploratory. Cross-celltype pairs are scored
over the OmniPath-protein intersection of their (different) node sets.

Output (results/<main>/disease_axis/): context_cosine_matrix.tsv (square, moving-filtered).

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/disease_axis_context_matrix.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from disease_axis_decompose import parse

DISEASES = {"crohn", "uc", "alz", "ild"}


def main(main_name, move_pct) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres = c["Z"][:, pi, :], c["present"][:, pi]
    tags = list(c["tags"]); idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}

    def primary(q, arm):
        return q["arm"] == arm and q["state"] == "allstates" and not q["loo"] and q["split"] is None

    H = {(P[t]["study"], P[t]["tissue"], P[t]["ct"]): t for t in tags if primary(P[t], "healthy")}

    # disease context -> list of (disease_tag_idx, healthy_tag_idx) over its (study) at that tissue/celltype
    ctx = {}
    for t in tags:
        q = P[t]
        if q["arm"] in DISEASES and primary(q, q["arm"]):
            key = (q["study"], q["tissue"], q["ct"])
            if key in H:
                ctx.setdefault((q["arm"], q["tissue"], q["ct"]), []).append((idx[t], idx[H[key]]))
    labels = sorted(ctx)                                            # (arm, tissue, ct)

    shift, present = {}, {}                                         # full-universe mean healthy-centered shift + present mask
    for k, pairs in ctx.items():
        shift[k] = np.mean([Z[di] - Z[hi] for di, hi in pairs], axis=0)
        pm = np.ones(Z.shape[1], bool)
        for di, hi in pairs:
            pm &= pres[di] & pres[hi]
        present[k] = pm

    n = len(labels)
    M = np.full((n, n), np.nan)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            a, b = labels[i], labels[j]
            m = present[a] & present[b]
            ra, rb = shift[a][m], shift[b][m]
            na, nb = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1)
            keep = np.maximum(na, nb) >= np.percentile(np.maximum(na, nb), move_pct)
            cos = float(((ra * rb).sum(1) / (na * nb + 1e-9))[keep].mean()) if keep.any() else np.nan
            M[i, j] = M[j, i] = round(cos, 4)

    names = ["/".join(k) for k in labels]
    df = pd.DataFrame(M, index=names, columns=names)
    out = res / "disease_axis"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "context_cosine_matrix.tsv", sep="\t")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string())
    print(f"\nwrote {out / 'context_cosine_matrix.tsv'}  (moving-filtered, move_pct={move_pct})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--move-pct", type=float, default=50.0)
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.move_pct))
