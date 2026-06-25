"""Per-protein agreement between diseases (mean-of-cosines, not cosine-of-means).

Same consensus-centered shift as disease_axis.py:
  v[d,p]      = mean over disease-d networks of Z[n,p]
  consensus_p = mean over diseases of v[d,p]
  shift[d,p]  = v[d,p] - consensus_p

For each disease pair (A,B) we cosine EACH protein's two shift vectors and average over proteins:
  agreement[A,B] = mean_p cos( shift[A,p], shift[B,p] )

This asks "do individual proteins move the same way in A and B", unlike disease_axis.py which cosines
the aggregate axis vectors. Proteins present in all diseases; healthy/donor-split excluded.

Output: results/<out_name>/disease_protein_cosine.tsv  (long form, one row per disease pair i<=j).

Run (.venv):
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/disease_protein_cosine.py \
      --out-name crohn_alzheimer_ild_uc_embedding_expressed
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_disease


def main(out_name) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    isp = d["node_type"] == "protein"

    use = [(i, t) for i, t in enumerate(tags) if "split" not in t and not t.startswith("healthy")]
    by_disease: dict[str, list[int]] = {}
    for i, t in use:
        by_disease.setdefault(tag_disease(t), []).append(i)
    diseases = sorted(by_disease)

    P, dim = Z.shape[1], Z.shape[2]
    V = np.full((len(diseases), P, dim), np.nan)
    for di, dx in enumerate(diseases):
        acc = np.zeros((P, dim)); cnt = np.zeros(P)
        for n in by_disease[dx]:
            m = present[n] & isp; acc[m] += Z[n, m]; cnt += m
        ok = cnt > 0; V[di, ok] = acc[ok] / cnt[ok, None]

    common = ~np.isnan(V).any(axis=2).any(axis=0)
    Vc = V[:, common, :]                                  # [n_disease, n_common, dim]
    shift = Vc - Vc.mean(axis=0)[None]                    # consensus-centered shift per protein
    n_common = int(common.sum())
    print(f"{n_common} proteins present in all {len(diseases)} diseases", flush=True)

    unit = shift / np.linalg.norm(shift, axis=2, keepdims=True)   # per-protein unit shift vectors
    # mean over proteins of cos(shift[A,p], shift[B,p]) = mean over p of <unit_A, unit_B>
    agree = np.einsum("aps,bps->ab", unit, unit) / n_common

    rows = []
    for i in range(len(diseases)):
        for j in range(i, len(diseases)):
            rows.append({"disease_a": diseases[i], "disease_b": diseases[j],
                         "mean_protein_cosine": round(float(agree[i, j]), 4),
                         "n_proteins": n_common})
    out = res / "tables" / "disease_protein_cosine.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)

    print("\nmean per-protein cosine between diseases (diagonal = 1):")
    print("        " + "".join(f"{dx:>9}" for dx in diseases))
    for i, dx in enumerate(diseases):
        print(f"{dx:>7} " + "".join(f"{agree[i, j]:>9.3f}" for j in range(len(diseases))))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
