"""Per-protein agreement between cell_type/disease groups (mean-of-cosines).

Identical to disease_protein_cosine.py but the grouping unit is (cell_type, disease) instead of just
disease, so each label is e.g. "macrophage/crohn":

  v[g,p]      = mean over group-g networks of Z[n,p]      (g = cell_type/disease)
  consensus_p = mean over groups of v[g,p]
  shift[g,p]  = v[g,p] - consensus_p
  agreement[A,B] = mean_p cos( shift[A,p], shift[B,p] )

Proteins present in ALL groups (so consensus is balanced); healthy/donor-split excluded.

Output: results/<out_name>/disease_celltype_protein_cosine.tsv  (long form, one row per group pair i<=j).

Run (.venv):
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/disease_celltype_protein_cosine.py \
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
from _layout import tag_celltype, tag_disease


def main(out_name) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    isp = d["node_type"] == "protein"

    use = [(i, t) for i, t in enumerate(tags) if "split" not in t and not t.startswith("healthy")]
    groups: dict[str, list[int]] = {}
    for i, t in use:
        groups.setdefault(f"{tag_celltype(t)}/{tag_disease(t)}", []).append(i)
    labels = sorted(groups)
    print(f"{len(use)} networks over {len(labels)} cell_type/disease groups:\n  "
          + ", ".join(f"{g}({len(groups[g])})" for g in labels), flush=True)

    P, dim = Z.shape[1], Z.shape[2]
    V = np.full((len(labels), P, dim), np.nan)
    for gi, g in enumerate(labels):
        acc = np.zeros((P, dim)); cnt = np.zeros(P)
        for n in groups[g]:
            m = present[n] & isp; acc[m] += Z[n, m]; cnt += m
        ok = cnt > 0; V[gi, ok] = acc[ok] / cnt[ok, None]

    common = ~np.isnan(V).any(axis=2).any(axis=0)
    Vc = V[:, common, :]
    shift = Vc - Vc.mean(axis=0)[None]
    n_common = int(common.sum())
    print(f"{n_common} proteins present in all {len(labels)} groups", flush=True)

    unit = shift / np.linalg.norm(shift, axis=2, keepdims=True)
    agree = np.einsum("aps,bps->ab", unit, unit) / n_common

    rows = []
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            rows.append({"group_a": labels[i], "group_b": labels[j],
                         "mean_protein_cosine": round(float(agree[i, j]), 4),
                         "n_proteins": n_common})
    out = res / "tables" / "disease_celltype_protein_cosine.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}  ({len(rows)} group pairs)", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
