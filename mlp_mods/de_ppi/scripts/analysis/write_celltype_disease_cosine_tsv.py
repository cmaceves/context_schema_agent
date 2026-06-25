"""Write the mean per-protein cosine matrix between cell_type/disease groups to ./tables.

Recomputes (standalone) the consensus-centered per-protein shift agreement:
  v[g,p]      = mean over group-g networks of Z[n,p]   (g = cell_type/disease)
  consensus_p = mean over groups of v[g,p]
  shift[g,p]  = v[g,p] - consensus_p
  cell[A,B]   = mean_p cos(shift[A,p], shift[B,p])      over proteins present in ALL groups

Output: ./tables/celltype_disease_cosine_matrix.tsv  (full symmetric matrix, short labels).
Run (.venv, from repo root):
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/write_celltype_disease_cosine_tsv.py
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

SHORT = {"fibroblast": "fib", "macrophage": "mac", "microglia": "mic", "stem": "stem"}


def main(out_name) -> int:
    d = np.load(DE_PPI / "results" / out_name / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    isp = d["node_type"] == "protein"

    use = [(i, t) for i, t in enumerate(tags) if "split" not in t and not t.startswith("healthy")]
    groups: dict[str, list[int]] = {}
    for i, t in use:
        groups.setdefault(f"{tag_celltype(t)}/{tag_disease(t)}", []).append(i)
    if "healthy_pinnacle_macrophage" in tags:                 # include the healthy macrophage origin as a group
        groups["macrophage/healthy"] = [tags.index("healthy_pinnacle_macrophage")]
    labels = sorted(groups)

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
    unit = shift / np.linalg.norm(shift, axis=2, keepdims=True)
    agree = np.einsum("aps,bps->ab", unit, unit) / n_common      # mean per-protein cosine

    short = [f"{SHORT.get(g.split('/')[0], g.split('/')[0])}/{g.split('/')[1]}" for g in labels]
    M = pd.DataFrame(np.round(agree, 2), index=short, columns=short)
    M.index.name = "group"

    out_dir = DE_PPI / "results" / out_name / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "celltype_disease_cosine_matrix.tsv"
    M.to_csv(out, sep="\t")
    print(f"{n_common} proteins common to all {len(labels)} groups")
    print(M.to_string())
    print(f"\nwrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
