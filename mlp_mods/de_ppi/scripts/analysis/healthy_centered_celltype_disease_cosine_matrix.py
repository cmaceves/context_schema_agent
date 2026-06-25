"""Healthy-centered cosine matrix: SECOND version of write_celltype_disease_cosine_tsv.py centering on
the healthy reference protein embedding (default healthy_pinnacle_macrophage) instead of the 7-group
MEAN. Scoped to MACROPHAGE disease groups (mac/crohn, mac/ild, mac/uc) since the origin is macrophage.
  v[g,p] = mean over group-g macrophage networks of Z[.,p]
  shift[g,p] = v[g,p] - Z_ref[p]   (proteins present in all groups AND the reference)
  cell[A,B] = mean_p cos(shift_A[p], shift_B[p])
Output: tables/healthy_centered_celltype_disease_cosine_matrix.tsv

Run: .venv/bin/python mlp_mods/de_ppi/influence_analysis/write_celltype_disease_cosine_tsv_healthy.py \
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
import numpy as np, pandas as pd
DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_celltype, tag_disease
SHORT = {"macrophage": "mac"}

def main(out_name, ref_tag="healthy_pinnacle_macrophage"):
    d = np.load(DE_PPI / "results" / out_name / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    isp = d["node_type"] == "protein"
    assert ref_tag in tags, f"{ref_tag} not embedded"
    ri = tags.index(ref_tag)
    # macrophage disease-state networks only
    use = [(i, t) for i, t in enumerate(tags)
           if tag_celltype(t) == "macrophage" and "healthy" not in t and "split" not in t
           and not t.endswith(("_s1","_s2","_s3")) and "loo" not in t]
    groups = {}
    for i, t in use:
        groups.setdefault(f"mac/{tag_disease(t)}", []).append(i)
    labels = sorted(groups)
    P, dim = Z.shape[1], Z.shape[2]
    V = np.full((len(labels), P, dim), np.nan)
    for gi, g in enumerate(labels):
        acc = np.zeros((P, dim)); cnt = np.zeros(P)
        for n in groups[g]:
            m = present[n] & isp; acc[m] += Z[n, m]; cnt += m
        ok = cnt > 0; V[gi, ok] = acc[ok] / cnt[ok, None]
    H = Z[ri]; presH = present[ri] & isp
    # proteins present in ALL groups AND the healthy reference
    common = (~np.isnan(V).any(2).any(0)) & presH
    shift = V[:, common, :] - H[common][None]                    # healthy-centered
    nz = (np.linalg.norm(shift, axis=2) > 1e-9).all(0)           # drop zero-norm shifts (unstable angle)
    shift = shift[:, nz]
    n = int(nz.sum())
    unit = shift / np.linalg.norm(shift, axis=2, keepdims=True)
    agree = np.einsum("aps,bps->ab", unit, unit) / n
    short = [f"{SHORT.get(g.split('/')[0],g.split('/')[0])}/{g.split('/')[1]}" for g in labels]
    M = pd.DataFrame(np.round(agree, 2), index=short, columns=short); M.index.name = "group"
    out_dir = DE_PPI / "results" / out_name / "tables"; out_dir.mkdir(parents=True, exist_ok=True)
    M.to_csv(out_dir / "healthy_centered_celltype_disease_cosine_matrix.tsv", sep="\t")
    print(f"healthy origin={ref_tag} | {n} proteins common to all {len(labels)} macrophage groups + ref")
    print(M.to_string())
    print(f"\nwrote {out_dir/'healthy_centered_celltype_disease_cosine_matrix.tsv'}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--ref-tag", default="healthy_pinnacle_macrophage")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.ref_tag))
