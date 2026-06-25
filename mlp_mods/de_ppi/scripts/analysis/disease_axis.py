"""Shared-disease-axis test, baseline-free (no healthy reference needed).

For each disease we build a mean-shift axis using the per-protein CONSENSUS across diseases as the
reference instead of a healthy baseline:

  v[d,p]      = mean over disease-d networks of Z[n,p]            (protein p's position in disease d)
  consensus_p = mean over diseases of v[d,p]                      (equal weight per disease)
  axis[d]     = mean_p ( v[d,p] - consensus_p )                   (disease d's distinctive direction)

Then the pairwise cosine between the 4 axis[d] vectors tests whether diseases share a common
direction (a pan-disease axis) or move in private directions. Proteins are restricted to those
present in ALL diseases so the consensus is balanced. Healthy and donor-split networks are excluded
(consistent with the factor_combinations comparison set).

Output: results/<out_name>/disease_axis.tsv  (long form: one row per disease pair i<=j;
diagonal rows carry that disease's axis norm).

Run (.venv):
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/disease_axis.py \
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
    node_type = d["node_type"]
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    isp = node_type == "protein"

    # exclude donor-split duplicates and healthy baselines -> only disease-state networks
    use = [(i, t) for i, t in enumerate(tags) if "split" not in t and not t.startswith("healthy")]
    by_disease: dict[str, list[int]] = {}
    for i, t in use:
        by_disease.setdefault(tag_disease(t), []).append(i)
    diseases = sorted(by_disease)
    print(f"{len(use)} networks over {len(diseases)} diseases: "
          + ", ".join(f"{dx}({len(by_disease[dx])})" for dx in diseases), flush=True)

    # v[d, p] = mean over disease-d networks where p is present; mask proteins absent in a disease
    P, dim = Z.shape[1], Z.shape[2]
    V = np.full((len(diseases), P, dim), np.nan)
    for di, dx in enumerate(diseases):
        idx = by_disease[dx]
        pres = present[idx] & isp                      # [n_net_d, P]
        cnt = pres.sum(0)                               # proteins present in >=1 net of this disease
        acc = np.zeros((P, dim))
        for n in idx:
            m = present[n] & isp
            acc[m] += Z[n, m]
        ok = cnt > 0
        V[di, ok] = acc[ok] / cnt[ok, None]

    # proteins present in ALL diseases -> balanced consensus
    common = ~np.isnan(V).any(axis=2).any(axis=0)
    Vc = V[:, common, :]                                # [n_disease, n_common, dim]
    n_common = int(common.sum())
    print(f"{n_common} proteins present in all {len(diseases)} diseases", flush=True)

    consensus = Vc.mean(axis=0)                         # [n_common, dim]
    axes = (Vc - consensus[None]).mean(axis=1)          # [n_disease, dim]  disease distinctive direction
    norms = np.linalg.norm(axes, axis=1)
    unit = axes / norms[:, None]
    cos = unit @ unit.T                                 # pairwise cosine between disease axes

    rows = []
    for i in range(len(diseases)):
        for j in range(i, len(diseases)):
            rows.append({
                "disease_a": diseases[i],
                "disease_b": diseases[j],
                "cosine": round(float(cos[i, j]), 4),
                "axis_norm_a": round(float(norms[i]), 4),
                "axis_norm_b": round(float(norms[j]), 4),
                "n_proteins": n_common,
            })
    out = res / "tables" / "disease_axis.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)

    print("\npairwise cosine between disease axes (diagonal = 1; norm in []):")
    print("          " + "".join(f"{dx:>10}" for dx in diseases))
    for i, dx in enumerate(diseases):
        line = "".join(f"{cos[i, j]:>10.3f}" for j in range(len(diseases)))
        print(f"{dx:>8}[{norms[i]:.2f}] {line}")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
