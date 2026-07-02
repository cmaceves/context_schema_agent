"""Control m: HEALTHY leave-one-study-out. Pair each LOO pool (N-1 studies) with its held-out single study
and score agreement vs the MAIN consensus, the same way as the other controls:

  for each (tissue,celltype,state,held-out g):
    pool   = healthy_loopool<g>_<tissue>_<celltype>_<state>   (cells of all study groups except g)
    single = healthy_loosingle<g>_<tissue>_<celltype>_<state> (held-out study g)
  mean_prot_cos = mean over shared present proteins of cos(Z_pool-consensus, Z_single-consensus)

Interpretation: compare control m's cosine to control i (single-vs-single between-study, ~0.49) and the
within-study ceiling (control h, ~0.74). If pooling reduces the between-study batch, m > i and climbs toward h
(the held-out study agrees better with a POOL of others than with any single other study).

Networks built by build_healthy_loo.py + inferred via infer_controls.py.
Output (controls/): control_m_pairs.tsv  (+ printed summary).
Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/control_m_healthy_loo.py --main-name <build>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_loo(tag):
    """healthy_loo{pool|single}<g8>_<tissue>_<celltype>_<state> -> (kind, g8, tissue, celltype, state) or None."""
    p = tag.split("_")
    if len(p) < 5 or p[0] != "healthy" or not (p[1].startswith("loopool") or p[1].startswith("loosingle")):
        return None
    kind = "loopool" if p[1].startswith("loopool") else "loosingle"
    g8 = p[1][len(kind):]
    return kind, g8, p[2], p[3], "_".join(p[4:])


def main(main_name="crohn_alzheimer_ild_uc_embedding_expressed") -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    m = np.load(res / "embeddings.npz", allow_pickle=True)
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    assert list(m["node_id"]) == list(c["node_id"])
    is_prot = m["node_type"] == "protein"; pi = np.where(is_prot)[0]
    Zm, presm = m["Z"][:, pi, :], m["present"][:, pi]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmean(np.where(presm[:, :, None], Zm, np.nan), axis=0)
    valid = ~np.isnan(consensus).any(axis=1)

    tags = list(c["tags"]); Zc, presc = c["Z"][:, pi, :], c["present"][:, pi]
    ti = {t: i for i, t in enumerate(tags)}
    pools, singles = {}, {}
    for t in tags:
        pr = parse_loo(t)
        if pr is None:
            continue
        kind, g8, tis, ct, state = pr
        (pools if kind == "loopool" else singles)[(g8, tis, ct, state)] = ti[t]

    R = Zc - consensus[None]; Rn = np.linalg.norm(R, axis=2)
    rows = []
    for key, pidx in pools.items():
        if key not in singles:
            continue
        sidx = singles[key]
        both = presc[pidx] & presc[sidx] & valid
        if both.sum() < 20:
            continue
        ra, rb = R[pidx, both], R[sidx, both]
        cos = float(((ra * rb).sum(1) / (Rn[pidx, both] * Rn[sidx, both] + 1e-9)).mean())
        shift = float(np.linalg.norm(Zc[pidx, both] - Zc[sidx, both], axis=1).mean())
        g8, tis, ct, state = key
        rows.append(dict(control="m", control_name="healthy_loo", held_out=g8, tissue=tis, celltype=ct,
                         state=state, n_proteins=int(both.sum()),
                         pool_vs_heldout_cos=round(cos, 4), shift=round(shift, 4)))
    df = pd.DataFrame(rows)
    out = res / "controls"; df.to_csv(out / "control_m_pairs.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_m_pairs.tsv'} ({len(df)} LOO pool-vs-heldout pairs)\n")
    if len(df):
        print(df.to_string(index=False))
        print(f"\nmean pool-vs-heldout cosine (control m) = {df.pool_vs_heldout_cos.mean():.4f}")
        print("compare to control i (single-vs-single between-study) and h (within-study ceiling):"
              " m > i means pooling studies reduced the between-study batch.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    raise SystemExit(main(ap.parse_args().main_name))
