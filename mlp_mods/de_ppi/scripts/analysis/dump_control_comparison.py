"""Tabulate the embedding SHIFT (magnitude) and DIRECTION (cosine) differences between every control
network pair set up in results/crohn_alzheimer_ild_uc/ — so the donor-split floor, between-study batch,
and pair-vs-withheld can be read side by side, per disease/cell type.

Control types:
  donor_split       : same population, donors split in two within ONE study (noise floor)
  between_study     : same state, two/three independent single studies (study batch)
  pool_vs_pool      : two different (n-1)-study pools (sharing one study) vs each other = how stable the
                      pooled signal is to which studies are included (Alz only; the LOO stability test)

DIRECTION metric = min‖dev‖-weighted mean cosine of consensus-centered deviations (same as
plot_pair_direction_heatmap.py: dir_cos(A,B) = Σ cos(r_A,r_B)·min(‖r_A‖,‖r_B‖) / Σ min(...)). The
consensus is recomputed HERE over the MAIN (non-control) networks only — control replicates are
excluded so a control pair is never centered against a version of itself. SHIFT (magnitude) is
‖Z_A-Z_B‖ and does not depend on the centroid.

Writes tables/control_comparison.tsv (one row per control pair) and tables/control_comparison_summary.tsv
(mean per disease x control_type).

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/dump_control_comparison.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_tissue
CROHN_STATES = ["inflammatory", "resident", "proliferating"]
ALZ_STATES = ["dam", "homeostatic", "interferon", "proliferating"]


def main(out_name, main_build) -> int:
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    is_prot = d["node_type"] == "protein"

    # consensus over MAIN (non-control) networks only — exclude control replicates so a control pair is
    # never centered against a version of itself.
    main_tags = set(np.load(DE_PPI / "results" / main_build / "embeddings.npz", allow_pickle=True)["tags"])
    is_main = np.array([t in main_tags for t in tags])
    print(f"consensus over {int(is_main.sum())}/{len(tags)} MAIN networks "
          f"(excluded {int((~is_main).sum())} control tags)", flush=True)

    pi = np.where(is_prot)[0]
    Zp, pres = Z[:, pi, :], present[:, pi]
    contrib = pres & is_main[:, None]
    import warnings
    with warnings.catch_warnings():                              # proteins in no main net -> all-nan slice
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmean(np.where(contrib[:, :, None], Zp, np.nan), axis=0)
    valid = ~np.isnan(consensus).any(axis=1)
    R = Zp - consensus[None]
    Rn = np.linalg.norm(R, axis=2)
    ti = {t: i for i, t in enumerate(tags)}
    have = set(tags)
    rows = []

    def add(dis, ct, ctype, state, a, b):
        if a not in have or b not in have:
            return
        ta, tb = tag_tissue(a), tag_tissue(b)
        assert ta == tb, f"tissue not controlled: {a}({ta}) vs {b}({tb})"       # guard: same tissue only
        ia, ib = ti[a], ti[b]
        both = pres[ia] & pres[ib] & valid
        if both.sum() == 0:
            return
        ra, rb = R[ia, both], R[ib, both]
        na, nb = Rn[ia, both], Rn[ib, both]
        cos = (ra * rb).sum(1) / (na * nb + 1e-9)
        w = np.minimum(na, nb)                                                   # min‖dev‖ weighting
        dir_cos = float((cos * w).sum() / (w.sum() + 1e-9))
        shift = float(np.linalg.norm(Zp[ia, both] - Zp[ib, both], axis=1).mean())
        rows.append(dict(disease=dis, cell_type=ct, tissue=ta, control_type=ctype, state=state,
                         network_a=a, network_b=b, n_proteins=int(both.sum()),
                         shift=round(shift, 3), dir_cos=round(dir_cos, 3)))

    # --- donor-split (within one study), every state ---
    for s in CROHN_STATES:
        add("Crohn", "macrophage", "donor_split", s, f"crohn_mac_{s}_splitA", f"crohn_mac_{s}_splitB")
    for s in ALZ_STATES:
        add("Alzheimer", "microglia", "donor_split", s, f"alz_microglia_{s}_splitA", f"alz_microglia_{s}_splitB")

    # --- between-study (single studies, same state) ---
    for s in CROHN_STATES:
        add("Crohn", "macrophage", "between_study", s, f"crohn_mac_{s}_s1", f"crohn_mac_{s}_s2")
    for s in ALZ_STATES:
        singles = [f"alz_microglia_{s}_{k}" for k in ("s1", "s2", "s3")]
        for a, b in combinations([t for t in singles if t in have], 2):
            add("Alzheimer", "microglia", "between_study", s, a, b)

    # (pool_vs_pool control removed; superseded by control m = healthy_loo)

    df = pd.DataFrame(rows)
    out = res / "tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "control_comparison.tsv", sep="\t", index=False)
    summ = (df.groupby(["disease", "cell_type", "tissue", "control_type", "state"])   # state + tissue kept separate
            .agg(n=("shift", "size"), shift_mean=("shift", "mean"), dir_cos_mean=("dir_cos", "mean"))
            .round(3).reset_index().sort_values(["disease", "state", "control_type"]))
    summ.to_csv(out / "control_comparison_summary.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_comparison.tsv'} ({len(df)} pairs)\nwrote {out/'control_comparison_summary.tsv'}\n")
    print("=== per-pair ==="); print(df.to_string(index=False))
    print("\n=== summary (mean per disease x control type) ==="); print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc")
    ap.add_argument("--main-build", default="crohn_alzheimer_ild_uc_embedding_expressed",
                    help="build whose tags define the MAIN (non-control) networks for the consensus")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.main_build))
