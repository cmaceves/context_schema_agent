"""Healthy-centered control floor: SECOND version of dump_control_comparison_meanprotcos.py centering on
the healthy reference protein embedding (default healthy_pinnacle_macrophage) instead of the all-network
MEAN. MACROPHAGE control pairs only (the origin is macrophage): crohn_mac donor_split + between_study.
  r_X[p]=Z_X[p]-Z_ref[p];  mean_prot_cos(A,B)=mean_p cos(r_A,r_B) over proteins present in A,B AND ref.
Output: tables/healthy_centered_control_comparison_meanprotcos_summary.tsv (+ per-pair).
Run: .venv/bin/python mlp_mods/de_ppi/influence_analysis/dump_healthy_centered_control_comparison_meanprotcos.py
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
import numpy as np, pandas as pd
DE_PPI = Path("mlp_mods/de_ppi")
CROHN_STATES = ["inflammatory", "resident", "proliferating"]

def main(out_name, ref_tag="healthy_pinnacle_macrophage"):
    res = DE_PPI / "results" / out_name
    (res / "tables").mkdir(parents=True, exist_ok=True); (res / "images").mkdir(parents=True, exist_ok=True)
    d = np.load(res / "embeddings.npz", allow_pickle=True)
    tags, Z, present = list(d["tags"]), d["Z"], d["present"]
    isp = d["node_type"] == "protein"
    assert ref_tag in tags, f"{ref_tag} not embedded"
    ri = tags.index(ref_tag); H = Z[ri]; presH = present[ri] & isp
    ti = {t: i for i, t in enumerate(tags)}; have = set(tags)
    rows = []
    def add(ctype, state, a, b):
        if a not in have or b not in have: return
        ia, ib = ti[a], ti[b]
        both = present[ia] & present[ib] & presH
        if both.sum() == 0: return
        ra, rb = Z[ia, both]-H[both], Z[ib, both]-H[both]
        na, nb = np.linalg.norm(ra,axis=1), np.linalg.norm(rb,axis=1)
        ok = (na>1e-9)&(nb>1e-9)
        cos = (ra[ok]*rb[ok]).sum(1)/(na[ok]*nb[ok])
        rows.append(dict(cell_type="macrophage", control_type=ctype, state=state, network_a=a, network_b=b,
                         n_proteins=int(ok.sum()), mean_prot_cos=round(float(cos.mean()),3)))
    for s in CROHN_STATES:
        add("donor_split", s, f"crohn_mac_{s}_splitA", f"crohn_mac_{s}_splitB")
        add("between_study", s, f"crohn_mac_{s}_s1", f"crohn_mac_{s}_s2")
    df = pd.DataFrame(rows)
    out = res/"tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"healthy_centered_control_comparison_meanprotcos.tsv", sep="\t", index=False)
    summ = (df.groupby(["cell_type","control_type","state"]).agg(n=("mean_prot_cos","size"),
            mean_prot_cos=("mean_prot_cos","mean")).round(3).reset_index())
    summ.to_csv(out/"healthy_centered_control_comparison_meanprotcos_summary.tsv", sep="\t", index=False)
    print(f"healthy origin={ref_tag} (macrophage control pairs only)")
    print(summ.to_string(index=False))
    print(f"\nwrote {out/'healthy_centered_control_comparison_meanprotcos_summary.tsv'}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc")
    ap.add_argument("--ref-tag", default="healthy_pinnacle_macrophage")
    a = ap.parse_args(); raise SystemExit(main(a.out_name, a.ref_tag))
