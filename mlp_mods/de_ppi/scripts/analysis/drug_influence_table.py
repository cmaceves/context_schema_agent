"""OpenTargets drugs for a build's disease, joined to network P^k influence.

Parameterized by --build <name>. For every drug associated with the build's disease
family in OpenTargets (diseaseId in the build's scope OR an in-scope id in the OT
`ancestors`; ALL clinical phases), map the drug's target gene(s) to their P^3
network-influence rank (results/<build>/P3_influence.tsv) and emit a per-drug table.

A drug gets one row. `influence_rank`/`influence_percentile` come from the drug's
MOST influential target present in the network (min rank); drugs whose targets are
all outside the network keep the row with an empty rank. Percentile is higher = more
influential (rank 1 ~ 100th percentile). The chosen target also carries look-only signed
diagnostics copied from P3_influence: `signed_rank`, `signed_percentile`,
`signed_influence`, `percent_edges_signed`. The drug ranking itself stays on REACH.

Output: de_ppi/results/<build>/influence_analysis/<disease_slug>_drug_influence.tsv

Run with .venv:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/drug_influence_table.py --build macrophage_crohn
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("mlp_mods/de_ppi")))   # de_ppi on path
from config import load_build


def main(build: str) -> int:
    cfg = load_build(build)
    cfg.influence_dir.mkdir(parents=True, exist_ok=True)
    scope = cfg.drug_scope

    kd = pd.read_parquet(cfg.known_drugs_parquet)
    anc_hit = lambda a: isinstance(a, (list, np.ndarray)) and any(s in set(a) for s in scope)
    in_scope = kd[kd["diseaseId"].isin(scope) | kd["ancestors"].apply(anc_hit)].copy()
    print(f"{cfg.disease} known-drug rows (all phases): {len(in_scope)} | "
          f"unique drugs: {in_scope.drugId.nunique()}", flush=True)

    # network influence: protein -> (rank, percentile). Higher percentile = more influential.
    infl = pd.read_csv(cfg.p3_influence, sep="\t")
    n = len(infl)
    rank = dict(zip(infl.protein, infl["rank"]))
    pct = {g: round(100.0 * (1.0 - (r - 1) / n), 1) for g, r in rank.items()}
    # look-only signed diagnostics for the chosen target (present iff P3 was rebuilt with them)
    have_signed = {"signed_rank", "signed_percentile", "signed_influence",
                   "percent_edges_signed"}.issubset(infl.columns)
    sr = dict(zip(infl.protein, infl.signed_rank)) if have_signed else {}
    sp_ = dict(zip(infl.protein, infl.signed_percentile)) if have_signed else {}
    sinf = dict(zip(infl.protein, infl.signed_influence)) if have_signed else {}
    pes = dict(zip(infl.protein, infl.percent_edges_signed)) if have_signed else {}

    rows = []
    for drug_id, sub in in_scope.groupby("drugId"):
        targets = sorted({s for s in sub.approvedSymbol.dropna() if s})
        in_net = [(rank[g], g) for g in targets if g in rank]
        best_rank, best_target = min(in_net) if in_net else (None, "")
        indications = sorted({s for s in sub.label.dropna() if s})
        rows.append({
            "drug_name": sub.prefName.dropna().iloc[0] if sub.prefName.notna().any() else "",
            "drugId": drug_id,
            "drug_phase": float(sub.phase.max()),
            "indications": "; ".join(indications),
            "influence_rank": best_rank,
            "influence_percentile": pct.get(best_target, np.nan) if best_target else np.nan,
            "influence_target": best_target,
            "signed_rank": sr.get(best_target) if best_target in sr else None,
            "signed_percentile": sp_.get(best_target) if best_target in sp_ else np.nan,
            "signed_influence": round(sinf[best_target], 4) if best_target in sinf else np.nan,
            "percent_edges_signed": pes.get(best_target) if best_target in pes else np.nan,
            "n_targets": len(targets),
            "all_targets": "; ".join(targets),
            "drugType": sub.drugType.dropna().iloc[0] if sub.drugType.notna().any() else "",
            "mechanism_of_action": sub.mechanismOfAction.dropna().iloc[0] if sub.mechanismOfAction.notna().any() else "",
        })

    df = pd.DataFrame(rows)
    # most influential drug-targets first; drugs with no in-network target sink to the bottom
    df = df.sort_values("influence_rank", na_position="last").reset_index(drop=True)
    df = df[["drug_name", "drugId", "drug_phase", "indications", "influence_rank",
             "influence_percentile", "influence_target", "signed_rank", "signed_percentile",
             "signed_influence", "percent_edges_signed", "n_targets", "all_targets",
             "drugType", "mechanism_of_action"]]
    df["influence_rank"] = df["influence_rank"].astype("Int64")
    df["signed_rank"] = df["signed_rank"].astype("Int64")
    df.to_csv(cfg.drug_table, sep="\t", index=False)

    n_ranked = int(df.influence_rank.notna().sum())
    print(f"wrote {cfg.drug_table}  ({len(df)} drugs; {n_ranked} with an in-network target, "
          f"{len(df) - n_ranked} without)", flush=True)
    print("\ntop 15 drugs by target influence (reach), with signed diagnostics:")
    print(df.head(15)[["drug_name", "drug_phase", "influence_rank", "influence_percentile",
                       "influence_target", "signed_rank", "percent_edges_signed"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OpenTargets drugs joined to network influence")
    ap.add_argument("--build", default="macrophage_crohn")
    raise SystemExit(main(ap.parse_args().build))
