"""Shared drug / target / mechanism couplings for each 1-vs-1 disease pair (OpenTargets known drugs),
annotated with the modeled network CONTEXTS each shared target appears in.

A "coupling" = a (drug, target gene, mechanism-of-action) triple. For every pair of diseases we emit one
row per coupling that appears in BOTH diseases. Then, for the shared target gene, we list which of OUR
disease-state networks contain it as a node (node set = expressed ∪ DE), separately for disease_a and
disease_b — each context written as cell_type/tissue/state. This shows whether the shared drug target is
actually present in the contexts we modeled for each disease (and in which cell types/states).

CAVEAT: an empty context list means the target is not a node in any modeled network for that disease —
which can be either "not expressed there" OR "we didn't model that disease's relevant cell type"
(coverage is uneven: UC = macrophage/colon only; Alzheimer = microglia+fibroblast/brain; etc.). So
absence here is NOT evidence the target is irrelevant to the disease.

Output: results/<out_name>/influence_analysis/tables/shared_drug_target_pairs.tsv
  columns: disease_a, disease_b, drug_name, shared_target, shared_mechanism, phase_a, phase_b,
           a_contexts, b_contexts, n_a_contexts, n_b_contexts

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/dump_shared_drug_targets.py \
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
from itertools import combinations
from pathlib import Path

import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
from _layout import tag_celltype, tag_tissue, tag_disease, tag_state
REPO = DE_PPI.parents[1]
OT = REPO / "mlp_mods/03_opentargets_rebuild"
DISEASES = {"Crohn": "EFO_0000384", "UC": "EFO_0000729",
            "ILD": "EFO_0004244", "Alzheimer": "MONDO_0004975"}
DIS_CODE = {"Crohn": "crohn", "UC": "uc", "ILD": "ild", "Alzheimer": "alz"}   # OT name -> tag_disease code


def load(efo):
    """(gene, drug, mechanism) -> max clinical phase across that disease's known-drug rows."""
    df = pd.read_csv(OT / f"known_drugs_{efo}.tsv", sep="\t")
    df = df.dropna(subset=["gene_symbol", "drug_name", "mechanism_of_action"])
    return (df.groupby(["gene_symbol", "drug_name", "mechanism_of_action"])["phase"].max().to_dict())


def gene_contexts(out_name):
    """gene -> {disease_code: ['cell_type/tissue/state', ...]} over disease-state networks (no healthy/splits)."""
    net = DE_PPI / "results" / out_name / "networks"
    g2: dict = {}
    for d in sorted(net.iterdir()):
        tag = d.name
        if tag.startswith("healthy_") or "split" in tag or not (d / "network_nodes.tsv").exists():
            continue
        ctx = f"{tag_celltype(tag)}/{tag_tissue(tag)}/{tag_state(tag)}"
        code = tag_disease(tag)
        for nid in pd.read_csv(d / "network_nodes.tsv", sep="\t", usecols=["node_id"])["node_id"]:
            g2.setdefault(nid, {}).setdefault(code, []).append(ctx)
    return g2


def main(out_name) -> int:
    phases = {name: load(efo) for name, efo in DISEASES.items()}
    g2 = gene_contexts(out_name)

    def ctx_for(gene, dis_name):
        return sorted(set(g2.get(gene, {}).get(DIS_CODE[dis_name], [])))

    rows = []
    for a, b in combinations(DISEASES, 2):
        shared = set(phases[a]) & set(phases[b])
        for (gene, drug, mech) in shared:
            ca, cb = ctx_for(gene, a), ctx_for(gene, b)
            rows.append((a, b, drug, gene, mech, phases[a][(gene, drug, mech)], phases[b][(gene, drug, mech)],
                         "; ".join(ca), "; ".join(cb), len(ca), len(cb)))
    out = (DE_PPI / "results" / out_name / "tables")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["disease_a", "disease_b", "drug_name", "shared_target",
                                     "shared_mechanism", "phase_a", "phase_b",
                                     "a_contexts", "b_contexts", "n_a_contexts", "n_b_contexts"])
    df = df.sort_values(["disease_a", "disease_b", "phase_a", "shared_target", "drug_name"],
                        ascending=[True, True, False, True, True]).reset_index(drop=True)
    fn = out / "shared_drug_target_pairs.tsv"
    df.to_csv(fn, sep="\t", index=False)
    print(f"wrote {fn}  ({len(df)} shared coupling rows)\n")
    print("rows per disease pair:")
    print(df.groupby(["disease_a", "disease_b"]).size().to_string())
    print(f"\nrows where target present in >=1 context in BOTH diseases: "
          f"{int(((df.n_a_contexts > 0) & (df.n_b_contexts > 0)).sum())} / {len(df)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
