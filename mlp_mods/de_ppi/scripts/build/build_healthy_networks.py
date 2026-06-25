"""Build per-cell-type HEALTHY baseline networks (reference arm, neutral weights) and drop them
into an existing joint-embedding networks/ dir as new tags healthy_<celltype>.

A healthy network is the disease-free anchor for a cell type: the PINNACLE cell-type backbone with
OmniPath directed edges among it (the same topology rule as the disease builds, build_literature_
weighted_influence.py line 95/130), but with NO differential expression -> no DE-added proteins, no
literature/metabolite nodes, and every sender weight = 1.0 (zero rank-shift). So it captures the
cell type's wiring with no disease perturbation.

Writes, for each cell type, into results/<out_name>/networks/healthy_<celltype>/:
  network_nodes.tsv   (node_id, node_type, source=pinnacle, direction, sender_weight=1.0)
  network_edges.tsv   (source, target, edge_origin=OmniPath, edge_property, weight=1.0, direction)

Run:
  .venv/bin/python mlp_mods/de_ppi/build_healthy_networks.py --out-name crohn_alzheimer_ild_embedding
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

import pandas as pd

HERE = Path("mlp_mods/de_ppi")
sys.path.insert(0, str(HERE))
from config import load_build   # noqa: E402

# cell type -> representative build (only its celltype_ppi + omni paths are used; disease is irrelevant)
CELLTYPE_BUILD = {
    "macrophage": "macrophage_crohn",
    "microglia": "microglia_alzheimers",
    "fibroblast": "fibroblast_crohn",
    "stem": "stem_crohn",
}


def read_ppi_nodes(path: Path) -> set[str]:
    nodes: set[str] = set()
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                nodes.update(parts[:2])
    return nodes


def main(out_name: str) -> int:
    net_root = HERE / "results" / out_name / "networks"
    for ct, build in CELLTYPE_BUILD.items():
        cfg = load_build(build)
        ppi_nodes = read_ppi_nodes(cfg.celltype_ppi)

        op = pd.read_csv(cfg.omni, sep="\t")
        op = op[op.src.isin(ppi_nodes) & op.dst.isin(ppi_nodes) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
        omni_incident = ppi_nodes & (set(op.src) | set(op.dst))      # drop OmniPath orphans (no metabolite rescue here)
        prot = sorted(omni_incident)

        tag = f"healthy_{ct}"
        d = net_root / tag
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"node_id": prot, "node_type": "protein", "source": "pinnacle",
                      "direction": "", "sender_weight": 1.0}).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
        edges = op[op.src.isin(omni_incident) & op.dst.isin(omni_incident)].copy()
        pd.DataFrame({"source": edges.src, "target": edges.dst, "edge_origin": "OmniPath",
                      "edge_property": "", "weight": 1.0, "direction": ""}).to_csv(
            d / "network_edges.tsv", sep="\t", index=False)
        print(f"{tag:20s}: {len(prot)} proteins (of {len(ppi_nodes)} backbone), "
              f"{len(edges)} OmniPath edges (all weight 1.0)  -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build per-cell-type healthy (neutral-weight) networks")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_embedding")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
