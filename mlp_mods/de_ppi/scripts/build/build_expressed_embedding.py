"""Assemble a joint-embedding networks/ dir where each disease-state network's node set is defined
by EXPRESSION (detect>=floor per state) instead of the PINNACLE backbone -- i.e. non-expressed
backbone proteins are removed and expressed non-backbone proteins are added (--expressed-backbone),
using the threshold expressed lists in expressed_genes_threshold/. Sender weights / topology /
metabolites are unchanged. The 4 per-cell-type healthy anchors are copied from the PINNACLE-backbone
embedding unchanged.

Output: results/<out_name>/networks/<tag>/{network_nodes,network_edges}.tsv  for 15 states + 4 healthy.

Run:
  .venv/bin/python mlp_mods/de_ppi/build_expressed_embedding.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import shutil
from pathlib import Path

import build_literature_weighted_influence as B

HERE = Path("mlp_mods/de_ppi")
EXPR_DIR = HERE / "expressed_genes_threshold"
SRC_HEALTHY = HERE / "results" / "crohn_alzheimer_ild_embedding" / "networks"
OUT = HERE / "results" / "crohn_alzheimer_ild_uc_embedding_expressed" / "networks"

TAG_BUILD = {
    "crohn_mac_inflammatory": "macrophage_crohn_inflammatory",
    "crohn_mac_resident": "macrophage_crohn_resident",
    "crohn_stem_proliferating": "stem_crohn_proliferating",
    "alz_microglia_homeostatic": "microglia_alzheimers_homeostatic",
    "alz_microglia_dam": "microglia_alzheimers_dam",
    "alz_microglia_interferon": "microglia_alzheimers_interferon",
    "alz_microglia_proliferating": "microglia_alzheimers_proliferating",
    "alz_fibroblast_homeostatic": "fibroblast_alzheimers_homeostatic",
    "alz_fibroblast_myofibroblast": "fibroblast_alzheimers_myofibroblast",
    "crohn_fibroblast_homeostatic": "fibroblast_crohn_homeostatic",
    "crohn_fibroblast_inflammatory": "fibroblast_crohn_inflammatory",
    "crohn_fibroblast_myofibroblast": "fibroblast_crohn_myofibroblast",
    "ild_macrophage_alveolar": "macrophage_ild_alveolar",
    "ild_macrophage_interstitial": "macrophage_ild_interstitial",
    "ild_macrophage_monocyte_derived": "macrophage_ild_monocyte_derived",
}
HEALTHY = ["healthy_macrophage", "healthy_microglia", "healthy_fibroblast", "healthy_stem"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, build in TAG_BUILD.items():
        print(f"\n===== {tag}  ({build}) =====", flush=True)
        B.main(build, expressed_backbone=True, rank_weight_all=True,   # rank-shift weights on ALL ranked genes
               net_out=str(OUT / tag),
               expr_genes_path=str(EXPR_DIR / f"{build}.txt"))
    # per-(disease,cell type) healthy baselines are built separately by build_healthy_per_disease.py
    print(f"\nDONE: {len(TAG_BUILD)} expressed-backbone disease nets -> {OUT}\n"
          f"  (run build_healthy_per_disease.py for the 6 normal-arm healthy baselines)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
