"""Pull small-intestine fibroblast (IBD family + healthy/normal) from CellxGene Census
into the disease/tissue/cell_type catalog layout:

    01_expression/<disease_slug>/<tissue_general_slug>/<cell_type_slug>.h5ad

Raw counts (X_name="raw"); gene symbols in var.feature_name. The healthy (normal) arm
is the matched reference a fibroblast disease build would pair against.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cellxgene_census

ROOT = Path("mlp_mods/01_expression")
CELL_TYPE = "fibroblast"
TISSUE_GENERAL = "small intestine"
DISEASES = ["inflammatory bowel disease", "Crohn disease", "ulcerative colitis", "normal"]
OBS_COLS = ["soma_joinid", "dataset_id", "donor_id", "tissue", "tissue_general",
            "cell_type", "disease", "assay", "is_primary_data"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    dz_quoted = ", ".join(f'"{d}"' for d in DISEASES)
    filt = (f'cell_type == "{CELL_TYPE}" and tissue_general == "{TISSUE_GENERAL}" '
            f'and disease in [{dz_quoted}] and is_primary_data == True')
    with cellxgene_census.open_soma(census_version="stable") as census:
        obs = census["census_data"]["homo_sapiens"].obs.read(
            value_filter=filt, column_names=["disease", "donor_id"]).concat().to_pandas()
        print("small-intestine fibroblast cells by disease:",
              obs.disease.value_counts().to_dict(), flush=True)
        a = cellxgene_census.get_anndata(
            census, organism="Homo sapiens", obs_value_filter=filt, X_name="raw",
            column_names={"obs": OBS_COLS, "var": ["feature_id", "feature_name"]})
    print(f"pulled {a.n_obs} cells x {a.n_vars} genes", flush=True)

    for dz in DISEASES:
        sub = a[a.obs.disease == dz].copy()
        if sub.n_obs == 0:
            print(f"  {dz}: 0 cells — skipped", flush=True)
            continue
        out = ROOT / slug(dz) / slug(TISSUE_GENERAL) / f"{slug(CELL_TYPE)}.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.write_h5ad(out)
        print(f"  wrote {out}  ({sub.n_obs} cells, {sub.obs.donor_id.nunique()} donors)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
