"""Pull healthy (normal) large-intestine macrophages from CellxGene Census.

This is the matched-tissue healthy baseline for the gut Crohn macrophages in
macrophage_ibd.h5ad, needed for a clean Crohn-vs-healthy macrophage
differential-expression contrast. Raw counts (X_name="raw"), gene symbols in
var.feature_name.
"""
from __future__ import annotations
import sys
import cellxgene_census

OUT = "mlp_mods/01_expression/healthy_macrophage_large_intestine.h5ad"
FILT = ('disease == "normal" and cell_type == "macrophage" '
        'and tissue_general == "large intestine" and is_primary_data == True')


def main() -> int:
    with cellxgene_census.open_soma(census_version="stable") as census:
        # quick count first
        obs = census["census_data"]["homo_sapiens"].obs.read(
            value_filter=FILT, column_names=["soma_joinid", "donor_id", "tissue"]
        ).concat().to_pandas()
        print(f"healthy large-intestine macrophages: {len(obs)} cells, "
              f"{obs['donor_id'].nunique()} donors, tissues={sorted(obs['tissue'].unique())}",
              flush=True)
        if len(obs) < 200:
            print("WARNING: few cells — consider broadening the tissue filter.", flush=True)
        a = cellxgene_census.get_anndata(
            census, organism="Homo sapiens",
            obs_value_filter=FILT, X_name="raw",
            column_names={"obs": ["soma_joinid", "donor_id", "tissue",
                                  "tissue_general", "cell_type", "disease", "assay"],
                          "var": ["feature_id", "feature_name"]},
        )
    a.write_h5ad(OUT)
    print(f"wrote {OUT}: {a.n_obs} cells x {a.n_vars} genes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
