"""Step 2 — pull IBD-family (parent + Crohn + UC) scRNA-seq for every cell
type with enough donor + cell coverage, regardless of tissue or PINNACLE
mapping (per the 2026-05-19 PLAN.md revision).

slice_manifest.tsv is keyed by (cell_type, tissue). h5ad files pool all
tissues for an eligible cell type into a single AnnData with obs['tissue']
and obs['tissue_general'] preserved per cell. Step 3 contrasts on cell_type;
Step 4 wires the (cell_type, tissue) bridge into the metagraph.

No matched-normal arm is pulled: Step 3 builds disease-state PPI layers by
running PINNACLE's marker-Wilcoxon recipe on the disease cells alone
(one-vs-rest across cell types), so the normal arm is unnecessary.

Usage:
    pull_ibd_family.py --dry-run     # discovery only; writes slice_manifest.tsv
    pull_ibd_family.py               # full pull
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cellxgene_census
import pandas as pd

PROJECT = Path("/home/caceves/context_schema_agent")
PPI_DIR = PROJECT / "db" / "pinnacle_official" / "ppi_edgelists"
OUT_DIR = PROJECT / "mlp_mods" / "01_expression"
LOG_PATH = OUT_DIR / "pull_ibd_family.log"

ORG = "Homo sapiens"
DISEASE_IBD_LABELS = (
    "inflammatory bowel disease",
    "Crohn disease",
    "ulcerative colitis",
)
MIN_DONORS = 3
MIN_CELLS_PER_DONOR = 10

OBS_COLS = [
    "soma_joinid", "dataset_id", "donor_id", "tissue", "tissue_general",
    "cell_type", "disease", "is_primary_data", "assay",
]

# Map CellxGene cell-type labels that don't exactly equal a PINNACLE filename
# stem to the correct stem. Confirmed at the 2026-05-18 Step 2 checkpoint.
CELLTYPE_ALIASES: dict[str, str] = {
    "conventional dendritic cell": "mature_conventional_dendritic_cell",
}

# Cell-type labels to exclude from eligibility regardless of donor/cell counts.
# "unknown" is a CellxGene placeholder; its cells can be anything biologically,
# so including it in Step 3's one-vs-rest Wilcoxon contrast would pollute the
# marker ranking of every other cell type.
CELLTYPE_BLACKLIST: set[str] = {"unknown"}


def log(msg: str):
    line = f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def pinnacle_stem_for(celltype: str) -> str | None:
    """Return the PINNACLE filename stem for a CellxGene cell_type, or None.
    Informational only — Step 2 no longer drops cell types without a stem
    (PLAN.md 2026-05-19); Step 4 uses this for the disease-state PPI install
    decision."""
    raw = celltype.lower().replace(" ", "_")
    p = PPI_DIR / f"{raw}.txt"
    if p.exists() and not p.is_symlink():
        return raw
    aliased = CELLTYPE_ALIASES.get(celltype.lower())
    if aliased:
        p = PPI_DIR / f"{aliased}.txt"
        if p.exists() and not p.is_symlink():
            return aliased
    return None


def h5ad_name_for(celltype: str) -> str:
    """h5ad filename stem. Prefer the PINNACLE stem when available so the
    directory stays consistent with downstream conventions. Otherwise fall
    back to the lowercased + space->underscore form of the cell_type label."""
    stem = pinnacle_stem_for(celltype)
    if stem is not None:
        return f"{stem}_ibd.h5ad"
    return f"{celltype.lower().replace(' ', '_')}_ibd.h5ad"


def discover(census) -> pd.DataFrame:
    """Single obs scan over IBD-family cells across ALL tissues."""
    log(f"scanning IBD-family obs (labels={list(DISEASE_IBD_LABELS)}; all tissues)...")
    exp = census["census_data"]["homo_sapiens"]
    ibd_quoted = ", ".join(f"'{d}'" for d in DISEASE_IBD_LABELS)
    filt = f"disease in [{ibd_quoted}] and is_primary_data == True"
    obs = exp.obs.read(value_filter=filt, column_names=OBS_COLS).concat().to_pandas()
    log(f"  scan returned {len(obs):,} primary cells "
        f"({obs['dataset_id'].nunique()} datasets, "
        f"{obs['donor_id'].nunique()} donors, "
        f"{obs['cell_type'].nunique()} cell types, "
        f"{obs['tissue'].nunique()} tissues)")
    return obs


def manifest(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell_type, tissue) slice. Per-slice eligibility +
    cell-type-level eligibility (any eligible slice OR combined-across-
    tissues thresholds, per PLAN.md Step 2 action 2)."""

    # tissue -> tissue_general (assume 1:1 within a Census snapshot)
    tg_by_tissue = (
        obs.dropna(subset=["tissue", "tissue_general"])
           .drop_duplicates(subset=["tissue"])
           .set_index("tissue")["tissue_general"]
           .to_dict()
    )

    # Per-(cell_type, tissue) slice aggregates
    by_slice = obs.groupby(["cell_type", "tissue"], observed=True)
    n_cells = by_slice.size().rename("n_cells")
    n_donors = by_slice["donor_id"].nunique().rename("n_donors")
    n_datasets = by_slice["dataset_id"].nunique().rename("n_datasets")
    median_cpd = by_slice.apply(
        lambda g: g.groupby("donor_id", observed=True).size().median(),
        include_groups=False,
    ).rename("median_cells_per_donor")
    label_breakdown = (
        obs.groupby(["cell_type", "tissue", "disease"], observed=True)
           .size()
           .unstack("disease", fill_value=0)
           .reindex(columns=list(DISEASE_IBD_LABELS), fill_value=0)
    )

    # Per-cell-type aggregates (for the "combined across tissues" clause)
    by_ct = obs.groupby("cell_type", observed=True)
    ct_n_donors = by_ct["donor_id"].nunique()
    ct_median_cpd = by_ct.apply(
        lambda g: g.groupby("donor_id", observed=True).size().median(),
        include_groups=False,
    )

    rows = []
    for (ct, tissue), n in n_cells.items():
        stem = pinnacle_stem_for(ct)
        slice_eligible = bool(
            n_donors[(ct, tissue)] >= MIN_DONORS
            and median_cpd[(ct, tissue)] >= MIN_CELLS_PER_DONOR
        )
        combined_ok = bool(
            ct_n_donors[ct] >= MIN_DONORS
            and ct_median_cpd[ct] >= MIN_CELLS_PER_DONOR
        )
        rows.append({
            "cell_type": ct,
            "tissue": tissue,
            "tissue_general": tg_by_tissue.get(tissue, ""),
            "pinnacle_stem": stem,
            "pinnacle_mapped": stem is not None,
            "n_cells": int(n),
            "n_donors": int(n_donors[(ct, tissue)]),
            "n_datasets": int(n_datasets[(ct, tissue)]),
            "median_cells_per_donor": float(median_cpd[(ct, tissue)]),
            "n_ibd_parent": int(label_breakdown.loc[(ct, tissue), "inflammatory bowel disease"]),
            "n_crohn": int(label_breakdown.loc[(ct, tissue), "Crohn disease"]),
            "n_uc": int(label_breakdown.loc[(ct, tissue), "ulcerative colitis"]),
            "slice_eligible": slice_eligible,
            "_combined_ok": combined_ok,
        })

    df = pd.DataFrame(rows)
    any_slice_eligible = (
        df.groupby("cell_type")["slice_eligible"].any()
          .rename("any_slice_eligible").reset_index()
    )
    df = df.merge(any_slice_eligible, on="cell_type", how="left")
    df["cell_type_eligible"] = df["any_slice_eligible"] | df["_combined_ok"]
    # Hard blacklist applied after threshold-based eligibility so the blacklist
    # reason is auditable (the cell type may pass thresholds but still be dropped).
    df.loc[df["cell_type"].str.lower().isin(CELLTYPE_BLACKLIST), "cell_type_eligible"] = False
    df = df.drop(columns=["any_slice_eligible", "_combined_ok"])

    return df.sort_values(
        ["cell_type_eligible", "pinnacle_mapped", "cell_type", "n_cells"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def pull_ibd(census, cell_type: str):
    """Pull all IBD-family cells of a given cell_type across all tissues into
    a single AnnData. obs['tissue'] / obs['tissue_general'] preserved per cell."""
    ibd_quoted = ", ".join(f"'{d}'" for d in DISEASE_IBD_LABELS)
    cellxgene_filter = (
        f"disease in [{ibd_quoted}] "
        f"and cell_type == '{cell_type}' "
        f"and is_primary_data == True"
    )
    return cellxgene_census.get_anndata(
        census=census, organism=ORG, measurement_name="RNA",
        obs_value_filter=cellxgene_filter,
        X_name="raw",
        obs_column_names=[
            "soma_joinid", "dataset_id", "donor_id", "tissue", "tissue_general",
            "cell_type", "disease", "assay", "is_primary_data",
        ],
        var_column_names=[
            "soma_joinid", "feature_id", "feature_name", "feature_length",
        ],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Discovery only; write slice_manifest.tsv and exit.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    log("opening CellxGene Census...")
    with cellxgene_census.open_soma() as census:
        obs = discover(census)
        man = manifest(obs)
        man_path = OUT_DIR / "slice_manifest.tsv"
        man.to_csv(man_path, sep="\t", index=False)
        log(f"manifest written: {man_path} ({len(man)} (cell_type, tissue) rows)")

        n_slices_elig = int(man["slice_eligible"].sum())
        elig_cts = man[man["cell_type_eligible"]]["cell_type"].drop_duplicates().tolist()
        n_mapped = man[man["cell_type_eligible"] & man["pinnacle_mapped"]]["cell_type"].nunique()
        log(f"slice-level eligible (cell_type, tissue) rows: {n_slices_elig}")
        log(f"cell_type-eligible cell types: {len(elig_cts)}")
        log(f"  PINNACLE-mapped among them: {n_mapped} "
            f"(non-mapped also get a disease-state PPI in Step 4, per 2026-05-19 scope)")

        if args.dry_run:
            log("--dry-run: skipping pulls.")
            return

        # One pull per cell_type — pools all tissues for that cell type.
        for ct in elig_cts:
            out = OUT_DIR / h5ad_name_for(ct)
            if out.exists():
                log(f"  (skip) {out.name} exists ({out.stat().st_size / 1e6:.1f} MB)")
                continue
            log(f"  pulling {ct!r} → {out.name}")
            try:
                a = pull_ibd(census, ct)
            except Exception as e:
                log(f"    ERROR: {e}")
                continue
            a.write_h5ad(out)
            log(f"    {a.n_obs:,} cells × {a.n_vars:,} genes, "
                f"{a.obs['tissue'].nunique()} tissues → "
                f"{out.stat().st_size / 1e6:.1f} MB")

    log("done.")


if __name__ == "__main__":
    main()
