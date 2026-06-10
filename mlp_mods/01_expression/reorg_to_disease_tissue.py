"""Reorganize the flat 01_expression/*.h5ad pulls into a provenance-explicit layout:

    01_expression/<disease_slug>/<tissue_general_slug>/<cell_type_slug>.h5ad

Each existing flat file pools diseases (Crohn/UC/IBD-parent, or normal) and tissues;
this splits every flat top-level *.h5ad by (cell_type, disease, tissue_general) so the
origin disease + tissue of each slice is encoded in its path. Only top-level flat files
are processed (already-split subdir files are left alone). Source files are NOT deleted
here — verify the split, then remove them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import anndata as ad

ROOT = Path("mlp_mods/01_expression")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def main() -> int:
    flat = sorted(p for p in ROOT.glob("*.h5ad"))      # top-level pooled files only
    print(f"reorganizing {len(flat)} flat files into <disease>/<tissue_general>/<cell_type>.h5ad", flush=True)
    total_cells = total_slices = 0
    for f in flat:
        a = ad.read_h5ad(f)
        obs = a.obs
        grp = obs.groupby(["cell_type", "disease", "tissue_general"], observed=True).size()
        grp = grp[grp > 0]
        n = 0
        for (ct, dz, tg), cnt in grp.items():
            mask = ((obs.cell_type == ct) & (obs.disease == dz) & (obs.tissue_general == tg)).values
            sub = a[mask].copy()
            out = ROOT / slug(dz) / slug(tg) / f"{slug(ct)}.h5ad"
            out.parent.mkdir(parents=True, exist_ok=True)
            sub.write_h5ad(out)
            n += 1; total_slices += 1; total_cells += sub.n_obs
        print(f"  {f.name}: {a.n_obs} cells -> {n} (disease,tissue) slices", flush=True)
        del a
    print(f"\ndone: {total_slices} slice files, {total_cells} cells written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
