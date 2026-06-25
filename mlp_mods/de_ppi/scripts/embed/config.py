"""Build config for the de_ppi influence pipeline — generalizes to (disease, cell type).

Resolves a build named in 02_build_ppi/builds_manifest.json into the concrete paths
the de_ppi scripts need. Read-only on the shared manifest; everything de_ppi-specific
(cell-type PPI, HMDB metabolites, literature TSVs, per-build OpenTargets drug scope,
the per-build results dir) is derived here from the build's slugs/fields.

Per-build outputs live under de_ppi/results/<build>/.

  from config import load_build
  cfg = load_build("macrophage_crohn")
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path("mlp_mods/de_ppi")          # mlp_mods/de_ppi
REPO = HERE.parents[1]                           # context_schema_agent/
MANIFEST = REPO / "mlp_mods/02_build_ppi/builds_manifest.json"

# Per-build OpenTargets disease-family scope for the drug table (diseaseId in this set
# OR an id here in the indication's OT `ancestors`). Mirrors 03_opentargets_rebuild.
# Builds not listed fall back to {primary EFO} (opentargets_key[0]) + ancestor expansion.
_CROHN_SCOPE = {"EFO_0003767", "EFO_0000384", "EFO_0000729", "MONDO_0005101", "MONDO_0005265"}
_ALZ_SCOPE = {"MONDO_0004975", "EFO_0000249"}         # Alzheimer disease (MONDO id in OT 25.06)
_ILD_SCOPE = {"EFO_0004244"}                           # interstitial lung disease; ancestors catch IPF / pulmonary fibrosis
DRUG_SCOPES = {
    "macrophage_crohn": _CROHN_SCOPE,
    # macrophage cell-state sub-builds reuse the same Crohn drug scope
    "macrophage_crohn_inflammatory": _CROHN_SCOPE,
    "macrophage_crohn_resident": _CROHN_SCOPE,
    "macrophage_crohn_proliferating": _CROHN_SCOPE,
}
# Alzheimer + ILD builds (incl. cell-state sub-builds) reuse one scope per disease
for _n in ("fibroblast_alzheimers", "glutamatergic_neuron_alzheimers", "microglia_alzheimers",
           "microglia_alzheimers_homeostatic", "microglia_alzheimers_dam",
           "microglia_alzheimers_interferon", "microglia_alzheimers_proliferating",
           "fibroblast_alzheimers_homeostatic", "fibroblast_alzheimers_myofibroblast",
           "fibroblast_alzheimers_inflammatory"):
    DRUG_SCOPES[_n] = _ALZ_SCOPE
for _n in ("macrophage_ild", "macrophage_ild_alveolar", "macrophage_ild_interstitial",
           "macrophage_ild_monocyte_derived"):
    DRUG_SCOPES[_n] = _ILD_SCOPE


@dataclass
class DePpiBuild:
    name: str
    cell_type: str
    disease: str
    disease_slug: str
    celltype_slug: str
    # inputs
    celltype_ppi: Path        # PINNACLE cell-type edgelist (node set)
    expressed_genes: Path     # optional: expressed-protein list (detect>=floor) for --expressed
    omni: Path                # OmniPath directed p->p edges (global)
    de_table: Path            # pseudobulk DE table (ranks + padj)
    ref_rank_col: str         # reference-arm rank column, e.g. "healthy_rank"
    disease_rank_col: str     # disease-arm rank column, e.g. "crohn_rank"
    metabolite_chebi: Path    # HMDB disease metabolites (accession, chebi)
    lit_genes: Path           # stage-L dysregulation_genes TSV
    lit_metabolites: Path     # stage-L dysregulation_metabolites TSV
    nodes: Path               # db/nodes.csv (global)
    edges: Path               # db/edges.tsv (global, MIND)
    opentargets_positive: Path
    ot_efo: str               # opentargets_key[0]
    ot_celltype: str          # opentargets_key[1]
    known_drugs_parquet: Path # OT known-drugs cache (global)
    drug_scope: set
    # outputs (de_ppi/results/<build>/)
    results_dir: Path

    @property
    def networks_dir(self) -> Path:
        return self.results_dir / "networks"

    @property
    def influence_dir(self) -> Path:
        return self.results_dir / "influence_analysis"

    @property
    def p3_influence(self) -> Path:
        return self.results_dir / "P3_influence.tsv"

    @property
    def embed_influence(self) -> Path:
        return self.results_dir / "embedding_influence.tsv"

    @property
    def network_nodes(self) -> Path:
        return self.networks_dir / "network_nodes.tsv"

    @property
    def network_edges(self) -> Path:
        return self.networks_dir / "network_edges.tsv"

    @property
    def drug_table(self) -> Path:
        return self.influence_dir / f"{self.disease_slug}_drug_influence.tsv"

    @property
    def phase_plot(self) -> Path:
        return self.influence_dir / "phase_vs_percentile.png"


def load_build(name: str) -> DePpiBuild:
    b = json.loads(MANIFEST.read_text())["builds"][name]
    cell_type, disease_slug, celltype_slug = b["cell_type"], b["disease_slug"], b["celltype_slug"]
    arms = b["arms"]
    ref = next(a for a in arms if a.get("disease_filter") is None)
    dis = next(a for a in arms if a.get("disease_filter") is not None)
    key = b.get("opentargets_key") or [None, None]
    efo, ot_ct = key
    ot_pos = REPO / b["opentargets_positive"] if b.get("opentargets_positive") else None
    return DePpiBuild(
        name=name,
        cell_type=cell_type,
        disease=b["disease"],
        disease_slug=disease_slug,
        celltype_slug=celltype_slug,
        # PINNACLE edgelist filename uses underscores for spaces in the cell-type name
        celltype_ppi=REPO / "db/pinnacle_official/ppi_edgelists" / f"{cell_type.replace(' ', '_')}.txt",
        expressed_genes=HERE / "expressed_genes" / f"{name}.txt",
        omni=REPO / "mlp_mods/omnipath_directed/omnipath_global_directed.tsv",
        de_table=REPO / b["de_table"],
        ref_rank_col=f"{ref['name']}_rank",
        disease_rank_col=f"{dis['name']}_rank",
        metabolite_chebi=REPO / f"mlp_mods/hmdb_{disease_slug}" / f"{disease_slug}_metabolite_chebi.tsv",
        lit_genes=REPO / f"mlp_mods/literature_{disease_slug}" / celltype_slug
                  / f"{disease_slug}_{celltype_slug}_dysregulation_genes.tsv",
        lit_metabolites=REPO / f"mlp_mods/literature_{disease_slug}" / celltype_slug
                        / f"{disease_slug}_{celltype_slug}_dysregulation_metabolites.tsv",
        nodes=REPO / "db/nodes.csv",
        edges=REPO / "db/edges.tsv",
        opentargets_positive=ot_pos,
        ot_efo=efo,
        ot_celltype=ot_ct,
        known_drugs_parquet=REPO / "mlp_mods/03_opentargets_rebuild/ot_cache/knownDrugsAggregated.parquet",
        drug_scope=DRUG_SCOPES.get(name, {efo} if efo else set()),
        results_dir=HERE / "results" / name,
    )
