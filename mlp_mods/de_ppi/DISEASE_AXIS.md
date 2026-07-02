# DISEASE_AXIS.md — shared / unique disease-axis decomposition

Worked example for the central claim: *disease perturbation decomposes into a **shared** axis (common to
diseases at a given cell type) and a **unique** axis (disease-specific), recoverable in one comparable space.*
The one confound-free contrast in our data is **IBD: Crohn vs UC, colon macrophage** (two diseases sharing a
tissue+cell type, each with a paired healthy arm). This file tracks the method, results, and caveats.

## Why this is the only clean contrast
A cross-disease shared/unique split needs **≥2 diseases in the same (tissue, cell type)**. In our build that
exists only for colon macrophage (Crohn + UC). Everywhere else a cell type carries a single disease
(microglia=Alz, stem=Crohn, lung mac=ILD), so "consensus across diseases" is not computable there. Scaling
the claim → densify one cell type with more **paired** (disease+normal) datasets (see CONTROLS.md / project plan).

## Build
Runs on **`crohn_alzheimer_ild_uc_embedding_expressed_combat_loc`** — the ComBat **location-only**,
batch-corrected build (between-study batch removed to the donor floor; see CONTROLS.md "ComBat" section).

## Method (`scripts/analysis/disease_axis_decompose.py`)
All in the 64-d embedding space, **batch-cancelled** by centering each disease net on its **own-study healthy**
arm; UC is pooled over its two studies *after* own-study centering (no shared-healthy-anchor inflation).

Per protein p (over proteins present in every contributing net):
```
r_C[p] = Z_Crohn(518d9049)[p] - Z_healthy(518d9049)[p]
r_U[p] = mean_s ( Z_UC(s)[p] - Z_healthy(s)[p] ),   s in {e6aaf5a4, macropha}
shared s[p] = (r_C + r_U)/2      shared_mag = ||s||
unique u[p] = (r_C - r_U)/2      unique_mag = ||u||     (identity: r_C = s+u, r_U = s-u)
shared_frac[p] = ||s||^2 / (||s||^2 + ||u||^2)          in [0,1]
movement       = max(||r_C||, ||r_U||)                  below move_pct percentile -> "static"
label          = static | shared (frac>=0.5) | crohn_unique (||r_C||>||r_U||) | uc_unique
uc_xstudy_cos[p] = cos(r_UC^e6aaf5a4[p], r_UC^macropha[p])   per-protein UC reproducibility (robustness col)
```
Global headline: `shared_fraction = sum||s||^2 / sum(||s||^2 + ||u||^2)`.

Per-protein cross-DISEASE agreement (`cos(r_C,r_U)`) is intentionally **not** a separate metric — `shared_frac`
already encodes it. The `uc_xstudy_cos` column is cross-STUDY (within UC) reproducibility, a robustness readout.

## Outputs (`results/<build>/disease_axis/`)
- `disease_axis_proteins.tsv` — one row per protein: `mag_crohn, mag_uc, shared_mag, unique_mag, shared_frac,
  uc_xstudy_cos, movement, label, rank_shared, rank_unique`.
- `disease_axis_summary.tsv` — global `shared_fraction`, mean per-protein disease cosine, mean UC reproducibility,
  label counts.

## Validation — EXTERNAL biological recovery (primary)
Do `shared` proteins recover known **pan-IBD** genes, and `*_unique` proteins recover known **Crohn-** vs
**UC-specific** markers? This is the load-bearing validation: a batch/noise artifact would not recover known IBD
biology, and it works symmetrically for both diseases (unlike internal cross-study, which UC alone supports).
The `uc_xstudy_cos` column is a cheap secondary robustness number, **not** a gate.

## Result (move_pct=50; 4420 colon-macrophage proteins)
- **shared_fraction ≈ 0.42** (≈42% of the IBD perturbation energy is shared, ≈58% unique) — substantial divergence.
- labels: 867 shared / 844 crohn_unique / 499 uc_unique / 2210 static.
- **Top shared = S100A8 / S100A9 (calprotectin)** — the canonical pan-IBD marker; MARCO, TIMP1, RNASE1, IL7R.
- **Top crohn_unique** = MMP9, CXCL2, FOS, AOAH. **Top uc_unique** = HLA-DRB5, NFKBIA, SPP1, APOE, APOC1.
- mean per-protein disease cosine = **−0.13** at `allstates` (vs **+0.27** at matched `inflammatory` state — see caveat).

The S100A8/A9 shared hit is a strong, untuned external-recovery signal.

## Caveats (read before trusting)
1. **`allstates` conflates cell-state composition with molecular disease biology.** The script uses `allstates`
   colon-macrophage nets (so UC can pool its 2 studies), but Crohn and UC cohorts differ in macrophage-state mix,
   so part of the "unique" signal is *compositional*, not per-state molecular — this is why the disease cosine is
   −0.13 here vs +0.27 at matched `inflammatory` state. **Open fork:** allstates (composition-inclusive, UC
   cross-study) vs matched-state (cleaner molecular contrast, UC=1 study, single state) vs per-state-then-average.
2. **Crohn colon macrophage is a single study** (`518d9049`) → `crohn_unique` proteins cannot be cross-study
   validated internally; only UC can (2 studies). External recovery is the symmetric check.
3. Depends on the ComBat correction (CONTROLS.md): pre-correction the disease shift did not reproduce across
   studies. Disclose the batch-correction dependency.

## Not to be confused with
`scripts/analysis/disease_axis.py` (older, separate): a baseline-free **consensus-axis cosine across all 4
diseases** on the uncorrected build — a different analysis, kept as-is.

## Run
```
.venv/bin/python mlp_mods/de_ppi/scripts/analysis/disease_axis_decompose.py \
    --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc
# options: --tissue colon --celltype macrophage --arm-a crohn --arm-b uc --move-pct 50
```
