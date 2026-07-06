# METHODS.md — single-cell directed-PPI joint embedding (de_ppi)

End-to-end recipe: how raw scRNA-seq becomes per-context PPI networks, how a shared encoder is trained, and
how we read disease signal out. Sections marked **[CURRENT]** describe what runs today; **[IN PROGRESS: scVI]**
marks the swap being built now (scVI for between-study normalization **and** cell-state assignment). Companion
docs: `CONTROLS.md` (control ladder + batch-correction results), `DISEASE_AXIS.md` (shared/unique decomposition).

## 0. Unit of analysis
For each **(arm, tissue, cell type, cell state)** — where arm ∈ {a disease, healthy} — we build a directed
PPI network whose **nodes are expressed proteins**, **edges are OmniPath**, and whose **node feature is that
context's expression**. A single shared encoder embeds proteins; disease signal is read as the **embedding
(or expression) shift of a disease network relative to its own-study healthy network**.

## 1. Data gathering  [CURRENT]
- **Sources** = "paired" atlases: each carries **disease + matched normal cells from the same study**, so the
  disease-vs-healthy contrast is within-study. Pulled from **CZ CELLxGENE Census** (per disease/celltype) and
  **GEO** (Smillie SCP259 UC; Garrido GSE214695 IBD colon). Raw counts, gene-symbol var names.
- **obs kept**: `disease`, `donor_id`, `dataset_id`, `cell_type`. Stored at `rank_shifts/<source>_paired/pulled_*.h5ad`.
- **Provenance** (dataset_id → study):

  | dataset_id | study | cell types / tissue |
  |---|---|---|
  | 518d9049 | Kong 2023, *Immunity* (10.1016/j.immuni.2023.01.002) | Crohn colon (macrophage) |
  | e6aaf5a4 / 40a0ade8 / 19053a82 | "metaplasia in inflammatory gut disease" *Nature* 2024 (10.1038/s41586-024-07571-1) | Crohn/UC colon+ileum |
  | 80a2c5b6 | Elmentaite 2021 *Nature* (10.1038/s41586-021-03852-1) | gut atlas |
  | macropha | Smillie 2019 *Cell* (SCP259) | UC colon (macrophage) |
  | garridoGSE214695 | Garrido-Trigo 2023 *Nat Commun* (GSE214695) | Crohn/UC/healthy colon (macrophage) |

  Diseases: Crohn, UC, Alzheimer, ILD. Cell types: macrophage, fibroblast, microglia, stem.
- **Full per-source inventory** (dataset_id → cell/donor counts, CELLxGENE collections, coverage gaps vs
  CELLxGENE, and the 2026-06-24 integration): see **Appendix A** below (relocated from the former
  `results/README.md`). The table here is the study-citation/DOI cut; Appendix A is the counts/coverage cut.

## 2. Cell QC  [CURRENT]
Keep cells with **≥500 counts and ≥300 genes** (drops low-quality/ambient cells). Applied everywhere downstream.

## 3. Cell-state assignment
- **[CURRENT]** `state_split.py`: **disease-blind Leiden** per source (PCA → kNN → Leiden); marker signatures
  only *name* clusters (inflammatory / resident / proliferating, etc.) — they never define them. States are
  computed **per source**, so they are not integrated across studies.
- **[IN PROGRESS: scVI]** Leiden on the **batch-corrected scVI latent** (§6), so states are defined in one
  integrated space and are comparable across studies (`run_scvi.py`). Markers still only *name* clusters.

## 4. Node universe — fixed shared node set per cell type  [CURRENT]
`build_shared_nodes.py`: a gene joins a cell type's node set iff it is **detected in ≥10% of pooled QC'd cells
AND in ≥2 datasets AND is OmniPath-incident**. Sizes: macrophage 4420, fibroblast 3398, stem 5532, microglia 4422.
- **Why fixed**: every network of a cell type then uses the *same* nodes, removing depth-driven membership churn
  (a batch confound).
- **Known bias**: the **pooled** detection floor excludes inducible cytokines (TNF, IL6) that *are* expressed in
  the disease arm of ≥2 studies but are diluted below 10% by healthy/other-tissue cells. A disease-arm-aware
  floor would readmit them (see CONTROLS.md discussion) — not yet adopted.

## 5. Network construction  [CURRENT]
Per (arm, tissue, celltype, state):
- **Nodes**: the fixed shared set (control networks) or the per-build expressed union (main networks).
- **Edges**: OmniPath **directed** protein→protein over the node set.
- **Node feature**: `expression` = `log1p(mean CP10k)` pseudobulk over that network's cells — **the only
  study-varying input**.
- **Weights**: **NEUTRAL everywhere** — `sender_weight` and edge `weight` all = 1.0. Main builders pass
  `B.main(neutral_weights=True)` so they emit neutral directly (the earlier DE rank-shift gating was removed —
  it barely moved the embedding and mixing weighted vs neutral networks created artifacts; the old post-hoc
  `neutralize_weights.py` is retired).
- **MAIN** networks (~44): studies pooled per context (the encoder's training set). **CONTROL** networks:
  per-study (the control ladder + disease-axis substrate). Both sets contain disease *and* healthy networks.

Consequence: within a cell type, all control networks share identical nodes, edges, and (neutral) weights, so
the *only* thing distinguishing Crohn / UC / healthy / study is the **per-gene expression feature** — they
differ in feature vector, not wiring.

## 6. Between-study normalization of the expression feature
The expression feature carries sequencing-depth / protocol batch. Correction lineage (build dirs in `results/`):
- **`..._expressed`** — raw mean-CP10k (no correction).
- **`..._combat_loc`** — **[CURRENT canonical]** ComBat **location-only**, fit per (tissue,celltype) with
  `batch=study8` and `arm`+`state` preserved; corrects the per-network expression (`apply_combat_expression.py`).
  See CONTROLS.md for the result (between-study floors collapse to the donor floor).
- **`..._scvi`** — **[macrophage built]** scVI: a cell-level VAE (negative-binomial likelihood + explicit
  library-size), `batch_key=study8`, trained per cell type (`run_scvi.py`). `get_normalized_expression()`
  (decoder at a fixed reference batch) → **pseudobulk per network = the node feature** (replaces the ComBat
  step); the same model's **latent** → §3 states. Depth/dropout-aware, so it targets the diagnosed driver more
  directly than ComBat. **Node membership is unchanged** — still derived from raw counts (§4/§5); scVI supplies
  only feature values + states, never the gene universe (denoised expression is dense and not a presence call).
  The staging→build adapter is `scripts/embed/adapt_scvi_build.py`: per (arm, tissue, celltype, state) context
  it writes membership = mean-CP10k≥0.5 (raw) ∩ OmniPath-incident, feature = `log1p(mean scVI-CP10k)`, neutral
  OmniPath edges. **Macrophage is built** (18 networks: Crohn ileum+colon, UC colon, ILD lung, matched healthy;
  encoder retrained with `--expr-feat`, N=3891); fibroblast/microglia/stem staging is pending. Because states
  are the integrated scVI-latent Leiden states, the scVI network set is not 1:1 with `_combat_loc`.
- **Identifiability (any correction)**: correction cannot separate disease from study where one study = one
  disease. We therefore always read disease as **disease − own-study healthy** (a within-study contrast), which
  is batch-free regardless of the correction method. Correction improves feature comparability; it does not
  manufacture cross-study disease signal — that requires independent studies per disease (why Garrido was added).

## 7. Encoder training  [CURRENT]  (`joint_embed.py`, `embedding_utils.py`)
- **One shared encoder** over the node universe (union of MAIN-network nodes; N≈8236), 64-dim.
- **Architecture**: 2-layer directed message passing over the row-normalized OmniPath operator `A`
  (`A[target,source]`, self-loops, receiver-normalized). With `--expr-feat` the **node input is a single scalar
  — its log-expression — via `Linear(1,dim)`** (replaces the learned identity table); identity then comes only
  from graph position. A residual self-path and an optional sender-weight feature exist but weights are neutral.
- **Objective (unsupervised)**: directed **link prediction** (asymmetric bilinear decoder + negative sampling)
  **+ edge-weight reconstruction**, summed over the 44 MAIN networks. 300 epochs, Adam (lr 1e-2), seed 3,
  10% edges held out for AUC.
- **Honest characterization**: the objective reconstructs **OmniPath** (shared across networks) and contains
  **no disease signal**; disease enters only through the per-network expression feature at inference. So the
  encoder is a **graph-aware projection of expression**, not a model "trained on disease," and within a cell
  type it adds OmniPath-neighborhood smoothing, not rewiring.

## 8. Inference  [CURRENT]  (`infer_controls.py`)
Control networks are **forward-passed through the frozen encoder** (no retraining) → `Z` per (network, protein).
This keeps controls from shaping the space they are meant to measure.

## 9. Analysis
- **Control ladder** (`compare_controls.py`, `control_centered.py` → CONTROLS.md): magnitude floors (donor `a`/`h`,
  batch `b`/`i`/`m`) vs factor effects (state, cell type, tissue, disease); plus healthy-anchored direction set.
- **Disease-axis decomposition** (`disease_axis_decompose.py` → DISEASE_AXIS.md): co-movement shared/unique split
  of Crohn vs UC colon macrophage, own-study-healthy centered, with cross-study reproducibility (`*_xstudy_cos`).

**Open methodological question (measured, unresolved).** Running the disease-axis on the **encoder embedding**
vs on the **raw corrected pseudobulk** gives *different* answers (per-protein Spearman ≈ 0.4–0.5; top-protein-list
Jaccard 0.15–0.32; substantial label disagreement). So the encoder is **not redundant**, but its divergence is
**not yet shown to be biology** — some embedding-unique hits are topology artifacts (e.g. HACD4/BEX3: large
embedding shift, ~0 expression change, unstable low-degree nodes). The corrected **pseudobulk is the direct,
interpretable substrate**; the encoder must earn its keep by beating pseudobulk on cross-study reproducibility.

## 10. Build-dir lineage
**See HISTORY.md for the authoritative, current lineage** (expressed → combat_loc → combat_loc_coexpr →
{coexpr_exprfilt, coexpr_healthyph → context_contrastive / masked / masked_delta}; side branches
pinnacle_combat_ct and expressed_scvi [macrophage built]). Original dirs are never overwritten; each change
makes a new dir. (This section previously held a stale hand-maintained diagram — removed to avoid drift.)

## Code map
`rank_shifts/de_scripts/` pulls + `state_split.py`; `de_ppi/scripts/build/controls/` (`build_shared_nodes`,
`build_pooled_controls`, `build_healthy_controls`, `apply_combat_expression`); `de_ppi/scripts/embed/`
(`joint_embed`, `embedding_utils`, `infer_controls`); `de_ppi/scripts/analysis/` (`compare_controls`,
`control_centered`, `disease_axis_decompose`).

---

## Appendix A — data provenance & build inventory  *(relocated from results/README.md, 2026-07-02)*

Every embedding build is assembled from the same pool of **9 source datasets** (one per cell type × disease),
each processed in `rank_shifts/<source>_states/` (disease-blind Leiden states + paired pseudobulk DE). All
sources are CZ **CELLxGENE** datasets except the UC macrophage data (Smillie et al., pulled separately).
Dataset IDs are CELLxGENE dataset UUIDs.

### Source datasets (per cell type × disease)

| source (`rank_shifts/<x>_states`) | disease | cell type | dataset_id(s) | n_datasets | cells | donors | CELLxGENE collection |
|---|---|---|---|---|---|---|---|
| `microglia_alzheimers` | Alzheimer | microglia | `203025fe`, `ac0c6561`, `cff99df2` | 3 | 21,488 | 83 | Brain vascular multi-omics; Cross-dementia snRNA-seq (Rexach); Molecular Signatures of Resilience to AD |
| `fibroblast_alzheimers` | Alzheimer | fibroblast | `203025fe` | 1 | 4,596 | 19 | Brain vascular single-cell multi-omics |
| `macrophage_crohn` | Crohn | macrophage (ileum) | `a37f857c` | 1 | 29,603 | 41 | The landscape of immune dysregulation in Crohn's disease (Kong et al.) — TI immune |
| `macrophage_crohn_colon` | Crohn | macrophage (colon) | `518d9049` | 1 | 11,440 | 22 | Kong et al. — colon immune |
| `macrophage_crohn_rep` | Crohn | macrophage (replicate, control) | `19053a82` | 1 | 1,375 | 19 | Single-cell integration reveals metaplasia in IBD |
| `fibroblast_crohn` | Crohn | fibroblast | `0f4865d5`, `19053a82`, `8e47ed12` | 3 | 36,622 | 74 | Kong et al. (TI stromal); metaplasia-in-IBD; Developing Human Gut (Gut Cell Atlas) |
| `stem_crohn` | Crohn | intestinal stem | `19053a82`, `8e47ed12` | 2 | 2,428 | 39 | metaplasia-in-IBD; Developing Human Gut |
| `macrophage_uc_smillie` | ulcerative colitis | macrophage | *(none recorded)* | 1 | 11,684 | 30 | Smillie et al. 2019 UC colon (`build_uc_smillie.py`) |
| `macrophage_ild` | interstitial lung disease | macrophage | `f14bc322` | 1 | 69,420 | 91 | Single-cell RNA-seq analysis of Interstitial lung disease |

Full dataset_ids:
`203025fe-fa99-4d57-81da-458ed8f0c334`, `ac0c6561-7a48-4185-af6f-af799f699172`,
`cff99df2-4904-44f7-9173-ff837f95606e`, `a37f857c-779f-464e-9310-3db43a1811e7`,
`518d9049-2a76-44f8-8abc-1e2b59ab5ba1`, `19053a82-9c89-4fb8-bd19-d7b1800b0b7b`,
`0f4865d5-8000-4f68-8ac7-f5efea9e5e70`, `8e47ed12-c658-4252-b126-381df8d52a3d`,
`f14bc322-1322-4184-8d16-409557525ea5`.

> Note: `19053a82` (metaplasia-in-IBD) contains both Crohn **and** UC cells, but we use only its
> Crohn cells (fibroblast/stem/macrophage-replicate). The UC-specific macrophage data comes from the
> Smillie study, not `19053a82`.

### Embedding builds  *(partial legacy inventory — authoritative lineage is in HISTORY.md)*

All embedding builds use the **expressed** node set (expressed proteins ∪ DE ∪ literature; PINNACLE backbone
dropped — see HISTORY.md step 2). Networks = directed PPI per (disease, cell type, cell state); shared-encoder
joint embedding (`joint_embed_*.py`), 64-dim. This table predates the coexpr/scVI/masked builds — **use
HISTORY.md for the current lineage**; kept here for the original 4 builds' network counts / source coverage.
**These four dir names are no longer on disk** (superseded by the coexpr/scVI/masked builds) — retained only as
a historical record of counts, not as pointers to live directories.

| build | networks | diseases | cell types | source datasets | notes |
|---|---|---|---|---|---|
| `crohn_alzheimer_ild_uc_embedding_expressed` | 30 (22 disease-state + 8 healthy) | Crohn, UC, ILD, Alzheimer | macrophage, microglia, fibroblast, stem | all 9 | primary expressed build; main analysis embedding |
| `crohn_alzheimer_ild_uc` | 68 (30 + 38 controls) | same | same | all 9 (+ control subsets: per-study singles, donor splits, LOO pools) | main networks symlinked from `_expressed`; adds donor_split / between_study / pool_vs_pool **controls** |
| `crohn_alzheimer_ild_uc_embedding_context` | as `_expressed` | same | same | all 9 | context-conditioned variant (factorized disease/tissue/cell/state vectors), `joint_embed_context.py` |
| `crohn_alzheimer_ild_uc_embedding_context_disease` | as `_expressed` | same | same | all 9 | context variant, disease factor only |
| `_backup_pre_expressed_*`, `_stale/` | — | — | — | — | archived; not active |

### Coverage gaps vs CELLxGENE (census 2025-11-08)

Datasets on CELLxGENE with ≥50 cells of our families (macrophage/microglia/fibroblast/stem) for each disease
that are **NOT** in our builds (top by cell count; family match is by cell-type label substring, so subtypes
are caught but higher-level labels like "myeloid cell" are not):

- **Alzheimer / microglia** — HAVE 3 of 10. Notably missing:
  - `a1b9c51e` "Live Human Microglia" (~14k microglia) — largest gap.
  - `2727d83a`, `9f1049ac`, `6c600df6`, `bdacc907` "Molecular characterization of selectively
    vulnerable neurons…" series (~3–5k microglia each).
  - `0a2d7e87`, `fe2eecbc` "Deciphering glial contributions to CSF1R-related disease".
- **Crohn** — HAVE 5 of 15. Missing are largely **other compartment slices of collections we already
  use** (Kong "immune dysregulation in Crohn's"; "metaplasia in IBD"): e.g. `fe4b89d5` (stem 6.9k),
  `e6aaf5a4` (fib/mac/stem 5k), `ef7bb7f0` (fib/mac 3.6k), plus `80a2c5b6` (Gut Cell Atlas, fib/mac/stem).
- **Ulcerative colitis** — HAVE 1 of 6 (weakest coverage). Missing: `e6aaf5a4` (fib/mac/stem 4.9k),
  `ef7bb7f0` (fib/mac 4.6k), both from the metaplasia-in-IBD collection. UC is currently anchored on a
  single source (Smillie macrophages), so it is the least replicated disease in the build.

Gap scan: `cellxgene_census` (stable = 2025-11-08), `disease in {Crohn disease, ulcerative colitis,
Alzheimer disease}`, cell-type family substring match. ILD not scanned (single source `f14bc322`).

### CELLxGENE integration update (2026-06-24, census 2025-11-08)

Additional Crohn/UC cells (existing cell types) were integrated by pooling into the existing
(disease, tissue, cell type) cohorts; states/DE re-derived; **both embeddings retrained**. Updated
dataset/donor inventory **now in use** per integrated source (disease + normal arm, post-augmentation):

| source | datasets in use | disease donors | normal donors |
|---|---|---|---|
| macrophage_crohn (ileum) | a37f857c, 40a0ade8, 80a2c5b6, e6aaf5a4 | 19 | 76 |
| macrophage_crohn_colon | 518d9049, 40a0ade8, 80a2c5b6, e6aaf5a4 | 5 | 71 |
| stem_crohn | 19053a82, 8e47ed12, 07574da2, fe4b89d5 | 26 | 76 |
| macrophage_uc_smillie | Smillie + 40a0ade8, e6aaf5a4 | 27 | 61 |
| fibroblast_crohn | 0f4865d5, 19053a82, 8e47ed12, e6aaf5a4 | 29 | 90 |

New dataset UUIDs added: `40a0ade8-6067-4e22-9224-4d3c5e9bfc0d`, `80a2c5b6-02e7-4fc0-9f12-179f5247c1bc`,
`e6aaf5a4-16e9-4ea6-9733-4eafd4e473d3`, `07574da2-6bd3-4708-a04d-27d95c009f4e`,
`fe4b89d5-461e-440c-a5a8-621b37b122c0`. (Other Crohn/UC gap datasets were either redundant on dedup or
DE-underpowered per state and not integrated.) Methodology + effects: see the ComBat/augmentation sections of
CONTROLS.md and HISTORY.md. Alzheimer/ILD builds unchanged.
