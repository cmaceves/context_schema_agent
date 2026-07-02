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
- **`..._scvi`** — **[IN PROGRESS]** scVI: a cell-level VAE (negative-binomial likelihood + explicit
  library-size), `batch_key=study8`, trained per cell type (`run_scvi.py`). `get_normalized_expression()`
  (decoder at a fixed reference batch) → **pseudobulk per network = the node feature** (replaces the ComBat
  step); the same model's **latent** → §3 states. Depth/dropout-aware, so it targets the diagnosed driver more
  directly than ComBat. **Node membership is unchanged** — still derived from raw counts (§4/§5); scVI supplies
  only feature values + states, never the gene universe (denoised expression is dense and not a presence call).
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
`..._expressed` (raw) → `..._combat_loc` (ComBat loc-only, current canonical) → `..._combat_loc_g` (+Garrido,
partial) → `..._scvi` (in progress). Original dirs are never overwritten; each correction makes a new dir.

## Code map
`rank_shifts/de_scripts/` pulls + `state_split.py`; `de_ppi/scripts/build/controls/` (`build_shared_nodes`,
`build_pooled_controls`, `build_healthy_controls`, `apply_combat_expression`); `de_ppi/scripts/embed/`
(`joint_embed`, `embedding_utils`, `infer_controls`); `de_ppi/scripts/analysis/` (`compare_controls`,
`control_centered`, `disease_axis_decompose`).
