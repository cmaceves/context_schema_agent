# CONTROLS.md — control / factor-contrast design (a–j)

> **⚠️ DEPRECATED (2026-07-02) — the a–j ladder is no longer the embedding-evaluation framework.**
> Retained below for the record, but superseded for two reasons: (1) it is **circular for any objective
> trained on the arm label** (contrastive, healthy_centered) — the disease control `g` rises by
> construction, not because disease signal is real (see CONTEXT_EMBED.md); (2) the ladder measures
> per-factor embedding-shift magnitude/direction on the frozen link-prediction space, and that space is
> now shown to be dominated by **out-degree + expression passthrough**, not target/disease biology (see the
> scVI in-silico-perturbation entry in HISTORY.md: projection ≈ hub propagation, sign set by link-pred
> geometry). Going forward, builds are judged by **generalization**, not this ladder: leave-one-study-out
> masked-feature reconstruction, and LOSO / held-out-protein performance of a supervised disease-shift
> (ΔZ) → OpenTargets head (`scripts/analysis/embedding_target_cv.py`).
> **Not deprecated:** the ComBat batch-correction result (the "ComBat batch correction…" section) — using
> the floors to show between-study batch is removable to the donor floor is a legitimate feature-level
> diagnostic and stands independent of whether the ladder is used to score embeddings.

Controls for the `crohn_alzheimer_ild_uc_embedding_expressed` joint embedding. The goal: put the
**noise floors** (disease-arm a, b; healthy-arm h, i) on the same ladder as the **biological factor
effects** (c–g), so we can ask whether a disease/state/cell-type/tissue effect exceeds what sampling,
study/batch, and normal cohort-to-cohort variation alone produce.

## Shared method (all of a–j)

- **One frozen encoder.** The encoder is trained on the 30 MAIN disease-state networks only
  (`encoder.pt`). Every control network is placed in that space by **forward-pass inference** — it is
  never retrained into the encoder. This keeps controls from shaping the space they are meant to measure.
- **Reference = main-network consensus.** `consensus[p]` = per-protein mean of `Z` over the 30 MAIN
  networks. Deviation `r_X[p] = Z_X[p] − consensus[p]`.
- **Two metrics per pair** (over proteins present in both networks and in ≥1 main network):
  - `average_magnitude_shift` = mean ‖Z_A[p] − Z_B[p]‖ (how far proteins move).
  - `average_cosine_similarity` = unweighted mean of cos(r_A[p], r_B[p]) (whether they move the same way).
- **The headline `control_summary.tsv` is MAGNITUDE-ONLY** (`avg_magnitude_shift` + `sd_magnitude_shift`).
  It answers the only question the ladder is for — *how much does varying each factor shift the embedding* —
  with one uniform, degeneracy-free number per control, so a–m sit on a single comparable effect-size axis.
  **Direction is deliberately excluded** from this table: cosine is a separate, harder question (does the
  shift *reproduce* / point the same way), and every centered cosine has a degeneracy or shared-offset
  caveat (see "Two centering schemes"). Per-pair cosines are still written to `control_pairs.tsv` as raw
  data, and the context-matched direction analysis lives in its own `control_centered_summary.tsv`.
- **Overlap column (required).** Each pair also reports `n_proteins` and `jaccard` (node overlap). Cosine
  correlates with overlap at r≈0.93, so every cross-tier comparison must be read against overlap — raw
  cosine alone re-encodes node membership, not the factor.
- **within-study / between-study label** per pair: "within" = both networks from the same dataset;
  "between" = different datasets. For the noise controls (a, b, h, i) this is the noise axis; for the
  factor effects (c–g) it flags whether the factor effect is study-confounded.

## The controls

Disease-arm controls (a–g) and healthy-arm controls (h–j). Healthy controls use the **normal** cells only
(disease=='healthy' in the tag) and are the disease-independent counterpart of the noise/factor floors;
mixed disease-vs-healthy pairs are not part of the ladder (that is the healthy-centering axis).

| id | hold constant | vary | pooling | study label | estimates |
|----|---------------|------|---------|-------------|-----------|
| **a** | disease, tissue, cell type, cell state | donors | split each study's donors in half, compare halves | within-study | inter-donor floor (disease arm) |
| **b** | disease, tissue, cell type, cell state | study | compare unique studies of the same context | between-study | study/batch floor (disease arm) |
| **c** | disease, tissue, cell type, study | cell state | pool donors per state | within-study | cell-state effect |
| **d** | disease, tissue, study | cell type | pool states | within-study | cell-type effect |
| **e** | disease, study | tissue | pool cell types | within-study | tissue effect |
| **g** | tissue, cell type, cell state | disease | diseases from separate datasets | between-study | disease effect, study-confounded |
| **h** | healthy, tissue, cell type, cell state | donors | split a study's healthy donors in half | within-study | inter-donor floor (healthy arm) |
| **i** | healthy, tissue, cell type, cell state | study | compare unique healthy studies of the same context | between-study | study/batch floor (healthy arm) |
| **j** | healthy, study, tissue | cell type | pool states | within-study | cell-type effect (healthy arm) |
| **m** | healthy, tissue, cell type, cell state | which study held out | pool of N−1 studies vs the held-out study | between-study | does **pooling** studies reduce between-study batch? |

**Control m (healthy leave-one-study-out).** For a context with ≥3 healthy study groups, compare a pool of
N−1 studies (`healthy_loopool<g>…`) to the held-out study (`healthy_loosingle<g>…`). On the magnitude ladder
(lower = less shift = better agreement): `m < i` ⇒ pooling reduces batch; `m ≈ i` ⇒ it doesn't. **Observed
(magnitude): m (0.31) ≈ i (0.32) ≫ h (0.21)** — naive (cell-mean) pooling does *not* shrink the
between-study batch, because the pool is dominated by its deepest/largest study. Built by
`build_healthy_loo.py`, scored by `control_m_healthy_loo.py`; `compare_controls.py` ignores `_loo` tags.
**Replaces the old ad-hoc `pool_vs_pool` control** (now removed).

**Reading the floors.** donor-split (a, h) splits *different donors* into the two halves, so it captures
**inter-donor variation (biological + technical), not pure resampling** — a true technical floor would split
one donor's cells. So the floor ladder is: donor-split (inter-donor) < between-study (inter-donor + batch/
cohort); the a→b and h→i increment is the **study/batch** component on top of donor variation.

**Healthy controls are the key reference for the disease claim.** Healthy between-study shift (i, magnitude
**0.32**) is about as large as disease between-study (b, **0.29**) — so the cross-study spread is intrinsic
cohort/batch, not disease. Cross-disease g (**0.38**, n=1) sits only modestly above this batch floor and is a
single, study-confounded pair, so it cannot separate disease from batch on magnitude alone. The disease claim
then requires **healthy-centering** (each disease network vs its own-study healthy baseline), not just
clearing the floor.

**The disease contrast (g) is between-study only.** No source carries two diseases in one dataset, so
there is no within-study disease control to isolate disease from study. The study confound on g is
handled statistically — compare g at matched node overlap against the cos~jaccard line set by the noise
controls (a, b, h, i), not by a within-study disease network.

**Study identity (donor-overlap grouping).** The integrated pools are already cell-deduped, but the pan-GI
depositions (40a0ade8/80a2c5b6/e6aaf5a4) share **donors** (same patients, different cells) — exact-cell
hashing misses this. So for the between-study controls (b, i) dataset_ids that **share donors** are
collapsed into one study (connected components; study8 = largest-cell member), with cell-hashing kept as a
safety net. This stops a between-study comparison from comparing a collection against itself.

## Direction (cosine) — a separate control set, NOT in the magnitude summary

The headline ladder above is magnitude-only. **Direction** (does the shift reproduce / point the same way)
is a distinct question with its own caveats, kept in separate files so it never contaminates the
effect-size table. The control cosines depend on what the per-protein deviation is measured against; we
compute two centering schemes.

**(1) Global consensus** — `compare_controls.py` (+ `control_k/l/m`). Deviation `r = Z − consensus`, where
`consensus` = per-protein mean over ALL main networks (disease + healthy). Simple, but the consensus is
dominated by cell-type/state/arm structure, so a shared arm offset can inflate/flatten the cosines.

**(2) Context-matched centering** — `control_centered.py`. Each control pair is centered on the MAIN
network of its **held-constant context** `(arm, tissue, celltype, state)`, so the cosine isolates the
**varied** factor. Centroid is always a MAIN network's `Z` (tests whether the mains are an accurate center):

| control | varies | centroid(s) |
|---|---|---|
| a donor_split | donors | disease-main of context **and** healthy-main of context |
| b between_study (same disease) | study | disease-main **and** healthy-main |
| c cell_state | state | mean of per-state disease-mains **and** per-state healthy-mains |
| different_disease (renamed from g) | disease | healthy-main of context (cross-disease → no disease centroid) |
| h healthy_donor_split | donors | healthy-main of context |
| i healthy_between_study | study | healthy-main of context |
| m healthy_loo | study | healthy-main of context |
| d cell_type | cell type | none — direct MAIN-vs-MAIN cosine over the node intersection |
| e tissue | tissue | none — direct MAIN-vs-MAIN cosine (same cell-type node set) |

Outputs (standalone, separate from the magnitude summary): `controls/control_centered_summary.tsv`
(`ctxcos_<centroid>` per control) / `controls/control_centered_pairs.tsv`. The global-consensus per-pair
cosines remain in `controls/control_pairs.tsv`.

**Caveat — own-arm context-centering is degenerate at small n.** Centering a small set of same-context nets
(e.g. control `b`'s two same-disease studies) on *their own* pooled main yields **negative** cosines, because
the main ≈ the mean of the compared nets, so their deviations are anti-correlated by construction (two nets →
cos→−1). So the **own-arm** variants (a/b on disease-main; h/i on healthy-main) are not interpretable at small
n. The informative variant is **cross-arm**: disease controls centered on the **healthy-main**. Observed:
`a` (donor halves) = **0.53** — the within-study disease shift reproduces across donor halves; `b` (across
studies) = **0.07** — that disease shift does NOT reproduce across studies (batch washes it out). `d`/`e`
direct cosines (~0.93–0.96) are inflated by the shared baseline and are not meaningful on their own.

## ComBat batch correction of the expression feature

The encoder's only study-varying input is the per-node `expression` feature (`log1p` mean CP10k); edges
(OmniPath) and healthy sender-weights (1.0) are study-independent. So the between-study batch lives in the
expression feature, and we test whether removing it collapses the batch floors (`b`/`i`/`m`) toward the
within-study donor floor (`a`/`h`) while preserving the biological factors (`c`–`g`).

**Method** (`scripts/build/controls/apply_combat_expression.py`): per `(tissue, celltype)` group, treat each
single-study control network's expression column as a pseudobulk sample; run a ComBat location/scale
adjustment with **batch = `study8`** and **`arm` + `state` preserved** as covariates (so disease-vs-healthy
and cell-state structure are kept, only cross-study offset is removed). Per-group fitting keeps tissue/celltype
constant, sidestepping the study↔context rank-deficiency. Pooled main nets and `loopool` get the mean per-gene
correction *delta* over their constituent studies (preserves their own magnitude, applies only the batch shift).
Single-study groups (brain fibroblast, lung macrophage) and singleton batches are left uncorrected. Writes a
new build dir; then retrain the encoder, re-infer, re-run `compare_controls`/`control_centered`.

Implementation notes: we use a **non-iterative** ComBat L/S (Johnson et al. 2007, **without** the empirical-Bayes
shrinkage) because scanpy's parametric EB solver **fails to converge / hangs** on these small, sparse pseudobulk
groups; `lstsq` makes it robust to arm↔batch collinearity (confounded variation is *preserved*, i.e. kept in
`stand_mean` — the correct identifiability behavior). **Default is LOCATION-ONLY** (remove the additive per-batch
mean `γ` only); `--with-scale` adds the multiplicative `δ`. At our batch sizes (2–3 samples) the `δ` variance
estimate is unreliable, so we drop it.

**Result (magnitude ladder; new build dirs `..._combat` = L/S, `..._combat_loc` = location-only):**

| control | what | orig | combat L/S | combat LOC |
|---|---|---|---|---|
| a | donor floor (disease) | 0.153 | 0.176 | 0.164 |
| h | donor floor (healthy) | 0.208 | 0.214 | 0.218 |
| **b** | batch (disease between-study) | 0.289 | 0.186 | **0.164** |
| **i** | batch (healthy between-study) | 0.318 | 0.182 | **0.191** |
| **m** | batch (healthy LOO) | 0.306 | 0.196 | **0.202** |
| c | cell-state | 0.228 | 0.253 | 0.247 |
| e | tissue | 0.274 | 0.237 | 0.245 |
| g | disease (n=1) | 0.378 | 0.404 | 0.399 |
| d / j | cell-type | 0.59 | 0.63 / 0.65 | 0.65 / 0.67 |

Cross-study **direction** reproducibility (`ctxcos_healthy_main`): `b` **0.07 → 0.48**, `g` **0.12 → 0.47**.

**Reading it.** ComBat does exactly what batch correction should: the three between-study floors (`b`/`i`/`m`)
collapse from well above the donor floor down to ≈ it (i 0.32→0.19, m 0.31→0.20 ≈ h 0.21; disease b 0.29→0.16 = a),
the within-study donor floors barely move, and the biological factors (`c`/`d`/`g`/`j`) are preserved or rise.
Once batch is removed, the disease shift **reproduces across studies** (direction cosine jumps from ~0 to ~0.47).
**Location-only ≈ location+scale everywhere** → the scale step added risk, not signal; location-only is canonical.
The `i`/`m` floors land *just below* `h` under **both** variants, so this is **not** a scale over-shrink artifact —
it is the expected consequence of per-gene mean removal (a full-study pseudobulk's residual sampling noise is
smaller than the donor-*half* noise in `h`).

**Caveat (unchanged, structural).** `g` is n=1 and, for single-study diseases, disease is confounded with study,
so arm-preservation *protects* that contrast rather than separating disease from batch. The clean, defensible
claim is the **healthy / multi-study** result: batch is removable to the donor floor, and removing it makes the
disease shift reproduce across studies. The single-study "g beats batch" claim still requires a within-study
disease control (not currently available).

## Known data gaps (verify before building)

- **b** needs ≥2 studies for the same (disease, tissue, cell type, state) — only Crohn macrophage and Alz
  microglia currently qualify; single-study contexts produce no (b) row.
- **e** needs one study spanning ≥2 tissues for a disease — current ileum/colon Crohn data are different
  datasets, so (e) may be empty unless such a study is fetched.
- **g** at fixed (tissue, cell type, state) reduces to roughly the single Crohn-colon-vs-UC contrast.

## Where the code lives

The a–j pipeline (current): **build → infer → compare**, all writing to
`results/crohn_alzheimer_ild_uc_embedding_expressed/controls/`.

- Build control networks: `scripts/build/controls/`
  - `build_pooled_controls.py` — disease-arm a/b/c/d/e networks (per dataset×state + allstates + splits).
  - `build_healthy_controls.py` — healthy-arm h/i/j networks (donor-overlap study grouping + cell-hash dedup).
  - `apply_combat_expression.py` — ComBat batch-correct the expression feature into a new build dir
    (location-only by default; `--with-scale` for L/S). See the ComBat section above.
- Infer through the frozen encoder: `scripts/embed/infer_controls.py` → `controls/control_embeddings.npz`.
- Classify pairs + metrics: `scripts/analysis/compare_controls.py` → `controls/control_pairs.tsv`,
  `controls/control_summary.tsv`.

Legacy/earlier control code (superseded by the above): `scripts/build/controls/`
(`build_control_experiments.py`, `build_donor_split_controls.py`, `build_alz_microglia_loo.py`,
`build_alz_microglia_replicates.py`, `build_crohn_rep.py`); `scripts/analysis/`
(`dump_control_comparison*.py`, `control_overlap_vs_cosine.py`, `compare_floor_vs_factors.py`).
