# SEQ_CONTEXT_EMBED.md — context-specific protein embedding via regulatory-neighbor link prediction

**Status: SCENIC networks DONE (2026-07-08); link-prediction training next.** A new modeling direction, distinct from the
graph joint-embed lineage in `HISTORY.md`/`CONTEXT_EMBED.md`. A protein's context-specific representation is
learned from its **frozen sequence embedding** plus **learned context-label embeddings**, and supervised by
**predicting its regulatory neighbors** (SCENIC-inferred TF→target edges) in that context. Working dir:
`mlp_mods/seq_context/` (top-level; source data — the staging h5ad and `protein_function.tsv` — stays in `de_ppi`).

## Goal / deliverable
One reusable **context-specific protein embedding**: the same protein (fixed sequence) adopts a different
vector in each (cell type, disease, tissue, cell state) context, while preserving sequence identity. The
embedding is the product; the prediction task is only the training signal.

**Validation questions the embedding must answer (Step 3):**
- Do drug targets **cluster** together in embedding space? (3a — geometry/retrieval)
- **Is there a decision boundary separating drug targets from non-targets?** (3b — supervised classifier)
- Does any of that signal beat the baselines (degree, DE, and crucially **ESM/sequence**), and is it
  **context-specific** (present in the relevant context, absent in healthy) rather than sequence-driven?

## Architecture (chosen)
```
Protein sequence ──► ESM frozen encoder ──► protein embedding (1280-D)
                                                      │
Context labels ──►  cell_type  Embedding (64-D)       │
                    disease    Embedding (32-D)        │
                    tissue     Embedding (32-D)        │
                    cell_state Embedding (32-D)        │
                          └──────────► concat ◄────────┘
                                         │
                                   MLP context encoder
                                         │
                                         ▼
                        Context-specific protein embedding
                        (e.g. STAT1 in Crohn's macrophages)
                                         │
                                         ▼
                     Context-conditioned link prediction:
                     is (protein → candidate) a regulatory edge
                     in THIS context's SCENIC network?
                                         │
                                         ▼
                        Binary cross-entropy over edges
```

### Inputs
- **Protein sequence** → frozen **ESM** (1280-D), `ESM/protein_embeddings.pt` + `protein_mapping.tsv`. Fixed
  per protein; fine-tuning deferred.
- **Context** = four categorical labels → trainable `nn.Embedding`: cell_type (64), disease (32), tissue (32),
  cell_state (32), learned jointly.

### Objective — regulatory-neighbor link prediction
For a context *c*, SCENIC gives directed edges TF→target. The context-specific embeddings of a source and a
candidate target are scored by a decoder; **BCE** trains the encoder to reconstruct the context's edges (present
edges positive, sampled non-edges negative). The learned per-protein context embedding is extracted and kept.

### Why this objective (and not the others considered)
Decision principle: **the training label must vary across context for a fixed protein**, or the context inputs
get no gradient and the embedding isn't context-specific. Of the candidates:
- **Regulatory neighbors (SCENIC), context-inferred — chosen.** The only context-*varying* label. Set-valued,
  so it fits link prediction (not a scalar decoder), and a richer signal than v0's single expression scalar.
- Pathway *membership* / functional annotation / protein complexes — **static** (same in every context) → cannot
  teach context; kept only as a **biology-retention probe** (below), not a training target.
- Expression scalar (earlier v0 plan) — context-varying and non-leaky, but a single scalar under-determines the
  embedding; superseded by the set-valued regulatory target.

**Accepted limitations (explicitly out of scope):**
- **No generalization to unseen contexts.** Contexts are named (ID embeddings), not described, so a context not
  seen in training has no vector. We do **not** care about this — only the built contexts matter. (A "leave a
  whole context out" eval is therefore N/A by construction; it was never achievable off a name.)
- **The context signal in SCENIC edges is co-expression** in that context's cells, so the labels are a
  transform of expression, not external ground truth. This is **not leakage** as long as expression is not a
  model input (it isn't). It does mean the model learns to reconstruct a co-expression-derived wiring; the
  meaningful test is the **context-lift** (below), not raw edge AUC (degree alone reconstructs edges well —
  link AUC was 0.90 and topology-bound in the graph lineage).
- **TF→target is directed / bipartite-ish.** Sources are TFs; a non-TF protein appears only as a target (its
  embedding is shaped by "who regulates me"). `cell_type` is currently constant (scVI build is macrophage-only)
  so that embedding is a no-op until more cell types are staged — wired in for later.

## Evaluation
- **Primary — context-lift, not raw AUC.** Compare the full model to a **context-blind** ablation (context
  embeddings zeroed / removed). The claim is only supported if conditioning on context measurably improves
  held-out edge prediction over the context-blind model and over a degree baseline.
- **Held-out regimes (within seen contexts only):** held-out edges, and held-out proteins (cold-start protein
  placed by ESM). No leave-one-context-out (N/A, above).
- **Biology-retention probe (GO/pathways).** Static labels can't train context, but guard against the embedding
  losing biological meaning: after training, linear-probe the **per-protein mean embedding** (the invariant
  component) to GO/Reactome terms; require AUC/AP **≥ raw ESM**. Optional small-weight auxiliary GO head later.
- **Downstream gate (the real point).** Does the context embedding's disease-vs-healthy delta improve the
  supervised ΔZ→OpenTargets head (`embedding_target_cv.py`, LOSO) over degree+DE? (see HISTORY.md.)

## SCENIC network generation  (`seq_context/scenic/`)
Produce one regulatory network per context to serve as link-prediction labels.
- **Cells:** `results/crohn_alzheimer_ild_uc_embedding_expressed_scvi/scvi_staging/macrophage.h5ad` — 115,141
  macrophage cells, raw `counts` layer, 6,820-gene expressed universe, obs `disease`/`state` (+ `source`,
  `study8`, `leiden`). **Tissue is NOT present** (Crohn colon vs ileum not separable here) — see open decisions.
- **Tool:** GRNBoost2 (co-expression TF→target, `scenic` env) → [optional] cisTarget motif pruning (`pyscenic`
  env + v10 hg38 gene-based DBs in `db/cistarget/`) → regulons. Both installed. See **Data preprocessing** below.
- **Per context:** subset cells → run SCENIC → write directed edge list (`seq_context/scenic/networks/
  <context>/edges.tsv`, columns `tf,target,importance[,sign]`) restricted to the 6,820-gene node universe.
- **Input = raw `counts` (by design).** GRNBoost2 infers regulation from real cell-to-cell co-variation; scVI
  batch-corrected/decoded values are a smoothed latent reconstruction that manufactures co-expression and is
  advised against for GRN inference. So counts, not the scVI-corrected feature.
- **Viability caveat:** co-expression inference needs enough cells; `proliferating` (4,502 total, before disease
  split) may be too sparse for a stable network in some contexts.
- **Batch caveat + v2 escalation.** Contexts pool multiple studies, so some co-expression is study-driven:
  lung (`ild_lung`, `healthy_lung`) = **single study, clean**; Crohn = 2 studies; UC colon = 3; healthy colon =
  4 (most exposed). Raw counts don't fix this. The principled fix is **per-study GRNBoost2 + consensus** (keep
  edges reproducible across a context's studies) — deferred to **v2**, invoked only if the downstream
  context-lift looks like it rides on colon batch structure. Not scVI-decoded values.

## Data preprocessing (THIS VERSION — decided 2026-07-09)
End-to-end recipe that turns scRNA-seq into the per-context SCENIC label networks. A **context = disease × tissue
× cell_type × state**.

1. **Cells / source.** Per cell type, the scVI **staging h5ad** (`de_ppi/results/..._expressed_scvi/scvi_staging/
   <celltype>.h5ad`) — provides the **raw `counts` layer**, harmonized `disease`/`source`, and the expressed-gene
   universe. Built for macrophage, fibroblast, microglia, stem.
   - **scVI is now OPTIONAL for this pipeline** (kept only because the staging objects already exist): SCENIC uses
     **raw counts** (not scVI-corrected), the context model uses **ESM + context IDs only** (no expression), and
     states come from marker scoring (below) — so scVI's corrected expression is never used downstream. Its lone
     remaining role was integrated Leiden state definition, which marker scoring replaces (and marker scoring is
     more batch-robust than clustering). Future data could skip scVI entirely.
2. **Disease → arm** (`DISEASE_ARM`): normal→healthy, Crohn disease→crohn, ulcerative colitis→uc, interstitial
   lung disease→ild, Alzheimer disease→alz.
3. **Source → tissue** (`SRC_TISSUE`; staging obs carries no tissue): macrophage per-source (lung/colon/ileum);
   fibroblast_crohn & stem_crohn→intestine; fibroblast/microglia alzheimer sources→brain (cortical regions
   collapsed).
4. **Cell state.**
   - **macrophage:** 3 named states (inflammatory/resident/proliferating) from marker-panel argmax over Leiden
     clusters (`run_scvi.py SIGS`) — pre-existing.
   - **fibroblast / microglia / stem:** PUBLISHED per-state marker signatures in `scenic/state_markers.tsv`
     (Keren-Shaul 2017 DAM microglia; Haber 2017 intestinal epithelium; Smillie 2019 / Kinchen 2018 colon
     fibroblast; tissue-aware — gut cells scored vs gut panels, brain cells vs brain panels). Each cell scored
     (`score_genes`) → argmax state. **Replaces the arbitrary per-Leiden-cluster states** that caused the
     40–47-context explosion. NOT scVI-Leiden-argmax, NOT CellTypist/CELLxGENE (both cell-*type* only, no states).
5. **Context assembly + floor.** Group cells by (arm, tissue, cell_type, state); keep contexts with **≥ 50 cells**
   (`MIN_CELLS`). Export per context the raw-count matrix (`prep_contexts.py --celltype`).
6. **SCENIC label networks** (`run_grnboost2.py --celltype`, cap **5,000 cells**/context, raw counts):
   - **GRNBoost2** co-expression (`scenic` mamba env, arboreto) → `edges.tsv` (raw ranked `tf,target,importance`).
   - **Positive labels** = each TF's **top-50 targets** (`threshold_topk.py` → `edges_topk.tsv`; trainer also
     enforces top-50 at load).
   - **cisTarget (optional, cleaner labels)** — `cistarget_prune.py` (`pyscenic` env) prunes GRNBoost2 modules to
     motif-supported regulons using **v10 hg38 gene-based ranking DBs** (`db/cistarget/`) → `edges_cistarget.tsv`.
     Trade: ~74% fewer source TFs but only ~5% fewer trainable proteins in well-powered contexts. Trainer switch:
     `--labels {topk|cistarget}`.
7. **Node universe / trainability.** Nodes = ESM-covered proteins appearing as an edge endpoint (~81% of expressed
   genes per context; ~83% ESM coverage). TFs are the only sources; non-TFs appear as targets.

**Batch caveat (unchanged, v2 escalation):** colon contexts pool 2–4 studies → some co-expression is study-driven;
lung/ILD single-study clean. Fix = per-study GRNBoost2 consensus, deferred to v2.

**Environments:** `.venv_scvi` (staging/prep/threshold/train, torch+CUDA) · `scenic` mamba (arboreto/GRNBoost2,
py3.10 pre-dask-expr stack) · `pyscenic` mamba (cisTarget, py3.10 + setuptools<81 for pkg_resources).

## Layout
```
mlp_mods/seq_context/                       # top-level workspace (source data stays in mlp_mods/de_ppi/)
  SEQ_CONTEXT_EMBED.md                       # this doc
  scenic/
    state_markers.tsv                        # published per-state marker panels (fibroblast/microglia/stem)
    scripts/     # prep_contexts.py, run_grnboost2.py, threshold_topk.py, cistarget_prune.py, build_celltypes.sh
    inputs/      # macrophage flat; other cell types under inputs/<celltype>/ (raw-count matrices + genes/tfs)
    networks/    # <tag>/{edges.tsv raw, edges_topk.tsv GRNBoost2 labels, edges_cistarget.tsv motif labels}
  scripts/       # train_link_context.py (ESM+context encoder, BCE link pred; --labels topk|cistarget)
  results/       # <run>/{encoder... , embeddings.npz, metrics.json}; images/<run>_curve.png
db/cistarget/                                # v10 hg38 gene-based ranking DBs + motif annotation
```

## Generated networks (2026-07-08)
GRNBoost2-only, `.venv`→mamba env `scenic` (py3.10, pandas1.5/dask2023.5/numpy1.23; arboreto 0.1.6 breaks on
modern dask). Runner `seq_context/scenic/scripts/run_grnboost2.py`, cap 5,000 cells/context, seed 0, 8 workers.
All **18/18** contexts written to `seq_context/scenic/networks/<tag>/edges.tsv` (`tf,target,importance`), 765 MB
total, ~1h05 wall. Edge counts 0.27M–1.95M (RAW ranked adjacency, ~50% dense — threshold at label-build time,
NOT finished regulons). Sanity: top edges are real (e.g. HSPA1A↔DNAJB1). TFs per context 249–491 (of 516),
genes 3,054–6,046. Smallest/weakest: crohn_colon_proliferating (103 cells), healthy_colon_inflammatory (119),
crohn_ileum_inflammatory (178) — low-cell, treat as low-confidence.

## Positive labels — top-k thresholding (2026-07-08, done)
`threshold_topk.py` keeps each TF's **top-50 targets by importance** (FIXED k across all 18 contexts for fair
context-lift), writing `networks/<tag>/edges_topk.tsv` + `networks/topk_summary.tsv`. Positives per context
**12,450–24,550** edges (~50 targets/TF; density ~1% vs ~50% raw). k is sweepable without re-running GRNBoost2.
BCE negatives to be sampled from non-top-k pairs (some hard = mid-importance) at train time.

## v1 result (2026-07-09, `results/link_v1/`) — no context lift, but eval is the culprit
`train_link_context.py`, 18 contexts, 728,558 examples (1:1 RANDOM negatives), ESM-covered nodes, 128-D, 15 ep, GPU.
Held-out edges: **FULL AUC 0.647 / AP 0.653; BLIND (context zeroed) 0.657 / 0.667; DEGREE 0.601 / 0.601;
context-lift −0.010 / −0.013** (slightly negative). ESM beats degree modestly; context conditioning did NOT help.
**But the eval can't detect context by construction:** with random negatives, a positive pair (88.5% of edges
are UNIQUE to one context; 0% shared by all 18; median 1 ctx/edge) is essentially never a negative elsewhere, so
a context-blind model labels it positive without using context. The negative lift is therefore uninformative,
NOT evidence context is useless — the label set is strongly context-specific.

## v1.1 result (2026-07-09, `results/link_v1_hardneg/`, `--neg hard`) — context helps, small over degree, undertrained
HARD cross-context negatives: for context c, negatives = pairs POSITIVE in another context but NOT in c AND
OUTSIDE c's top-300/TF (`PROTECT_TOPN`, so rank-51..300 sub-threshold real edges never leak in as false
negatives); endpoints present in c. Same (TF→target) is then positive in one context, negative in another → the
model MUST use context; a context-blind model scores them identically. Held-out edges (15 ep):
**FULL AUC 0.596 / AP 0.613; BLIND 0.558 / 0.588; DEGREE 0.583 / 0.586; context-lift +0.038 / +0.025.**
(Pre-margin version was FULL 0.583 / lift +0.032; the top-300 margin lifted everything slightly.)
Reading: (1) **context-lift POSITIVE** and BLIND collapsed toward chance as predicted → context-specific signal
is REAL and partly learned; v1's random-neg null was an eval artifact, confirmed. (2) Absolute AUC weak (0.596)
and **degree (0.583) ≈ FULL** → still mostly topology, thin sequence+context margin (same ceiling as graph
lineage). (3) **Curve (`images/link_v1_hardneg_curve.png`): full val-AUC still CLIMBING at ep15 (undertrained);
blind plateaus early ~0.555; full–blind gap widens with training** → train longer (more epochs) is the cheapest
next lever, likely widens the lift.

## v1.1 converged (2026-07-09, `results/link_v1_hardneg_long/`, 60 ep) — context real, modest ceiling
60-epoch run (DEGREE baseline removed per user; only FULL vs context-BLIND reported): **FULL AUC 0.618 / AP 0.633;
BLIND AUC 0.509 / AP 0.552; context-lift +0.109 / +0.081.** Curve `images/link_v1_hardneg_long_curve.png`:
- **FULL val-AUC plateaus ~0.618 by epoch ~40** → converged; the architecture+data ceiling (more epochs won't help).
- **BLIND val-AUC peaks ~0.558 (ep 8) then declines to 0.509** → the correct 0.5 floor (blind CANNOT separate
  context-flipped pairs); confirms the hard-neg test is clean. The +0.109 lift is mostly blind falling to floor.
- **Overfits LOSS not RANKING:** full val-loss rises after ~ep20 while val-AUC stays flat → AUC is the fair readout;
  bigger net / more epochs won't lift it. Honest: context is genuinely used (0.618 vs 0.5 floor) but discrimination
  is moderate. Pushing higher needs better FEATURES (motif-grounded regulons, more cell types), not capacity/epochs.

## Tuning sweep (2026-07-09, all hard-neg, 60 ep) — FULL test AUC
| run | change | FULL AUC | BLIND | lift |
|---|---|---|---|---|
| link_v1_hardneg_long | fixed negatives | 0.618 | 0.509 | +0.109 |
| link_v2_resample | **negatives resampled/epoch** | 0.651 | 0.547 | +0.104 |
| link_v3_k75 | resample + **positives k=50→75** | 0.659 | 0.551 | +0.108 |
| **link_v4_cistarget** | resample + **cisTarget labels** (`--labels cistarget`) | **0.749** | 0.600 | **+0.149** |

**cisTarget labels were the biggest lever (+0.090 AUC over GRNBoost2, lift +0.108→+0.149).** Reading: the ~0.65
"ceiling" was partly **label noise**, not pure topology — motif-grounded regulons are cleaner and more learnable,
and context matters MORE with clean labels (lift grew). Caveat: BLIND also rose (0.551→0.600, cleaner labels are
more predictable for any model) but FULL rose more; costs ~16% coverage (fewer trainable proteins). Updates the
project-wide "topology-bound" story: **label quality is a real lever.**
Per-epoch negative resampling was the bigger win (+0.033, and it KILLED the harmful val-loss rise — loss now
plateaus ~0.65 instead of climbing to 0.76; residual train-val gap is benign, val-AUC stable/climbing). k=75
positives added +0.008 (more data). Both curves: full val-AUC still creeping up at ep60 (slightly undertrained).
NOTE: `edges_topk.tsv` now regenerated at **k=75** (k=50 restorable via `threshold_topk.py --k 50`).

## Step 3 — target-recovery validation (2026-07-09, `validation/`)
On the cisTarget Crohn macrophage embedding (`link_v4_cistarget`, `crohn_colon_macrophage_inflammatory`). Bar =
beat degree / ESM / degree+DE. (Steps 1 func-class-retention and 2 pathway are deferred; drug data was local.)

Label = each disease's known drug targets, all phases. Run **per-context across all 11 disease contexts** (not just
one — an earlier single-context read of crohn_colon_inflammatory was cherry-picked and overstated).

**3a — centroid LOO retrieval** (`centroid_retrieval.py`, `centroid_percontext_*.tsv`): seed = disease drug targets
present; LOO centroid, cosine-rank; controls = degree, ESM, permutation null.
- **Only crohn_colon (inflammatory emb 0.693 p=0.007; resident 0.680 p=0.011) passes all 3 controls — 2/11.**
  Elsewhere (crohn_ileum, ild_lung, most uc_colon) emb ≈ 0.42–0.61, **below ESM and often below chance.**
  **Median across contexts: emb 0.470, esm 0.559.** So targets do NOT robustly cluster; crohn_colon is an outlier
  (likely annotation depth, not general mechanism).

**3b — supervised classifier** (`target_classifier.py`, `classifier_percontext_*.tsv`, L2 LogReg balanced, `--drugfile`,
optional `--fix-disease`): 5×5 CV AUC, features EMB / BASE(degree+DE) / ESM.
- Per-context (own-disease label): **median BASE 0.460, EMB 0.703, ESM 0.676. EMB > BASE in 11/11** — targets are
  linearly *separable* even where they don't *cluster*. EMB ≈ ESM. BASE is weak (drug targets are **anti-hubs** in the
  cisTarget net, unlike OmniPath).

**Clean test — FIX Crohn label, VARY context (`--fix-disease crohn`, 18 contexts):** isolates context relevance from
the disease-target confound. Result: **EMB does NOT peak in crohn_colon** (uc_colon 0.81, crohn_ileum 0.79, ild_lung
0.72, healthy_lung_resident 0.72 all comparable), and **ESM ≥ EMB on median (0.715 vs 0.690)**. Even *healthy* contexts
(no disease relevance) recover the Crohn targets → recovery is the **context-invariant sequence/druggability axis**,
not context relevance. The faint real signal: **EMB > ESM in gut-disease contexts, EMB ≤ ESM in healthy** (6/7).

**Two confounds documented:**
- **Cross-disease drug overlap:** Crohn∩UC drug targets = **84%** (only 4 Crohn-unique) → per-disease "specificity" is
  untestable; the fixed-label test is the valid design.
- **ESM/compression:** EMB is a **128-d bottleneck compressing 1280-d ESM + context**, so EMB is a *lossy* ESM plus
  whatever context adds — explains EMB<ESM in healthy (compression loss, no context gain) and EMB>ESM in gut-disease
  (context pays it back). Resizing the bottleneck only tunes ESM-retention, not context signal — not worth it.

**Phase boxplot** (`crohn_phase_boxplot.py`, `images/crohn_phase_boxplot.png`): out-of-fold P(target) for Crohn targets
by clinical phase vs 20 random controls — **targets not separated from controls, no phase gradient** (a control has the
highest P). Only phases 3–4 present among targets.

**Synthesis (honest, all-contexts):** the embedding beats degree+DE, but **mainly by carrying ESM's sequence/
druggability signal — it does not beat ESM, and context-specific target signal is NOT demonstrated** (targets cluster
only in crohn_colon; recovery is as strong in healthy as in disease). Consistent with the project-wide topology/
sequence-bound ceiling. Heavy caveat: 5–20 positives/context → noisy. Fair remaining test = **ΔZ (disease-shift)**, the
one signal ESM structurally cannot carry.

## Roadmap
- generate SCENIC networks ✅ · top-k labels ✅ · v1→v3 tuning ✅ · **v4 cisTarget labels (FULL 0.749, biggest lever) ✅** · Step-3 validation ✅.
- **In progress:** add cell types (fibroblast/microglia/stem) — marker-named states done; SCENIC on 22 new contexts running (activates cell_type).
- **Next:** ΔZ target-recovery (3a/3b on disease-shift); Step 2 pathway coherence (needs Reactome/MSigDB gene sets); 4-cell-type retrain.
- **Later:** GO func-class retention floor; per-study consensus (colon batch).
