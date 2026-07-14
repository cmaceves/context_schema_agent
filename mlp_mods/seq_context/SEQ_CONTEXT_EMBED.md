# SEQ_CONTEXT_EMBED.md — context-specific protein embedding via regulatory-neighbor link prediction

**Status (2026-07-10): v12 DONE (252 contexts = v11 203 + 49 B-cell; FULL test AUC 0.805 / AP 0.798). The training
objective is context-specific regulatory-neighbor link prediction, and the model is monotonically better at it as
data grows: FULL AUC v9 0.765 → v10 0.785 → v11 0.799 → v12 0.805. That IS the representation capturing more
context-specific regulatory structure — the primary, objective-aligned signal.**

**Do not call v12 "worse."** It regressed only on ONE downstream probe (drug-target recovery: MLP-EMB H@100 v11 47 →
v12 31; ESM unchanged ~71). That probe is weak and indirect — small positive set (~314), pooled, disease-agnostic,
ESM-dominated — a sanity check, not the measure of representation quality. A representation that improved on its actual
loss is not indicted by a flat/worse score on a downstream application. See "Framing (2026-07-10)" below.

Two caveats worth keeping: (1) FULL AUC is measured on a held-out **edge** split **within contexts the model trained
on** — in-distribution; it confirms the objective is learned, not that the structure generalizes to unseen contexts
or external tasks. (2) The sharper objective-aligned metric is the **context-lift** (FULL − BLIND AUC): raw AUC can
rise just from more protein-pair coverage, while the lift isolates what the context embeddings specifically add
(context-specific structure vs. protein identity + topology). v12 was run BLIND-off, so its lift isn't computed yet.

Classifier step (the downstream probe) is per-disease pool + max-agg, MLP-ONLY (LogReg dropped). MLP2 rank-loss
experiment FAILED and was removed. Trainer has early stopping (val-AUC, patience) + `--edge-weight`/`--protein-repr`/
`--blind` flags. B-cell + Bipolar-I SCENIC done (293 ctx total; bipolar = 3 glial/neuronal cell types, OT labels now
pulled = MONDO_0004985, 128 targets → bipolar is a validation arm).

**Verdict (2026-07-13): the frozen-readout approach isn't converting context into a useful representation.** A full
downstream battery (v13 edge-weighting, attention-MIL, per-disease/GO-class/MoA recovery, and 3 no-ML geometry tests)
all agree: the disease-context shift is real in *magnitude* (~0.4× inter-protein distance) but **incoherent in
direction** (target Δz ~orthogonal) and **non-recoverable** (disease-centroid LOO H@10 0–2); target recovery is
dominated by **ESM sequence-family membership**; every frozen-readout lever moves a proxy (e.g. v13 lift +0.234) while
the embedding gets no better (v13 recovery *down*). See "Downstream diagnostics (2026-07-13)" below.
Priority = the one untried structural lever: **FINE-TUNE THE ENCODER** (backprop into the representation) or a
context-*requiring* supervised target task.** See dated sections below. A modeling direction distinct from the graph joint-embed
lineage in `HISTORY.md`/`CONTEXT_EMBED.md`. A protein's context-specific representation is
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
   PROTEIN                                   CONTEXT (this cell state)
      │                          ┌──────────────┬──────────────┬──────────────┐
      ▼                       cell_type       disease         tissue         state
 ESM-2 (frozen)                  │              │              │              │
 1280-d                          ▼              ▼              ▼              ▼
      │                     nn.Embedding   nn.Embedding   nn.Embedding   nn.Embedding
 Linear 1280→256                 64-d           32-d           32-d           32-d      (learned)
 (ESM_PROJ, learned)               └────────────┴──────┬───────┴──────────────┘
      │                                       context vector (160-d)
      └───────────────────────────┬───────────────────┘
                                   ▼
                          concat  (256 + 160 = 416-d)
                                   ▼
                        MLP encoder: Linear 416→512 → ReLU → 512→512 → ReLU → 512→128
                                   ▼
      ┌───────────────────────────────────────────────────┐
      │  z : context-specific protein embedding  (128-d)   │  ◄── THE DELIVERABLE (embeddings.npz)
      │      STAT1-in-Crohn-colon ≠ STAT1-in-healthy-lung  │
      └───────────────────────────────────────────────────┘
                                   │        directed link-prediction decoder
              ┌────────────────────┴────────────────────┐
        z_TF ─► src_head(128→128)          z_target ─► tgt_head(128→128)
              └──────────────► dot + bias ◄─────────────┘
                                   ▼
              logit → sigmoid → P(TF regulates target in this context)
                                   ▼
        BCE  vs  SCENIC cisTarget edges (positives) + hard cross-context negatives (resampled/epoch)
```

| block | in → out | learned? | notes |
|---|---|---|---|
| ESM-2 | seq → **1280** | **frozen** | one fixed vector per protein (sequence/identity) |
| ESM projection | 1280 → **256** | yes | `ESM_PROJ=256` — down-project so ESM doesn't dimensionally dominate the 160-d context. (`--protein-repr id` swaps this for a random-init 256-d learned protein-ID embedding — no sequence) |
| cell_type / disease / tissue / state | id → **64 / 32 / 32 / 32** | yes | `nn.Embedding`; concat = **160-d** context vector |
| MLP encoder | **416** → 512 → 512 → **128** | yes | 2 hidden ReLU, `hidden=512`, `EMB_DIM=128` (input = 256 ESM-proj + 160 context) |
| **z** (output) | → **128** | — | context-specific protein embedding = deliverable |
| src_head / tgt_head + bias | 128 → 128, dot | yes | directed decoder (regulator vs target role) |

Training: `--epochs` (default varies per run, e.g. 60–80), Adam lr 1e-3, batch 8192, **early stopping** on val-AUC
(`--patience`, best weights restored); **only** ESM-projection + context embeddings + MLP + decoder heads learn (ESM
frozen). Flags: `--neg {random,hard}` (hard = cross-context negs, default for builds), `--labels {topk,cistarget}`
(cistarget = motif-pruned regulons), `--protein-repr {esm,id}`, `--edge-weight {none,inv_ctx}` (default **none**;
inv_ctx tried in v13, inflated lift but hurt recovery → not default), `--aux-pathway <λ>` (architecture B),
`--blind` (opt-in context-lift ablation).
Current instantiation (v12/v13): **293 contexts**, 9 cell types (tcell/endo/bcell/macrophage/fibroblast/microglia +
bipolar oligodendrocyte/astrocyte/glutamatergic_neuron), 8 disease axes + matched healthy.

## Architecture B — + pathway-activity auxiliary loss, PER-PROTEIN-PER-CONTEXT (BUILT 2026-07-13)
Same backbone as above (frozen ESM → proj → concat 4 context embeddings → MLP → z 128-d → link-pred decoder).
**New:** a second head reads **`z_{protein,context}`** (128-d, NOT the shared context vector) and predicts, for that
protein in that context, the **context-c activity of the pathways the protein belongs to**. This trains the full
encoder (protein-specific), grounding `z` in curated biology (Reactome) rather than only noisy co-expression edges.
Target is **membership-gated**: member pathways → their context activity, non-member pathways → 0, so the target is
protein-specific (forces `z` to encode identity × context, not just context).

```
   PROTEIN                                   CONTEXT (this cell state)
      │                          ┌──────────────┬──────────────┬──────────────┐
      ▼                       cell_type       disease         tissue         state
 ESM-2 (frozen)                  │              │              │              │
 1280-d                          ▼              ▼              ▼              ▼
      │                     nn.Embedding   nn.Embedding   nn.Embedding   nn.Embedding
 Linear 1280→256                 64-d           32-d           32-d           32-d      (learned)
 (ESM_PROJ, learned)               └────────────┴──────┬───────┴──────────────┘
      │                                       context vector (160-d) ──────────┐
      └───────────────────────────┬───────────────────┘                       │
                                   ▼                                           │
                          concat  (256 + 160 = 416-d)                          │
                                   ▼                                           │
                        MLP encoder: 416→512 → ReLU → 512→512 → ReLU → 512→128  │
                                   ▼                                           │
      ┌───────────────────────────────────────────────────┐                  │
      │  z : context-specific protein embedding  (128-d)   │  ◄── DELIVERABLE  │
      └───────────────────────────────────────────────────┘                  │
                                   │
        ┌──────────────────────────┴───────────────┐
        ▼  directed link-prediction decoder         ▼  AUXILIARY pathway head  ◄── NEW in B (reads z, per protein)
  z_TF ─►src_head(128→128)   z_tgt ─►tgt_head    for each endpoint z_{p,c}:  Linear 128→Np → p̂  (Np pathways)
        └────────► dot + bias ◄───────┘                        │
                     ▼                                          ▼
        logit → sigmoid → P(TF regulates target)     target_{p,c} = activity[c] ⊙ membership[p]
                     ▼                          (member pathways → context activity;  non-members → 0)
   L_SCENIC = BCE vs SCENIC cisTarget edges          L_pathway = mean_member (p̂−activity)²  +  mean_nonmember (p̂)²
      (+ hard cross-context negatives)                          ▲ per-protein-per-context (endpoints of the edge batch)
                     └───────────────────────┬──────────────────────┘
                                             ▼
                          L = L_SCENIC  +  λ · L_pathway      (λ ≈ 0.1)
                          (backbone trained jointly; ESM frozen)
```

| new piece | detail |
|---|---|
| **activity labels** | `scenic/pathway_activity.tsv` = **293 contexts × 1490 pathways**, z-scored. `gen_pathway_activity.py`: per context, CPM+log1p of `inputs/<ct>/<tag>/counts.npz`, per-gene mean over cells, center each gene across contexts, pathway score = mean centered-expr of member genes (≥5 present), keep pathways scored in ≥90% of contexts. (Fast mean-centered proxy for AUCell; swappable — trainer only reads the matrix.) |
| **membership** | built in-trainer from Reactome GMT, aligned to the activity columns and to the protein order → `M` (P × 1490), binary. Mean ~pathways/protein reported at launch. |
| **auxiliary head** | `Linear(128→1490)` reading **`z_{p,c}`** (per protein, per context) — computed on the **endpoints of each edge batch** (tf + target proteins in their context). |
| **loss** | `L = L_SCENIC + λ·L_pathway`, `L_pathway = mean_member (ẑ−activity[c])² + mean_nonmember (ẑ)²`, `λ≈0.1` (sweep 0.03–0.3). ESM frozen; rest trained jointly. |
| **flag** | `--aux-pathway <λ>` (0 = off = architecture A). Reads `pathway_activity.tsv`. |

Rationale + honest priors: **+** curated, lower-noise biology signal than raw regulons; **per-protein-per-context** so
it trains `z` (the deliverable) and forces identity × context, directly targeting the protein-specific failure mode.
**−** activity is expression-derived like the SCENIC edges (correlated, not independent — a cleaner view of the same
source, not new signal); membership is static (identity-derivable), so part of the objective just re-learns "which
pathways is this protein in." ESM stays frozen here — encoder fine-tuning is a separate lever. Alternative form
(not built): pathway activity as an **input** (5th context branch).

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
  embedding is shaped by "who regulates me"). `cell_type` now **varies** (macrophage + microglia + fibroblast;
  stem pending) — as of `link_v5_allct` it is a real input, no longer a no-op.

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

## Multi-cell-type + ESM-projection + factor ablation (2026-07-09)
- **v5 (`link_v5_allct`):** retrain on **32 cisTarget contexts, 3 cell types** (macrophage+microglia+fibroblast) —
  `cell_type` finally a real varying input. Link-pred FULL 0.759 / BLIND 0.632 / lift +0.127 (vs v4 macrophage-only
  0.749/0.600/+0.149). UMAP: protein-identity-dominated blob; cell type only a **faint** axis. 3b median EMB 0.689 ≤ ESM 0.700.
- **v6 (`link_v6_esmproj256`):** add learnable **ESM 1280→256 projection** (`ESM_PROJ`) so ESM doesn't dimensionally
  swamp the 160-d context. Metrics ~unchanged (FULL 0.755 / lift +0.129; 3b EMB 0.673 slightly worse) **but UMAP
  geometry rebalances — cell type separates more** (microglia distinct). Trade: more context-organized geometry,
  small cost to ESM-driven target recovery. (ESM_PROJ=256 is now the default architecture.)
- **Context-factor ablation** (`validation/ablation_context.py`, `ablation_v6_esmproj256.tsv`,
  `images/ablation_curve.png`) — zero each factor, held-out link-pred AUC drop vs full:
  **state 0.044 > disease 0.032 > tissue 0.019 ≫ cell_type −0.002 (none); blind(all) 0.127.**
  → **`cell_type` is redundant** (each cell type lives in specific tissues, so tissue+disease already encode it) —
  the dedicated embedding is dead weight. **`state` is the most important factor.** Drops are complementary
  (Σsingles 0.093 < 0.127 joint). Context-lift is driven by **state + disease**, not cell type.

## v7 — 4-cell-type retrain (2026-07-09, `results/link_v7_4ct/`)
Stem cisTarget completed → **40 cisTarget contexts, 4 cell types** (macrophage + microglia + fibroblast + stem),
hard negatives, 60 ep. **FULL AUC 0.764 / BLIND 0.641 / context-lift +0.123.** Current default build for the
downstream validation below.

## ΔZ disease-shift target recovery (2026-07-09)
ΔZ = z(`crohn_colon_macrophage_inflammatory`) − z(`healthy_colon_macrophage_inflammatory`), v7; proteins ranked by ‖ΔZ‖.
- **Recovery of OT>0.5 Crohn genes:** N=3,362 proteins present in BOTH contexts, 11 positives, **MRR 0.0022
  (random ≈0.0026), top-10% 2/11.**
- **On/off genes excluded by construction (ΔZ needs both contexts):** crohn context 4,070 nodes, healthy 3,614,
  intersection 3,362 → **708 crohn-only + 252 healthy-only proteins have no ΔZ.** Of OT>0.3 Crohn genes, **82 are
  in the intersection and 22 are crohn-only** (IL23A, TNFSF15, TLR4, IL6R, CCL2, CCL7, MMP9, ALOX5, SOCS1, CIITA…).
  This pipeline adds no zero-placeholder padding (unlike the de_ppi build), so induced/silenced genes are dropped.
- **‖ΔZ‖ vs degree / DE (3,362 proteins):** Spearman(‖ΔZ‖, out-degree)=**+0.174**, in-degree −0.015, |Δexpr|=+0.120.
  TFs (out-deg>0) mean ‖ΔZ‖ 3.75 vs non-TFs 2.50; top-50 movers = 22/50 TFs (TF base rate 3%). Top movers:
  REL, MYC, RELA, RPS24, KLF4, BHLHE40/41, FOS, HLA-B/-DQ.

## Pooled protein-disease embedding → target prediction (2026-07-09, `validation/`)
Mean-pool each protein's v7 embedding across contexts → one vector/protein; L2-LogReg / MLP (`class_weight=balanced`),
5-fold out-of-fold P, positives = all-phase drug targets. Metrics = H@10 / H@100 / MRR over the full ranking.
- **Crohn-pooled** (13 crohn contexts, `crohn_phase_boxplot_pooled.py` original version) — logreg:
  EMB(Crohn) 0/1/0.002, EMB(healthy) 0/1/0.001, ESM 3/5/0.088.
- **Global all-OT pooled** (`all_ot_target_pooled.py`; 40 contexts; 142 targets from 5 `known_drugs_*.tsv` =
  Crohn / UC / IBD / bronchiolitis obliterans / Alzheimer; 9,547 proteins) — logreg: EMB 0/4/0.002, ESM 4/27/0.021;
  mlp: EMB 4/12/0.017, ESM 9/29/0.041. Output overwrites `images/crohn_phase_boxplot_pooled[_mlp].png`
  (v8 variant suffixed `_link_v8_idrepr`).

## Combined ESM ⊕ pooled embedding (2026-07-09, `combined_esm_pooled_target.py`)
Concatenate ESM (1280-d) + global-pooled embedding (128-d); table `results/<run>/esm_plus_pooled_mrr.tsv`,
plots `images/esm_plus_pooled_target_<run>_<clf>.png`. MRR (H@100 in parens):

| run | classifier | ESM | EMB alone | ESM+EMB |
|---|---|---|---|---|
| v7 (ESM-based emb) | logreg | 0.0207 (27) | 0.0016 (4) | 0.0128 (29) |
| v7 | mlp | 0.0407 (29) | 0.0170 (12) | 0.0301 (29) |
| v8 (learned-ID emb) | logreg | 0.0207 (27) | 0.0007 (0) | 0.0308 (29) |
| v8 | mlp | 0.0407 (29) | 0.0009 (1) | 0.0378 (32) |

(ESM column identical across runs — same feature. n_pos=88.)

## kNN GO-BP enrichment (2026-07-09, `validation/nn_enrichment/knn_go_enrichment.py`)
`crohn_colon_macrophage_inflammatory`, v7, 3,665 proteins with informative GO-BP terms (terms annotating 3–500 genes),
k=15. Fraction of k-NN sharing ≥1 GO-BP term / pairwise AUROC(same-term | cosine):
**EMB 0.207 / 0.513 · ESM 0.283 / 0.529 · random 0.114 / 0.500.**

## ZNF638 trace — Crohn-pooled logreg (2026-07-09)
Flagged as a high-ranked non-target. Rank **155/7,983**, P=0.823, not a drug target, present in all 13 crohn contexts.
- Out-degree 0 (rank 7965/7983); Spearman(P, out-degree)=−0.02. cos(ZNF638, target-centroid)=+0.188 (median +0.190);
  10 nearest EMB neighbors all non-targets.
- P by classifier: **logreg-balanced 0.823, logreg-unbalanced 0.035, MLP 0.015** (20 positives / 7,983).
- Nearest Crohn targets — ESM: NR3C1 0.974, PPARG 0.960, JAK1 0.948, VDR 0.946; EMB: JAK1 0.44, then CRBN/RBX1/CUL4A,
  NR3C1. Best rank toward any single target: #261 (EMB, CRBN) / #629 (ESM, CD86).

## v8 — learned protein-ID embedding, no ESM (2026-07-09, `results/link_v8_idrepr/`)
`train_link_context.py --protein-repr id`: frozen-ESM projection replaced by a **random-init learnable
`nn.Embedding(15194, 256)`** (protein identity learned from the graph, not sequence); context factors + decoder
unchanged; transductive (no vector for unseen proteins). Matched v7 settings (hard neg, cisTarget, 60 ep).
**FULL 0.755 / BLIND 0.615 (identity-only) / context-lift +0.140.** Pooled all-OT target (EMB alone): logreg
0/0/0.001, mlp 0/1/0.001. Combined ESM+EMB in table above.

## heart_valve SCENIC — new disease axis (2026-07-09)
CELLxGENE pull `01_expression/pull_new_diseases.py` (atherosclerosis, COVID-19, heart valve disorder; macrophage +
fibroblast; disease + matched-normal). heart_valve prepped via `scenic/scripts/prep_new_disease.py` (no scVI; gene
universe = top-6,000-variance HVG ∪ present TFs ∪ state-marker genes ∪ OT/drug-target genes = **7,944 genes, 1,970
source TFs**; 12 contexts ≥50 cells across {hvd,healthy} × heart × {macrophage,fibroblast} × states). GRNBoost2
+ cisTarget **DONE** (12/12; the 48-worker rerun ran ~2–3 min/context vs 54 min at 8 workers) → folded into **v9**.

## Architecture note — context conditioning (2026-07-09)
Current: context-factor embeddings concatenated with the projected protein vector → MLP mixes them. Alternatives
considered (not yet implemented): **FiLM** (context → per-feature scale/shift on the protein), **additive offset**
(`z = f(protein) + g(context)`, TransE-style), **multiplicative/bilinear** protein×context interaction, or
**conditioning the link-decoder** instead of the encoder.

## OpenTargets drug labels for new diseases (2026-07-10)
`03_opentargets_rebuild/pull_ot_new_diseases.py` (reuses cached knownDrugsAggregated + `emit_known_drugs`, phase≥3):
`known_drugs_MONDO_0100096` (COVID-19, 372 rows), `known_drugs_EFO_0003914` (atherosclerosis, 220),
`known_drugs_EFO_0009940` (heart valve disease, 18 rows / 13 targets — sparse, surgically managed). **All-OT union
now 314 targets across 8 files.**

## T-cell build (2026-07-10, `inputs/tcell/`)
Pulled T cells across crohn/uc/ild/covid/athero + matched normal (`01_expression/pull_tcells.py`, ~99k cells).
Prepped `scenic/scripts/prep_tcell.py` — **5 marker-scored subtypes** (`state_markers.tsv` cell_type=tcell:
cd4_helper, cd8_cytotoxic, treg, naive, proliferating), gene universe HVG∪TFs∪targets∪markers (7,932 genes, 1,970 TFs).
**80 contexts** (16 disease×tissue × 5 subtypes; introduces covid/athero disease axes + blood/vasculature/nose tissues).
SCENIC via 48-worker GRNBoost2 + **overlapping cisTarget loop** (cisTarget runs on finished contexts while GRNBoost2
continues — collapses two sequential stages into ~one). Trainer gained `--exclude <substr>` to drop contexts by tag.

## v9 — + heart_valve (2026-07-10, `results/link_v9/`, 52 contexts)
Existing 40 + heart_valve 12 (`--exclude tcell`), ESM, cisTarget, hard neg, 60 ep. **FULL 0.765 / BLIND 0.648 /
context-lift +0.117.** heart_valve added 12 contexts (6 `hvd_heart` + 6 `healthy_heart`) and **1,039 heart-only
proteins**. Per-shared-TF regulon overlap hvd vs matched healthy_heart ≈ **Jaccard 0.017** (labels are context-specific;
partly SCENIC stochasticity). Example — TNF's regulators: Crohn-inflammatory-macrophage {NFKB1, STAT1, ETS2, KLF6,
CUX1, TCF7L2} vs healthy {KLF10}, zero overlap.

## v10 — + T cells (2026-07-10, `results/link_v10/`, 132 contexts)
v9's 52 + 80 T-cell (1.83M train pos). **FULL 0.785 / BLIND 0.678 / context-lift +0.107.** Adding T cells raised
FULL AUC (+0.020 over v9) **but BLIND rose too (+0.030) and lift fell (−0.010)** — the AUC gain is baseline, not
added context-specificity.

## Pooled all-OT classifier — v9 vs v10 (2026-07-10, `v10_classifier_boxplot.py`)
Global mean-pool per protein → LogReg + MLP on EMB and ESM, predicting all-OT target-ness (314). 2-panel boxplot
(Phase 3/4/Control × EMB/ESM) + per-protein table (`v{9,10}_classifier_table.tsv`: prob_model/esm × logreg/mlp,
8 per-disease 0/1 cols). Also disease-colored variant (`disease_classifier_boxplot.py`) + per-disease pooled
(`per_disease_target_pooled.py`). H@10 / H@100 / MRR:

| build | logreg EMB | logreg ESM | mlp EMB | mlp ESM |
|---|---|---|---|---|
| v9 (52 ctx, 10,586 prot) | 1 / 10 / 0.0031 | 9 / 55 / 0.0176 | 9 / 28 / 0.0142 | 10 / 73 / 0.0204 |
| v10 (132 ctx, 10,902 prot) | 0 / 7 / 0.0020 | 5 / 49 / 0.0111 | 9 / 44 / 0.0132 | 9 / 63 / 0.0190 |

Per-disease (`per_disease_target_pooled.py`): ESM > EMB in every disease (crohn/uc/ild/alz/hvd), both classifiers.
(`pool_all` bug fixed: it had re-read the full `emb` array from the npz on every loop iteration → load once.)

## Does the added data help hits@100? — controlled test (2026-07-10)
Test whether v10's gains are real training benefit vs just averaging over more contexts: **pool BOTH v9 and v10 over
the SAME 52 contexts** (equal pool/gene set — ESM comes out identical, confirming clean apples-to-apples). mlp EMB:

| pooling | mlp EMB H@100 | mlp EMB MRR |
|---|---|---|
| v9 @ 52 (v9 embedding) | 28 | 0.0142 |
| **v10 @ 52** (v10 embedding, same 52 ctx) | **39** | 0.0115 |
| v10 @ 132 (v10 embedding, all) | 44 | 0.0132 |

**Decomposition of the 28→44 H@100 gain: training effect (v9@52→v10@52) = +11 (~⅔); extra-context pooling
(v10@52→v10@132) = +5 (~⅓).** → Training on the 80 T-cell contexts **genuinely improved the shared-context
embeddings' mid-rank target capture** — most of the gain is a real learning benefit, not just variance reduction
from averaging. **Scope of the benefit:** confined to **H@100 (mid-rank)** — MRR and H@10 flat (top-rank unchanged),
still well below ESM (39 vs 73), and context-lift did **not** rise. Takeaway: more data gives real but incremental
mid-rank returns; it does not move the top-rank / ESM-dominance ceiling (that needs an objective change).

## endothelial build (2026-07-10, `inputs/endothelial/`, in progress)
Pulled endothelial across crohn/uc/ild/covid + atherosclerosis vasculature/brain/heart + matched normal
(`01_expression/pull_endothelial.py`, ~83k cells). Prepped `prep_endothelial.py` — **5 marker-scored EC states**
(arterial/venous/capillary/lymphatic/angiogenic, `state_markers.tsv` cell_type=endothelial). **70 contexts.**
SCENIC (48-worker GRNBoost2 + overlapping cisTarget loop) **DONE (70/70)** → folded into **v11**.

## Per-disease classifier — methodology fix (2026-07-10, `v10_classifier_boxplot.py`)
The pooled classifier had been **mean-pooling each protein globally across ALL contexts (= all diseases)**, which
washes out disease context and hands ESM (disease-invariant) a structural advantage. **Fixed:** now pools **per
disease arm** (crohn/uc/ild/alz/hvd/covid/athero) and takes each protein's **max predicted P across diseases** →
one combined ranking. LogReg + MLP, EMB vs ESM, H@10/H@100/MRR corner box + phase n= on the boxplot.
New numbers (H@10 / H@100 / MRR):

| build | logreg EMB | logreg ESM | mlp EMB | mlp ESM |
|---|---|---|---|---|
| v9  | 0 / 4 / 0.0016 | 6 / 45 / 0.0149 | 5 / 31 / 0.0089 | 9 / 66 / 0.0209 |
| v10 | 2 / 12 / 0.0069 | 6 / 43 / 0.0150 | 5 / 35 / 0.0110 | 10 / 67 / 0.0200 |
| v11 (partial endo) | 2 / 18 / 0.0041 | 6 / 51 / 0.0146 | 6 / 45 / 0.0109 | 9 / 75 / 0.0191 |
| **v11 (full endo, 203 ctx)** | — | — | **8 / 47 / 0.0159** | 10 / 72 / 0.0191 |

**LogReg dropped (2026-07-10):** the classifier step now runs **MLP only** — MLP-EMB is the working readout while
LogReg-EMB stays near-zero. `v10_classifier_boxplot.py` + `disease_classifier_boxplot.py` are now single-MLP-panel
(table keeps `prob_model_mlp` / `prob_esm_mlp`). **MLP-EMB is the one line that improves with data:** H@100 31→35→47
and MRR 0.0089→0.0110→**0.0159** (v9→v10→v11-full), reaching **0.83× of MLP-ESM MRR** (0.0159 vs 0.0191) — the
closest EMB has come. Still below ESM and still mostly mid-rank, but the "more data helps the embedding" trend holds
and is now nudging the top rank too.

**Verdict unchanged by the fix:** proper per-disease pooling does NOT make EMB beat ESM — MLP-EMB (~0.009–0.011 MRR)
stays ~half of MLP-ESM (~0.019–0.021) across all builds. (Cross-build ESM drift is just the pool/positives changing:
10,586→10,902→10,841 proteins, 247→275 positives — within-build EMB-vs-ESM is the fair read.)

## v11 — + endothelial (2026-07-10, `results/link_v11/`)
Partial build (171 contexts = v10's 132 + 39 finished endothelial at launch): **FULL 0.796 / BLIND 0.690 /
context-lift +0.106.** Full-endothelial rerun (202 contexts) auto-fires when endothelial cisTarget completes.
Trend across the expansion: **FULL AUC keeps rising (0.765→0.785→0.796) but that's the BLIND baseline rising too;
context-lift stays flat/slightly down (+0.117→+0.107→+0.106)** — more data ↑ absolute prediction, not context-specificity.
Loss curve: `images/v11_loss_curve.png` (naming convention `<tag>_loss_curve.png`).

## MLP2 rank-loss experiment — FAILED, removed (2026-07-10)
Tested whether changing the target-recovery readout loss helps the embedding. Replaced pointwise balanced-BCE with
**rank-weighted pairwise (LambdaMRR)** on frozen pooled features: (a) hard negatives, (b) random negatives, (c)
**continuous OT-association graded ranking**, disease-specific + max-agg (`mlp2_graded.py`). **All failed:** EMB got
*worse* than the pointwise MLP with hard negs (MRR ~halved, e.g. v10 0.0132→0.0111), no better with random negs, and
the disease-specific continuous version stayed ≪ ESM (v11 EMB 0.0045 vs ESM 0.0059 on the drug-target gate). Reason:
a ranking loss optimizes *extraction*, not *signal* — on a signal-poor feature, hard negatives chase noise and hurt.
**Conclusion: readout-loss changes are a dead end; the lever is fine-tuning the encoder or a cleaner objective, not
the readout.** Code + outputs removed (`mlp2.py`, `mlp2_graded.py`, `v*_mlp2*` images/metrics).

## B-cell build (2026-07-10, `inputs/bcell/`, DONE)
Pulled B-lineage across crohn/uc/ild/covid/athero + matched normal (`01_expression/pull_bcells.py`, ~83k cells).
Prepped `prep_bcell.py` — **5 marker-scored B states** (naive/memory/germinal_center/plasma/plasmablast;
`state_markers.tsv` cell_type=bcell). **73 contexts, GRNBoost2 73/73 + cisTarget 73/73.** Fed v12.
Noise check (motif-survival = cisTarget/GRNBoost2 edge fraction, the direct co-expression-noise readout): bcell **1.2%**,
in line with tcell 1.4% / endo 1.2% / macrophage 1.0% → B-cell networks are NOT noisier. Only thinner axis: fewest
cells/context (median 674 vs 1100–2268), a mild concern, not the driver of the v12 recovery dip. Targets are well
covered in B-cell contexts (252/314 all-OT; crohn 28/31, athero 78/113, covid 172/226) — coverage isn't the issue.

## v12 — + B cells (2026-07-10, `results/link_v12/`, 252 contexts)
FULL test **AUC 0.805 / AP 0.798** (v11 203 + 49 B-cell contexts, esm repr, hard negs, 60 ep, best_ep 59 — val-AUC
plateaued ~0.805, essentially converged; not under-trained). **Link-pred AUC is monotonic in data: v9 0.765 → v10
0.785 → v11 0.799 → v12 0.805** = the objective improving as contexts grow. Downstream drug-target probe (per-disease
pooled MLP): EMB H@10 6 / H@100 31 / MRR 0.0095 vs ESM 10 / 71 / 0.0186. With the *current* consistent methodology,
EMB H@100 across builds = v9 31, v10 35, **v11 47**, v12 31 — v11 peaks, v12 falls back. This is real (reproduces
under one method), NOT the earlier confounds I floated: pool size ~constant (11,070→11,117 unique proteins), and v12
is not under-trained. Pooling-dilution hypothesis (B-cell contexts diluting the mean-pool) **falsified**: holding v12
fixed and excluding B-cell from each disease's pool did NOT recover (H@100 31→35, but H@10 6→4, MRR ↓ — noise-level).
So the recovery move is in the **retrained weights**, and/or CV noise. It does **not** indict v12 — see Framing.

## Framing — link-pred objective vs drug-target probe (2026-07-10)
The **training objective is context-specific regulatory-neighbor link prediction**; drug-target prioritization is one
*application* of the learned representation, not the definitive quality measure. v12 improved on its objective (AUC
0.765→0.805 with data) even though the drug-target probe got worse — both are true at once, and "v12 got worse" was a
framing error (worse at a probe, not the objective). The probe is weak/indirect: ~314 positives, pooled, disease-
agnostic, ESM-dominated. **Caveats that keep AUC honest:** (1) FULL AUC is a held-out **edge** split *within trained
contexts* — in-distribution; confirms the objective is learned, not that it generalizes to unseen contexts / external
tasks. (2) The objective-aligned quality metric is the **context-lift = FULL − BLIND AUC** — raw AUC rises partly
because the BLIND baseline rises with more protein-pair coverage (see v11 note: lift flat +0.117→+0.106 while AUC
climbed); the lift isolates what the context embeddings *specifically* add (context-specific structure vs. identity +
topology). v12(293) ran BLIND-off; **v13 (BLIND-on) gave lift +0.234** — but that was inflated by construction (see
Downstream diagnostics). **To judge the representation on its own terms** (not via drug targets): context-lift
(FULL−BLIND). NOTE: held-out-*context* link prediction is **N/A by construction** — contexts are named ID embeddings,
an unseen context has no vector (see Accepted limitations); do not use it as a generalization test.

## Bipolar-I build (2026-07-10, `inputs/bipolar/`, in progress)
Pulled Bipolar-I brain + matched normal (`01_expression/pull_bipolar.py`, 48k cells; oligodendrocyte / astrocyte /
glutamatergic_neuron, 8k each × 2 arms). Prepped `prep_bipolar.py` — **3 cell types share one inputs dir**, state
marker-scored PER cell type (new `state_markers.tsv` panels: oligo opc/newly_formed/myelinating, astro
protoplasmic/fibrous/reactive, glut upper_layer/deep_layer/pan_excitatory). **18 contexts** (2 arms × 3 ct × 3 states).
SCENIC in the **mamba `scenic` env** (`.venv_scenic` is broken — pandas 2.1.4 breaks `dask.dataframe`); GRNBoost2 (48
workers) + `run_bipolar_cistarget_loop.sh` (48 workers). NOTE: contexts are `<arm>_brain_<ct>_<state>`, so any
`*_bipolar_*` glob misses them (arm=bipolar is at the START; healthy_* has no "bipolar") — count from
`context_cells.tsv` (the loop's completion check was fixed for this). **DONE (18/18 GRNBoost2+cisTarget, fed v13).**
OT labels pulled (`known_drugs_MONDO_0004985.tsv`, 128 targets phase≥3, scope = bipolar disorder + BP-I + BP-II) → bipolar
is now a validation arm; wired into the validation dicts (`DIS_EFO`, `DISNAME`).

## kNN seed-set size/heterogeneity confound (2026-07-13) — IMPORTANT for interpreting kNN enrichment
In the disease-enriched-biology validation, "seeds" = a disease's OT drug targets. **Diseases with a large,
functionally heterogeneous seed set collapse to a generic ribosomal/translation neighborhood** (covid 182 seeds,
athero 93, bipolar 87 → top pathways = mitochondrial translation / SRP / rRNA), because averaging the k-NN of many
unrelated targets has no coherent common neighborhood except the high-connectivity hub. **Small, functionally
coherent seed sets stay disease-relevant** (alz 21 → Neuronal System/synapses; crohn 29 → immunoregulatory lymphoid).
**Consequence:** kNN pathway/target-enrichment differences across diseases are **partly a seed-count/coherence
artifact, not purely biology** — always note n_seeds and coherence before reading a "collapse" as a missing-signal
result. (Earlier I mis-attributed covid/athero/bipolar collapse to "missing driver cell type"; for bipolar that was
wrong — oligo/astro/glutamatergic ARE the disease-relevant compartments — the large-heterogeneous-seed-set effect is
the parsimonious explanation. Missing driver cells still applies to covid/athero, which additionally lack macrophage.)

## Downstream diagnostics — the approach isn't converting context into a useful representation (2026-07-13)
A battery of downstream tests on v12 (293 ctx). All point the same way: the frozen-readout link-pred approach
extracts little usable disease/target signal beyond ESM sequence-family membership.

- **v13 — edge-weighting (`--edge-weight inv_ctx`, upweight context-specific positives 1/#ctx):** FULL AUC dropped
  0.807→0.702, **context-lift jumped +0.106→+0.234 (highest ever) BUT partly by construction** — BLIND fell *below*
  chance (0.468) because we upweighted exactly the edges a blind model can't predict. Downstream target recovery got
  **worse** (EMB H@100 48→42, MRR 0.0126→0.0084). **Lift is not a sufficient proxy for embedding quality.**
- **Attention-MIL downstream (`validation/mil_attention_target.py`):** replace mean-pool-over-contexts with a learned
  gated-attention pooler over a protein's per-context embeddings. Did **not** beat mean-pool (MIL MRR 0.0094 vs mean
  0.0110, both ≪ ESM 0.0169). Learning "which context matters" adds nothing — averaging wasn't the bottleneck.
- **Per-disease MRR (EMB vs ESM, `per_disease_mrr`):** not uniform — **EMB beats ESM for alz (0.054 vs 0.015, 3.7×)
  and athero (1.4×)**; ESM wins the other 6. Driver-cell presence does NOT explain it (crohn/uc/ild/hvd have their
  causal compartments and ESM wins). alz stays an unexplained standout. n_contexts and %unique-edges don't predict it
  either (Spearman ~0).
- **Recovery by target class (GO Molecular Function, `protein_function.tsv`) and by OT `mechanism_of_action`:** ESM's
  advantage is a **sequence-family effect** — it recovers targets best exactly in tight druggable families
  (signaling_receptor/GPCR, ion_channel, GABA-A, phosphodiesterase, kinase). EMB's only wins are non-family classes
  (GTPase, tubulin inhibitors). Target recovery ≈ "is this protein in a druggable family," which ESM answers by sequence.
- **MoA-stratified kNN enrichment (`moa_disease_knn_enrichment.py`, disease×MoA seeds):** seeding by MoA gives *far*
  more coherent, on-target Reactome neighborhoods than disease-pooling (dopamine-antagonist→GPCR, glutamate→Neuronal
  System, tubulin→replication, VEGFR/FGFR→ECM) — random controls cleanly unrelated. **But disease ≈ healthy in nearly
  every row (disease-agnostic), and most coherence is the embedding echoing the seeds' own protein family.**
- **Geometry tests (no ML), OT targets, disease vs matched-healthy pooled Δz:**
  - *Within- vs between-protein distance* (`context_distance`): a protein's disease→healthy shift is **~0.37–0.56×**
    the distance between two different proteins — real & sizeable but secondary to identity; consistent across diseases.
  - *Displacement coherence* (`displacement_coherence`): targets' Δz vectors are **~orthogonal** (within-cos 0.03–0.11),
    barely above random-pair (Δ +0.01–0.08; only ild modest). **No shared disease direction.**
  - *Disease-centroid recovery* (`centroid_recovery`, score = cos(Δz, mean Δz_targets)): leave-one-out recovery is
    **near-zero** (H@10 0–2, H@100 0–8, MRR ≤0.03 per disease; crohn 0/29). Naive (in-centroid) numbers are circular
    inflation. On par with the MLP → **the readout was never the bottleneck; the disease-target signal isn't in the
    displacement geometry.**

**Synthesis:** the disease-context shift is real in *magnitude* but *incoherent in direction* and *non-recoverable*;
target recovery is dominated by ESM sequence-family membership; every frozen-readout lever (more data, readout-loss,
edge-weighting, attention pooling, displacement geometry) moves a proxy without producing a better embedding. The one
structurally untried lever is **fine-tuning the encoder** (backprop into the representation) or a context-*requiring*
supervised target task.

## Architecture B (pathway aux) results — FIRST lever that helps + macrophage dev-loop sweeps (2026-07-13)
**v14 (full 293, arch-B, pathway λ=0.1, per-protein-per-context): the pathway aux IMPROVED EMB target recovery — the
first lever to do so** (more-data, edge-weighting, MIL all failed/hurt). vs matched v12 (293, arch-A, no aux): EMB
H@100 **48→61**, MRR **0.0126→0.0142**; ESM unchanged (78 / 0.0174) → a real embedding gain (EMB now 0.78× ESM,
closest yet). FULL AUC 0.808 ≈ v12 (aux doesn't cost link-pred). Caveat: likely mechanism = `z` now encodes
pathway/functional identity better (targets are pathway-enriched), may overlap ESM's family signal; single-seed CV.

**Macrophage-only dev loop** (`--include macrophage`): **24 contexts, 308K edges, ~130–150 s/run** — fast tuning set.
Drops alz/bipolar/covid/athero, keeps crohn/uc/ild/hvd (4 arms, pool 8071). Numbers NOT comparable to full 293
(smaller pool/fewer arms) but internally consistent; single-seed → treat orderings within ~noise.

- **Pathway-λ sweep (link-pred + pathway aux):** dose-response — EMB H@100 climbs with λ (0.10→**39**, 0.25→**52**,
  0.50→**55**); MRR peaks at λ=0.25 (0.0178). ESM=69/0.0219. FULL AUC flat ~0.756 across λ. → the pathway loss injects
  **real target-relevant structure**; sweet spot **λ≈0.25–0.5**. Confirms v14 wasn't a fluke.
- **Expression aux (pathway held 0.25, +expr λ) — HURTS at every λ:** pathway-only H@100 52 → +expr 0.10=41,
  0.25=37, 0.50=38 (MRR 0.0178→0.008–0.014). No helpful dose. Predicting per-protein abundance pulls `z` toward
  expression magnitude and **dilutes the pathway signal**. → **drop expression.**
- **Pathway as the loss, ablating link-pred and expression (`--link-weight 0`):**
  | loss | EMB H@100 | EMB MRR |
  |---|---|---|
  | link-pred + pathway 0.25 (l025) | 52 | 0.0178 |
  | pathway 1.0 + expr 0.10, no link (pmain_e10) | 53 | 0.0165 |
  | **pathway 1.0 ONLY (pathonly)** | **54** | **0.0180** |
  | ESM | 69 | 0.0219 |
  **Pathway-only ties/marginally beats every variant.** → link-prediction adds **nothing** to target recovery
  (pathonly 54 ≥ link+pathway 52); expression **slightly hurts** (0.0180 > 0.0165). **Minimal winning recipe =
  per-protein-per-context pathway activity as the SOLE loss** — the entire SCENIC link-prediction machinery and the
  expression term can both be dropped.

**Net:** the per-protein-per-context **pathway auxiliary is the first genuine positive lever** for target recovery,
and it is in fact the WHOLE signal — it works best as a **standalone objective** (no link-pred, no expression). But
it's still **below ESM** (best macrophage EMB ≈0.78–0.8× ESM), so pathway/functional-class grounding, not a rescue.
Expression aux and (for target recovery) the link-pred objective are both dead weight. **Confirm on full 293 next.**

New infra (`train_link_context.py`): flags `--aux-pathway`/`--aux-expression`/`--link-weight`/`--include`;
generators `gen_pathway_activity.py` (293×1490), `gen_expression_activity.py` (293×7757); `validation/mac_sweep_eval.py`
(per-disease pooled MLP for a list of runs + ONE shared ESM baseline — ESM is deterministic given the same pool).

## PIVOT — drug-target-recovery is an ESM-shadow; disease-DRIVER reversal via a connectivity autoencoder works (2026-07-13)
**Drug-target recovery was confirmed a mirage.** An **untrained** (random-init) encoder scores EMB H@100 **47** / MRR
**0.0162** on macrophage — ~90% of any trained model (52–54 / 0.018). So the "signal" is a **frozen-ESM shadow** (a
random projection of ESM, per-disease pooled, already recovers targets because druggability = sequence-family +
pathway-**annotation** richness: OT targets sit in ~2× more Reactome pathways, 11 vs 5.5). Training (link-pred,
pathway) adds almost nothing over that floor — which is why models at link-AUC 0.50 and 0.76 give the same MRR.
**Not a code bug** (cisTarget gen is standard pyscenic; matrices healthy; embeddings differ across runs cos 0.17;
pooling correct). **Root cause is structural:** the regulatory network doesn't encode drug-target-ness — 227/242 OT
targets are passive downstream nodes (only 15 are TFs), and targets are indistinguishable by graph degree
(in 34 vs 44, out 43 vs 43). Drug-target-ness is a sequence/genetics/accessibility property, orthogonal to
"who-regulates-whom." **All EMB target-recovery numbers above should be read as deltas over the ~0.016 untrained floor,
not over 0.** Reframed the question from "is this a drug target" (ESM/genetics-bound, data can't add) to "what DRIVES
the disease state" (disease-vs-healthy is exactly what the data encodes).

**Connectivity autoencoder → disease-driver reversal (`validation/macrophage_ae_reversal.py`) — FIRST coherent result.**
NO ESM (pure network role, sidesteps the ESM shadow). Each (protein, context) → its regulatory-neighbor indicator
vector → AE (9543→256→64→256→9543, BCE) → latent z = compressed *context-specific regulatory role*. Macrophage only
(24 ctx, 108,898 protein×context rows). Crohn: pool z over crohn-macrophage (colon/ileum) vs tissue-matched
healthy-macrophage; per-protein role-shift `||z_healthy − z_crohn||`, influence-weighted; direction = reduce
(disease-gained connectivity) / restore (disease-lost). **Top-10 is biologically face-valid Crohn-macrophage
inflammation:** reduce STAT1 (IFN/JAK-STAT — JAK inhibitors are approved IBD drugs), CEBPB (inflammatory-macrophage
master TF), ETV7 (IFN-induced), JUNB/HES1 (AP-1/Notch); restore SPIC (homeostatic resident-macrophage identity,
deg 1→536), SPIB, JUND. Out: `results/macrophage_crohn_reversal.tsv`.
Caveats: everything on the list is a **TF** (network is TF→target, so role-shift = regulators = drivers, not
directly-druggable targets — bridge driver→pathway→drug, e.g. STAT1→JAK); influence-weighted so partly degree-driven
(USF2/BCLAF1 look like generic hubs); and it's a **correlational** disease-vs-healthy difference (hypothesis), though
recovering known IBD drivers (STAT1, CEBPB) is real face-validity the target readouts never had. **This is the
data-appropriate question**; the protein link-prediction embedding was the wrong instrument.

## Roadmap
- …heart_valve→v9 ✅ · T-cells→v10 ✅ · endothelial→v11 ✅ · **B-cells→v12 ✅** · **Bipolar-I→v12-rerun ✅ (293 ctx, FULL 0.807; bipolar OT labels pulled + wired in)** · **MLP2 (FAILED) · v13 edge-weighting (FAILED — lift↑ by construction, recovery↓) · attention-MIL (FAILED — ≈mean-pool) · 3 no-ML geometry tests (disease shift incoherent + non-recoverable)**.
- **Established (2026-07-13):** frozen-readout approach doesn't convert context into a useful representation; target recovery is ESM sequence-family-bound; no coherent/recoverable disease axis. See "Downstream diagnostics".
- **Open (objective pivot — now the ONLY sensible next step):** **fine-tune the encoder** with a supervised target/disease objective (backprop into the representation — read-out on frozen features never beats ESM, confirmed across every lever); and/or a context-*requiring* target task; and/or a less-noisy context label than co-expression (context-PPI/PINNACLE, per-context pathway activity, DE signatures).
- **N/A by construction:** held-out-*context* link prediction — contexts are named ID embeddings, so an unseen context has no vector (see Accepted limitations). Context-lift (FULL−BLIND) is the objective-aligned metric instead.
- **Later:** full 275-context v12 rerun (+23 all-cell-types) + more epochs; covid/athero macrophage+fibroblast SCENIC (data on disk, driver cells missing → both collapse to ribosomal); per-study consensus (colon batch); epithelial / smooth-muscle builds.
