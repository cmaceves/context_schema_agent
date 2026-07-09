# SEQ_CONTEXT_EMBED.md — context-specific protein embedding via regulatory-neighbor link prediction

**Status: PLAN + SCENIC generation in progress (2026-07-08).** A new modeling direction, distinct from the
graph joint-embed lineage in `HISTORY.md`/`CONTEXT_EMBED.md`. A protein's context-specific representation is
learned from its **frozen sequence embedding** plus **learned context-label embeddings**, and supervised by
**predicting its regulatory neighbors** (SCENIC-inferred TF→target edges) in that context. Working dir:
`mlp_mods/de_ppi/seq_context/`.

## Goal / deliverable
One reusable **context-specific protein embedding**: the same protein (fixed sequence) adopts a different
vector in each (cell type, disease, tissue, cell state) context, while preserving sequence identity. The
embedding is the product; the prediction task is only the training signal.

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
- **Tool:** pySCENIC. GRNBoost2 (co-expression TF→target) → [optional] cisTarget motif pruning → regulons.
  Neither pyscenic nor the cisTarget motif databases are installed yet.
- **Per context:** subset cells → run SCENIC → write directed edge list (`seq_context/scenic/networks/
  <context>/edges.tsv`, columns `tf,target,importance[,sign]`) restricted to the 6,820-gene node universe.
- **Viability caveat:** co-expression inference needs enough cells; `proliferating` (4,502 total, before disease
  split) may be too sparse for a stable network in some contexts.

## Open decisions (see conversation)
1. **SCENIC scope:** GRNBoost2 co-expression only (no motif DB, fast, pure co-expression) vs full SCENIC
   (+cisTarget motif databases, ~15–25 GB download, motif-grounded / less circular, the canonical network).
2. **Context granularity:** run per full (disease×tissue×state) — requires recovering tissue, splits cells
   thinner — vs per (disease×state) or (source×state), directly available and larger/more-stable per network.
3. **Install target:** fresh venv vs `.venv_scvi` for pyscenic + arboreto/ctxcore/dask.

## Planned layout
```
mlp_mods/de_ppi/seq_context/
  scenic/
    scripts/     # subset cells + run pySCENIC per context
    inputs/      # per-context expression matrices (counts) fed to SCENIC
    networks/    # per-context edge lists = link-prediction labels
  scripts/       # train_link_context.py (ESM+context encoder, BCE link prediction) — later
  results/       # trained encoders + context-specific embeddings — later
```

## Roadmap
- **Now:** generate SCENIC context networks (this doc).
- **Next:** `train_link_context.py` — ESM ⊕ context-ID encoder, BCE edge reconstruction, context-lift eval.
- **Later:** reintroduce cell types (stage more than macrophage), motif-grounded regulons, GO auxiliary head.
