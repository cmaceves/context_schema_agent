# Build history — de_ppi learned PPI embeddings

Chronological record of the network **builds** under `results/` and **why each one changed**. This is the
narrative companion to `METHODS.md` (which documents what runs today) and `CONTEXT_EMBED.md` (training
objectives). Each build is a new `results/<name>/` dir; **originals are never overwritten**.

Three things evolve across the lineage, roughly independently:
1. **Node membership** — which proteins are nodes in each network (PINNACLE backbone → expression → per-context → placeholder-augmented).
2. **Node feature / edges** — how expression is corrected (raw → ComBat → scVI) and edge weights (topology → coexpression).
3. **Training objective** — how the encoder is trained (link-prediction → contrastive → masked feature modeling).

Lineage (main branch):
```
(PINNACLE backbone) → _expressed → _combat_loc → _combat_loc_coexpr ─┬─→ _coexpr_exprfilt
                                                                     └─→ _coexpr_healthyph ─→ _context_contrastive
                                                                                          └─→ _masked
side branches:  _pinnacle_combat_ct (NOD2/TNF readmission)   _expressed_scvi (scVI, in progress)
```

---

## 1. PINNACLE-backbone embedding  *(origin; `crohn_alzheimer_ild_embedding`, then +UC)*
- **What:** nodes = the context-specific **PINNACLE cell-type PPI backbone**; OmniPath directed edges; node feature = expression.
- **Why:** start from a curated context-specific interactome rather than a generic PPI.
- **Problem that drove the change:** the fixed backbone both **included non-expressed backbone proteins** and **excluded expressed non-backbone proteins**, so node sets didn't match what a given context actually expresses.

## 2. `_expressed`  — expression-defined node membership
- **Change:** node set = **proteins detected ≥ floor** per context (CP10k ≥ 0.5 ⟺ log1p ≥ 0.405), replacing the PINNACLE backbone. Feature = raw mean CP10k.
- **Why:** make membership reflect what is expressed in each (arm, tissue, cell type, state), not a static backbone.

## 3. `_combat_loc`  — ComBat location-only expression correction  *(long-time canonical base)*
- **Change:** correct the node feature with **ComBat, location-only**, fit per (tissue, cell type) with `batch=study8`, preserving `arm` + `state` (`apply_combat_expression.py`).
- **Why:** between-study batch effects were inflating the expression feature (and downstream donor/batch control floors). ComBat removes study offsets while keeping disease/state contrasts.

## 4. `_combat_loc_coexpr`  — coexpression edge weights  *(the "coexpr base")*
- **Change:** replace neutral edge weights with **per-donor pseudobulk coexpression (|corr|) on EVERY edge**.
- **Why:** inject a second-moment (co-variation) signal orthogonal to expression. Improved cell-state separation (≈1.36 → 1.79) and revealed real orthogonal structure (R² of shift-vs-expression dropped 0.55 → 0.12).
- **Lesson baked in:** a v1 that mixed coexpression weights on some edges and neutral weights on others was a **confound** (batch floor jumped) — coexpression must be applied to *all* edges in one consistent regime. *(A transient `_minedge` edge-weight test left average shifts unchanged and was dropped.)*

## 5. `_coexpr_exprfilt`  — per-context node membership
- **Change:** keep only genes **expressed in each individual network** (membership varies by tissue/celltype/state/arm), instead of a fixed per-cell-type shared set.
- **Why:** a gene expressed in one context (e.g. **BEX3** in ileum macrophage) was carried at ~0 expression into every other macrophage network (e.g. colon), where its embedding shift was **topology-driven noise**, not biology.

## 6. `_coexpr_healthyph`  — healthy-placeholder membership  *(current base for embedding objectives)*
- **Change (a):** disease networks additionally keep every gene expressed in their **paired healthy** network as a **zero-feature placeholder**, so a gene silenced in disease still appears (expr → 0) and the disease-vs-healthy shift is defined for it.
- **Change (b), symmetric:** healthy networks also gain, as **isolated (edgeless) zero-feature placeholders**, genes expressed only in the paired disease net (same tissue/celltype/state). Isolated so they can't alter any connected node's message passing.
- **Why:** the pure per-context filter dropped **down-regulated** genes from disease nets and **disease-only** genes (e.g. **ITGA4**, off in healthy) from the comparison entirely, so they had no shift/perturbation readout. Placeholders restore coverage at (verified) negligible cost to existing embeddings.

## 7. `_context_contrastive`  — contrastive training objective
- **Change:** on the `_coexpr_healthyph` base, add a **center loss on disease arm** (pull a protein's same-arm embeddings together, push arm centroids apart) so disease becomes an explicit axis. *(Sibling `healthy_centered` variant: an aux head reconstructing the disease−healthy expression delta.)*
- **Why:** the plain link-prediction embedding is dominated by degree + protein identity; the contrastive term was meant to surface disease direction. Used as the encoder for the **in-silico perturbation** experiments.
- **Key negative finding (drove step 10):** in-silico normalize/KO perturbation projected onto the disease→healthy axis carries **no OpenTargets-target signal** — the response decomposes into DE magnitude × out-degree (residual test: Spearman(target, residual) ≈ −0.2, negative), reproduced in Crohn *and* the larger ILD data. The signal that exists is generic TF-hub behavior, not target relevance.

## Side branch: `_pinnacle_combat_ct`  — readmit censored targets
- **What:** PINNACLE cell-type backbone ∪ expressed nodes, ComBat feature, no states.
- **Why:** expression-floor membership **censors low-abundance canonical targets** — **NOD2** and **TNF** fall below scRNA-seq detection in colon macrophage and are absent from the whole `_coexpr*` universe. This build readmits them (NOD2 via topology; TNF via an arm-aware floor) so target-recovery tests can even include them.

## Side branch: `_expressed_scvi`  — scVI correction + states  *(macrophage built)*
- **What:** replace ComBat with **scVI** (negative-binomial VAE, explicit library size, `batch_key=study8`) for both **expression correction** and **cell-state definition** (Leiden on the integrated scVI latent), one model per cell type (`run_scvi.py`).
- **Why:** scVI models counts and library size more faithfully than ComBat and defines states in a single integrated space (comparable across studies).
- **Staging→build adapter (`scripts/embed/adapt_scvi_build.py`):** turns one cell type's staging AnnData into the per-context networks the encoder consumes. Node MEMBERSHIP stays raw-count-derived (mean CP10k ≥ 0.5 ∩ OmniPath-incident — scVI supplies values, not the gene universe); the expression FEATURE is `log1p(mean scVI-normalized CP10k)` (the exact analog of the ComBat build's feature, on the scVI scale); edges are OmniPath, neutral. arm = healthy(normal) / disease-slug; tissue is per-source. Because states are the **integrated** scVI-latent Leiden states, the network set is **not** 1:1 with `_combat_loc` (ILD macrophages re-state as resident/inflammatory/proliferating rather than alveolar/interstitial/…).
- **Status:** **macrophage built** — staging (`run_scvi.py`, 115k cells × 6820 genes, 7 sources, states resident/inflammatory/proliferating) → adapter → **18 networks** (Crohn ileum+colon, UC colon, ILD lung, matched healthy) → encoder retrained (`joint_embed.py --expr-feat`, N=3891, 64-dim). The Crohn-vs-UC colon-macrophage disease-axis contrast is present (≈1.4–1.6k shared nodes per state). Fibroblast/microglia/stem not yet staged, so this build is **macrophage-only** (no Alzheimer coverage).
- **In-silico perturbation (scVI, `insilico_perturb/`, inflammatory state, Crohn/UC/ILD).** Two readouts point different ways; both are real:
  - *Whole-list residual test* — regress |projection| ~ out_degree + |Δexpr|, then Spearman(residual, OT): **negative** = −0.151 / −0.189 / −0.235 (all p<0.001). Targets do **not** rank above what wiring+DE predict, so there is **no wiring-independent target signal** — this reproduces the step-7 finding, now also on the scVI feature + integrated states (a plain link-prediction encoder). Top residual movers are generic TF hubs (NFKB1, HIF1A, EGR1, REL, FOS, MYC).
  - *Top-of-list recovery* (measured for Crohn colon inflammatory only) — ranking proteins by raw projection (toward-healthy) gives **above-random early enrichment**: at OT>0.3, MRR 0.024 = **6.1× random (perm p=7e-4)**, Hits@10 3/70, Hits@50 8/70; at OT>0.5, MRR **9.7× random (p=0.012)** but 48–88% of it carried by TNF alone. **Top-decile enrichment is the clearer read:** in the top 10% of ranks (209/2082), **6/9 OT>0.5** (random ≈0.9, ~6.7×) and **29/70 OT>0.3** (random ≈7, ~4.1×). Recall is flat deeper down — **Hits@1000 = 44/70 ≈ the random baseline (33.6)**. So real Crohn NF-κB/TNF-axis targets (NFKB1, TNF, IRF1, PRDM1, RIPK2, TNFAIP3, CD40, PTGS2) concentrate near the top, **but because they are high-DE / high-degree** (hence the negative residual), not from a target-specific response — and the enrichment does not extend to whole-list ordering. Net: the flat-"null" framing was too strong; honest statement is *weak, DE/degree-driven top enrichment, no wiring-independent signal.*
  - *Dose sweep* (`insilico_dose.py`, set each gene to 0.5×/1×/2× healthy expression) — OT>0.3 genes are slightly more dose-monotonic (87% vs 77%); ranking by dose **slope is worse** than the single point (MRR 0.012 vs 0.025); using monotonicity as a **filter** gives a minor lift (MRR 0.025→0.031 at OT>0.3) but drops 9/70 real targets. Adds a small cleanup, not new signal. Table: `insilico_perturb/…_dose_sweep.tsv`.
- **PCA** (`images/pca_{networks,proteins}.png`): PC1 dominates (96% network-level); disease arms do not cleanly separate within macrophage — consistent with disease being a faint axis on this objective.

## 8. `_masked`  — masked-feature-modeling pretraining
> **Status: build dir deleted (2026-07-02), no surviving metrics.** The `_masked`/`_masked_smoketest` dirs were untracked and were removed during a session; **no result number survives**, so any earlier claim that it "beat baseline" is unsupported. Code (`train_masked.py`) is intact; re-run to re-establish the number.
- **Change:** new self-supervised objective (`train_masked.py`) — mask a subset of nodes' expression, predict it from the **context-only** encoder (`use_expr_feat=True`, identity table bypassed), evaluate on **held-out nodes** vs a predict-the-mean baseline. Placeholders excluded from scoring.
- **Why:** link-prediction rewards rebuilding the graph → embedding ≈ degree, and the perturbation target test (step 7) was null. Masked feature modeling instead rewards learning **context → expression**, a more general representation, at **no extra data cost**. Held-out-node reconstruction is the transferability check; leave-one-study-out is the planned stronger test.

## 9. `_masked_delta`  — masked-FM + masked differential disease-direction aux loss  *(first run: NULL)*
- **What:** the full objective in CONTEXT_EMBED.md "Method 4 (refined)": `L = L_mask + λ_aux·L_aux + λ_link·L_link` on the `_coexpr_healthyph` base. `L_aux` = masked differential head `aux(ΔZ_p) ≈ expr_disease(p) − expr_healthy(p)` over 21 matched disease↔healthy pairs, with the target protein's own expression **masked in both passes** (kills the expression-passthrough shortcut that made the old `healthy_centered` trivial). Code: `scripts/context_embed/train_masked_delta.py`.
- **Result (2026-07-02, lr 1e-2, λ_aux 1, λ_link 0.25, 300 ep, CPU):** **both held-out R²_vs_baseline negative** — recon −0.021 (18/44 nets >0), delta −0.014 (5/21 pairs >0, `crohn_colon_macrophage_inflammatory` the worst at −0.092). `L_aux` flat in training (0.057→0.047); `L_mask` bounced (0.41–0.48). So neither term beats predict-the-mean here.
- **Not a clean test.** The primary `L_mask` is also ≈0. Whether plain masked-FM (step 8) beats baseline on this data is **unknown** — its build dir was deleted (untracked, no surviving number), so we cannot claim this run regressed a working baseline. The result is confounded by additions relative to step 8: the link anchor competing (loss ~1.58 ≫ L_mask), the **global shared mask set** (same ~1.5k universe nodes masked in every net → co-masked neighbourhoods degrade context), and possibly lr. **Next (to isolate):** first establish whether plain masked-FM beats predict-the-mean here at all; then add `L_aux` alone (`--lam-link 0`); then the anchor; consider lr 1e-3 / lower mask_frac / per-net masking. Build artifacts kept in `results/crohn_alzheimer_ild_uc_masked_delta/` for the record.
