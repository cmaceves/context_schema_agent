# Build history — de_ppi learned PPI embeddings

Chronological record of the network **builds** under `results/` and **why each one changed**. This is the
narrative companion to `METHODS.md` (which documents what runs today), `CONTEXT_EMBED.md` (graph training
objectives), and `SEQ_CONTEXT_EMBED.md` (the sequence+context→expression model, a no-graph branch). Each build
is a new `results/<name>/` dir; **originals are never overwritten**.

Three things evolve across the lineage, roughly independently:
1. **Node membership** — which proteins are nodes in each network (PINNACLE backbone → expression → per-context → placeholder-augmented).
2. **Node feature / edges** — how expression is corrected (raw → ComBat → scVI) and edge weights (topology → coexpression).
3. **Training objective** — how the encoder is trained (link-prediction → contrastive → masked feature modeling).

Lineage (main branch):
```
(PINNACLE backbone) → _expressed → _combat_loc → _combat_loc_coexpr ─┬─→ _coexpr_exprfilt
                                                                     └─→ _coexpr_healthyph ─┬─→ _context_contrastive
                                                                                            ├─→ _masked
                                                                                            └─→ _masked_delta
side branch:  _pinnacle_combat_ct (NOD2/TNF readmission)
scVI branch (MACROPHAGE ONLY):  _expressed_scvi ─┬─→ _protein_linked      (multiplex coupling, ω)
                                                 ├─→ _scvi_esm            (ESM + topology, context-free)
                                                 ├─→ _scvi_expr_esm       (expression + ESM)
                                                 ├─→ _scvi_masked         (masked feature modeling, clean rerun)
                                                 └─→ _scvi_exprsep        (copy-arrangement by expression: distance & diffvec-cosine)
controls (target-recovery gate):  differential_expression_control (DE)   random_walk_control (PageRank)
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
  - *DE-alone baseline* (`results/differential_expression_control/`, `de_target_rank.py`) — the repurposing "does the graph beat DE?" gate. DE baseline = logistic on |disease−healthy| expression; ranks compared for the 9 OT>0.5 Crohn-colon-inflammatory targets. Perturbation ranks them **better than DE on this set** (median rank 166 vs 891 of 2082; 7/9 below the diagonal), and the two rankings **disagree** (Spearman +0.20, p=0.61) — so the perturbation is not just re-expressing DE. Mechanistic split: hubs NR3C1/STAT3 are DE-invisible (~rank 1000) but perturbation top-20; the ligand TNFSF15 is DE-visible (43) but perturbation-blind (1469, out-degree 0). **Caveat:** 9 targets, one context, and the perturbation's edge leans on node degree — not conclusive. Proper gate = incremental cross-validated AUPRC `OT ~ DE + out_degree + projection` across contexts (pending). Plot: `differential_expression_control/images/de_vs_perturbation_ranks.png`.
- **PCA** (`images/pca_{networks,proteins}.png`): PC1 dominates (96% network-level); disease arms do not cleanly separate within macrophage — consistent with disease being a faint axis on this objective.

## 8. `_masked`  — masked-feature-modeling pretraining
> **Status: build dir deleted (2026-07-02), no surviving metrics.** The `_masked`/`_masked_smoketest` dirs were untracked and were removed during a session; **no result number survives**, so any earlier claim that it "beat baseline" is unsupported. Code (`train_masked.py`) is intact; re-run to re-establish the number.
- **Change:** new self-supervised objective (`train_masked.py`) — mask a subset of nodes' expression, predict it from the **context-only** encoder (`use_expr_feat=True`, identity table bypassed), evaluate on **held-out nodes** vs a predict-the-mean baseline. Placeholders excluded from scoring.
- **Why:** link-prediction rewards rebuilding the graph → embedding ≈ degree, and the perturbation target test (step 7) was null. Masked feature modeling instead rewards learning **context → expression**, a more general representation, at **no extra data cost**. Held-out-node reconstruction is the transferability check; leave-one-study-out is the planned stronger test.

## 9. `_masked_delta`  — masked-FM + masked differential disease-direction aux loss  *(first run: NULL)*
- **What:** the full objective in CONTEXT_EMBED.md "Method 4 (refined)": `L = L_mask + λ_aux·L_aux + λ_link·L_link` on the `_coexpr_healthyph` base. `L_aux` = masked differential head `aux(ΔZ_p) ≈ expr_disease(p) − expr_healthy(p)` over 21 matched disease↔healthy pairs, with the target protein's own expression **masked in both passes** (kills the expression-passthrough shortcut that made the old `healthy_centered` trivial). Code: `scripts/context_embed/train_masked_delta.py`.
- **Result (2026-07-02, lr 1e-2, λ_aux 1, λ_link 0.25, 300 ep, CPU):** **both held-out R²_vs_baseline negative** — recon −0.021 (18/44 nets >0), delta −0.014 (5/21 pairs >0, `crohn_colon_macrophage_inflammatory` the worst at −0.092). `L_aux` flat in training (0.057→0.047); `L_mask` bounced (0.41–0.48). So neither term beats predict-the-mean here.
- **Not a clean test.** The primary `L_mask` is also ≈0. Whether plain masked-FM (step 8) beats baseline on this data is **unknown** — its build dir was deleted (untracked, no surviving number), so we cannot claim this run regressed a working baseline. The result is confounded by additions relative to step 8: the link anchor competing (loss ~1.58 ≫ L_mask), the **global shared mask set** (same ~1.5k universe nodes masked in every net → co-masked neighbourhoods degrade context), and possibly lr. **Next (to isolate):** first establish whether plain masked-FM beats predict-the-mean here at all; then add `L_aux` alone (`--lam-link 0`); then the anchor; consider lr 1e-3 / lower mask_frac / per-net masking. Build artifacts kept in `results/crohn_alzheimer_ild_uc_masked_delta/` for the record.

## 10. `_protein_linked` — multiplex coupling of a protein across contexts  *(macrophage; NULL for targets)*
- **What:** on the **scvi** build, build one supra-graph (nodes = protein×network; intra edges = each network's PPI; inter edges = the SAME protein across the contexts it appears in, weight ω). One encoder forward over the whole thing (`train_protein_linked.py`, `--couple all`). Motivation: stop a protein's per-context versions being blind to each other. **lr 1e-3 required — 1e-2 dead-ReLU-collapses to all-zero embeddings** (a single large full-batch graph; same trap later hit the ESM builds).
- **Effective coupling ≫ nominal:** because the operator is receiver-normalized, ω=0.1 gives a **median ~21%** of each node's incoming signal from its own cross-context copies (degree-dependent: low-degree ~29%, hubs ~2%); ω=0.02 → ~5%. Built at both ω=0.1 and ω=0.02.
- **Effect:** coupling pulls a protein's versions together, so **protein identity dominates the geometry — by construction** (the uncoupled scvi space does NOT cluster a protein's versions; PCA/UMAP by disease/tissue/state show no separation). Loosening to ω=0.02 moves it back toward the uncoupled structure.
- **Target recovery (the point):** coupled perturbation is **≤ uncoupled scvi and ≈ PageRank** at both ω; loosening ω did not help. Coupling adds no target signal, slightly hurts. Analysis scripts + figures kept in the build dir.

## 11. ESM node-feature builds — `_scvi_esm` (A) and `_scvi_expr_esm` (B)  *(ESM doesn't beat topology on perturbation)*
- **ESM2 1280-d sequence embeddings** (`ESM/protein_embeddings.pt`, 3714/3891 covered, **L2-normalized**) added as node input via the `Encoder` `feat_dim` extension. lr 1e-3 (1e-2 collapses).
- **Build A (`_scvi_esm`): ESM + topology, no expression.** Link AUC **0.90** (sequence predicts PPI edges well), but **context-free** — ESM is identical in disease vs healthy, so there is no expression to perturb and no disease→healthy axis. Perturbation is **N/A**; it's a pure identity/interaction embedding. Only meaningful eval is supervised (fine-tune), not perturbation.
- **Build B (`_scvi_expr_esm`): expression + ESM + topology** (node feat = [expr | ESM], 1281-d; perturbation edits only the expr component, ESM fixed). Link AUC **0.90**; perturbation OT recovery **≈ expression-only (slightly worse) and ≈ PageRank** (OT>0.3 top-10% 26/70 vs scvi 29, PageRank 23; MRR 0.018 vs 0.024 vs 0.015). **ESM shifts positions but not the perturbation dynamics** (which change only expression), so the ranking stays topology-bound. No improvement.

## Target-recovery gate + baseline controls  *(the repurposing bar)*
- **Controls built:** `results/differential_expression_control/` (DE baseline = logistic on |disease−healthy| expression) and `results/random_walk_control/` (graph-only **PageRank**). Gate metric = OpenTargets recovery on `crohn_colon_macrophage_inflammatory` (N=2082): MRR / Hits@10,50 / top-decile.
- **Result across every build:** all perturbations **beat DE**, **none clearly beats PageRank**, and the perturbation ranking **correlates with PageRank at ρ=0.85**. OT>0.3 top-10%: scvi-pert 29, exprESM-pert 26, coupled 22–24, PageRank 23, DE 13. So the perturbation readout is **topology-bound** — node features (expression, coupling, ESM) don't lift it above graph centrality.
- **The one non-topology whiff:** a target's single most-outlying context leans disease-arm more than matched controls (borderline p=0.05); magnitude/spread/distance-to-healthy do **not** distinguish targets from controls (matched tests p≈0.4–0.9).
- **Implication:** a repurposing-useful representation needs the **supervised step** (fine-tune on known targets, PINNACLE-style — self-supervised pretrain THEN fine-tune) evaluated by **generalization** (LOSO, cross-disease), plus features that break the topology tie (ESM/GO). The unsupervised perturbation alone cannot clear the PageRank bar. Everything here is **macrophage-only** (scVI staged for macrophage only).

## Expression-signal & copy-arrangement experiments (macrophage scvi, 2026-07-08)
Question driving these: can we make expression (the only per-context input) actually shape the embedding, so a protein's contexts are comparable (UC↔Crohn)? Findings:

- **How much does expression already matter?** Variance of all (gene, context) expression values decomposes to **89% between-gene identity, 11% within-gene across-context** — the disease/context signal lives in that 11%, dwarfed by "which gene it is." **But** within a protein, on the plain `_expressed_scvi` build, **corr(embedding distance, expression difference) across contexts = 0.52** (Pearson; 0.42 Spearman, 218k pairs), and within-protein spread (0.14) ≈ 40% of between-protein spread (0.35). So expression is **not** irrelevant — it drives ~half of a protein's context-to-context movement; disease is a *faint, not null* axis, and UC↔Crohn context comparison rests on moderate real signal. (`_expressed_scvi` remains the best/base build.)

- **Masked feature modeling, clean rerun** (`_scvi_masked`, `train_masked.py`, scVI values, lr 1e-3): held-out node-expression reconstruction **R² = +0.016 vs predict-the-mean** (10/18 nets >0). Barely beats the mean — OmniPath neighbours carry little about a held-out gene's expression (most edges are signaling not coexpression; the context signal is only 11%). The earlier "poor masked-FM" holds even at clean lr; masked_delta's negative result was partly lr 1e-2 + confounds, but the clean version is still only marginally positive.

- **Copy-arrangement objectives** (`_scvi_exprsep`, iterated in-place): force a protein's copies apart ∝ expression difference.
  - *Rigid* (distance = α·expr-gap): held-out corr **0.998** — but **imposed, not learned** (one shared linear expression axis satisfies every protein) and it **re-creates an absolute-expression LEVEL gradient** (level and change share the single expression input direction). link AUC 0.80.
  - *One-sided* (distance ≥ α·expr-gap): corr 0.97 — floor binding, still near-linear.
  - *Difference-vector cosine, learned-identity input, NO raw expression* (`train_diffvec_cosine.py`): removes the level gradient, but **train corr 0.50 / held-out 0.17 → does NOT generalize**; link AUC 0.95; no expression axis; perturbation is N/A (no expression input).

- **Key conceptual result — the catch-22.** Arranging a protein's copies by expression **cannot be learned from the graph**, because the graph carries no expression (it's exogenous). So either expression is an **input** → the arrangement is a trivial passthrough of its own input *and* drags in the level gradient (generalizes only by re-reading the input); or it is **not** an input → nothing to place unseen proteins from, so the objective only memorizes training proteins (held-out corr collapses to ~0.17). **Expression positioning is a provided feature, not a pattern learnable from topology.** Corollary: builds with expression as a node feature are perturbable; builds without (ESM-only `_scvi_esm`, `_scvi_exprsep` diffvec) cannot be perturbed (nothing to edit).

- **Signed edges (assessed, not built).** OmniPath directed edges: **55% activation / 8% inhibition / 37% unsigned**, ~half transcriptional. Using them (rebuild edges + signed message-passing) is the principled way to carry regulator→target expression propagation, but only 8% are inhibition and the link-prediction objective still wouldn't reward disease signal → scoped experiment, **modest** expected payoff. The objective (supervised fine-tune), not the edge representation, is the target-recovery bottleneck.

- **Net across the whole arc:** no unsupervised objective (link-prediction, masked-FM, multiplex coupling, ESM features, copy-arrangement) beats the **PageRank** target-recovery ceiling. Expression's role is real but moderate (0.52 within-protein) and cannot be amplified into a *generalizing, target-relevant* axis without supervision. Standing recommendation unchanged: **supervised fine-tune on known targets, evaluated LOSO / cross-disease.**

## 12. `_signed` — directed signed-edge build + perturbation anatomy (macrophage scvi, 2026-07-08)
Built `results/crohn_alzheimer_ild_uc_embedding_expressed_scvi_signed/` — same node features as `_expressed_scvi` but a **two-channel encoder** (`train_signed.py`, `TwoChannel`): a signed stream (OmniPath sign ±1, `norm_op` row-normalizes by receiver in-strength of |weights|) + a neutral stream for the 37% unsigned edges. lr 1e-3. Trained loss 0.60, link AUC **0.826**. Perturbation (`insilico_perturb_signed*.py`) OT recovery ≈ scvi ≈ PageRank; across all contexts (`_all`) crohn_colon_inflammatory best (OT>0.5 MRR 0.043), ILD null — so the signal stays **cell-type-specific and topology-bound** (no lift over PageRank), consistent with §11 / the gate.

**Perturbation anatomy (dose-response + drug-level readout).** For OT>0.5 Crohn targets we swept each target's expression to {0,0.1,0.5,1,2,5,10}× its healthy value (linear CP10k) and measured the whole-network shift vs healthy (mean-over-present-proteins Euclidean + cosine). Scripts: `dose_response_signed.py`, `dose_response_controls.py`, `drug_target_perturb_by_phase.py`; degree table `target_degree.tsv`.
- **Sanity behavior:** distance-to-healthy bottoms out at the healthy level (dose ~0.5–1), rises away from it, and is **asymmetric** — over-expression (5–10×) displaces the network far more than knockdown (0–0.1×).
- **Only hubs move the network.** Euclidean range across the sweep: STAT3 0.061, NR3C1 0.025, TNF 0.013, then NFKBIA/PTPN2/IFNGR2/TAGAP/PTGER4 ≤0.006. This tracks **out-degree** (STAT3 243, NR3C1 143, TNF 79 vs the flat set 1–14). The flat targets aren't insensitive — they have almost no *outgoing* edges to broadcast a change; the readout (mean over ~1,600 proteins) then averages their local effect to nothing.
- **out vs in degree** (`target_in_out_degree.png`): the high-OT targets split into TF/hub **broadcasters** (STAT3/NR3C1/TNF, high out) and **receivers** (NFKBIA in 68, PTGER4 in 12, IFNGR2 in 9, TNFSF15 0-out/7-in) that a *forward* perturbation cannot exploit. Drug targets skew toward receptors/enzymes = low out-degree, so the whole-network readout is structurally blind to them.
- **TNF anomaly resolved.** TNF has out-deg 79 but a small effect (0.013) because (a) fewer receivers than STAT3/NR3C1 and (b) its out-neighbors have higher in-degree (median 19, mean 29 vs ~11/17) so each message is diluted more. Δexpr is equal across the three (input range 0→10× ≈ log 2.2–2.6). It is the **receivers'** in-degree that dampens, via `norm_op` row-normalization (each receiver's incoming weights sum to 1 → update = weighted average, so a source with many co-inputs contributes little). Nothing to do with TNF's own in-degree.
- **Controls are indistinguishable at matched degree** (`dose_response_controls.png`): out-degree-matched non-target controls trace the same curves; the only reason the target plot's top looks bigger is that **the hubs *are* targets** — no non-OT protein has out-deg above ~14, so out-degree and target-status are confounded and the whole-network readout cannot separate them.
- **OmniPath edge limits (confound behind the above):** the edge table is `src,dst,sign,layer` only — **unweighted** (sign ∈ {−1,0,+1}; 145.9k activation / 96.7k unsigned / 20.5k inhibition) — so `norm_op` normalizes by edge *count*, not strength. In-degree is **curation-biased**: top in-degree = TP53, EGFR, CDKN1A, STAT3, MDM2, SRC, MYC (the most-studied proteins), skewed (median 5, mean 15, max 604). So count-normalization imports study bias, and "receiver dilution" partly reflects annotation density, not biology. Count-based degree normalization is a robustness proxy, **not** a biological law (ignores dominant master regulators). Coexpression as an *edge feature* would not fix this (the count-normalization still runs); strength-weighted edges or learned/attention weights would, but attention on a link-prediction objective tends to re-learn hub bias.
- **Drug-target-by-phase (`drug_target_perturb_by_phase.png`, negative):** perturbed each Crohn known-drug target (`known_drugs_EFO_0000384.tsv`, target's max clinical phase) + out-degree-matched controls; y = max−min Euclidean across doses. **Only 9 of 31 targets are present** in the macrophage network with healthy>0 (ITGA4/ITGB7=vedolizumab, IL23A=ustekinumab, CRBN/CUL4A/DDB1=thalidomide live in T-cell/epithelial/other contexts). Effect tracks **out-degree, not phase**: Phase 4 (n=2) = NR3C1+TNF hubs (confounded, no degree-matched control exists); Phase 3 (n=7, low-deg receptors ALOX5/CSF2RA/PTGS2/CD86) sits *at or below* controls; incidental high-degree controls (EIF5B, USF2) beat every Phase-3 target. No drug-phase signal beyond degree.

**Two levers (framing) + repurposing.** The pipeline has exactly two readouts. **Lever 1** = perturb a gene → whole-network shift toward healthy = *driver/influence* ("does changing it help?"), out-degree-bound. **Lever 2** = a protein's own disease-vs-healthy embedding distance = *marker/dysregulation* ("is it changed?"), ≈ DE. Good targets need both; **DE-seeded RWR wins (OT>0.5 MRR 0.138)** precisely because it seeds from Lever-2 and propagates via Lever-1. Repurposing bridge = protein score → drugs hitting that protein: Lever 1 also gives **direction** (dose-response says knockdown vs over-expression helps → inhibitor vs agonist), and face-validity holds — top drivers TNF/NR3C1/STAT3 = infliximab / corticosteroids / tofacitinib targets. The repurposing *edge* is **context-specificity** (score per cell type). Honest next validation = recover approved IBD *drugs* (drug-level), not OT protein-associations.

**Best-improvement diagnosis (no-availability-constraints discussion).** Bottleneck is not propagation mechanics but that we score targets with an unsupervised geometric proxy on a biology-invariant graph. Ranked: (1) **supervise against measured perturbation outcomes** (LINCS/Perturb-seq) — ideal but **blocked** (LINCS is mostly cancer lines, no colon macrophage; no matched Perturb-seq); (2) **strength-weighted/causal edges** raise the ceiling; (3) **linear-response / steady-state readout** — model expression as a resting balance `x=(1−r)Wx+r b`, clamp gene *i*, solve, read the whole-network Δ (one column of a matrix inverse; sign-aware; reaches the full downstream path with distance attenuation instead of one GNN layer) — the most promising given no perturbation data, and it also yields **path/mechanism** support that rescues low-out-degree receivers (a receiver can be supported by the path it acts *through*, e.g. TNFSF15→…→NF-κB); (4) degree-norm / attention / combination perturbations are refinements inside the topology-only frame (marginal on single-target recovery; combinations useful for the *synergy* question). **Still missing entirely: metabolites** (HMDB-side nodes not represented in the PPI networks).

**Supervised target classifier + degree/DE ablation (`target_classifier.py`).** Direct test of "do the embedding coordinates carry target signal beyond degree+DE?" Labels = OpenTargets *association* (positive OT>0.3, negative OT=0; not clinical drug targets), over proteins present in the context; logistic regression (balanced), 5-fold CV + Crohn→UC transfer. Features: BASE = [log out-deg, log in-deg, disease expr, |Δexpr|]; EMB = signed disease-context embedding (64-d); EMB+BASE. **Result: the embedding is a *worse* feature than degree+DE and adding it *hurts*.** Crohn CV AUC/AP: BASE **0.843 / 0.412**, EMB 0.678 / 0.140, EMB+BASE 0.794 / 0.374 (random AP = prevalence 0.055). Crohn→UC transfer: BASE **0.858 / 0.454**, EMB 0.661 / 0.122, EMB+BASE 0.755 / 0.351. So even a supervised head extracts **no** target signal from the embedding beyond degree+DE; the embedding is a noisier re-encoding of degree. The genuinely useful finding is that **degree+DE is a strong, cross-disease-generalizing classifier** (AP 0.41→0.45 Crohn→UC) — though note OT labels favor well-studied (high-degree) genes, the same curation bias, so classifier and label share that confound.

**Signed+ESM build — `_scvi_expr_esm_signed` (`train_expr_esm_signed.py`).** Combined the two-stream signed encoder with the 1281-d `[expr | ESM]` input (`TwoChannelFeat`; ESM cov 3714/3891, 58% signed edges). **Held-out link AUC 0.909 — highest of the lineage** (signed 0.826, expr+ESM 0.90) → sequence helps edge prediction. **But perturbation OT recovery (`insilico_perturb_signed_esm.py`, edits only the expr column) ≤ PageRank and ≤ plain scvi:** OT>0.5 MRR 0.043 (4/9) < PageRank 0.052; OT>0.3 MRR 0.017 (22/70) ≈ PageRank, < scvi 0.024 (29/70). Higher link AUC bought nothing for targets — ESM is context-free and perturbation edits only expression, so it shifts positions but not dynamics. Another confirmation of the ceiling.

**Consolidated OT-recovery table** (all on `crohn_colon_macrophage_inflammatory`, N=2082, recomputed consistently 2026-07-08; random OT>0.5 MRR ≈ 0.004):

| method | OT>0.5 MRR | OT>0.5 top-10% | OT>0.3 MRR | OT>0.3 top-10% |
|---|---|---|---|---|
| DE ( \|Δexpr\| )            | 0.009 | 3/9 | 0.005 | 13/70 |
| **PageRank**               | 0.052 | 5/9 | 0.015 | 23/70 |
| **DE-seeded RWR**          | **0.138** | 5/9 | **0.028** | 22/70 |
| scvi perturbation          | 0.038 | **6/9** | 0.024 | **29/70** |
| signed perturbation        | 0.043 | 5/9 | 0.018 | 28/70 |
| expr+ESM perturbation      | 0.039 | 5/9 | 0.018 | 26/70 |
| signed+ESM perturbation    | 0.043 | 4/9 | 0.017 | 22/70 |

Reading: **DE-seeded RWR wins top-of-list (MRR)** by a wide margin at OT>0.5 (0.138, 2.6× PageRank) and edges it at OT>0.3; **embedding perturbations get slightly more targets into the top decile than PageRank** (scvi 29 vs 23 at OT>0.3) but never beat PageRank's MRR at OT>0.5; **DE alone is worst**; **ESM/signed variants add nothing** (all clustered 0.038–0.043 / 0.017–0.024). The supervised classifier (above) confirms the same verdict from the other direction. Net: simple topology+DE baselines meet or beat every learned embedding; the standing recommendation (supervision that isn't the OT-association proxy, or the linear-response readout) is unchanged.

## 13. `seq_context` — ESM+context embedding, SCENIC regulatory-neighbor link prediction  *(PLAN + SCENIC gen; no graph joint-embed; 2026-07-08)*
New direction, **not** part of the joint-embed lineage — full spec in `SEQ_CONTEXT_EMBED.md`, working dir
`seq_context/`. Learns a **context-specific protein embedding** = MLP(frozen ESM 1280-D ⊕ learned embeddings
cell_type 64 / disease 32 / tissue 32 / state 32), trained by **link prediction (BCE) on per-context SCENIC
regulatory networks** (predict a protein's TF→target neighbors *in that context*). The per-protein context
embedding is the deliverable; edge reconstruction is only the signal.
- **Objective evolution:** started as predict-context-*expression* (single scalar → under-determines the
  embedding), then evaluated four label families; kept the **only context-varying one — regulatory neighbors
  (SCENIC)**; static labels (pathway membership, functional annotation, complexes) can't teach context and are
  demoted to a **GO biology-retention probe** on the per-protein mean embedding.
- **Explicit non-goals / honest caveats:** unseen-context generalization is **out of scope** (contexts are named
  ID-embeddings, not described — never achievable off a name; user does not care); SCENIC's context signal is
  co-expression, so labels transform expression but are **not leakage** while expression is not a model input;
  key metric is **context-lift** vs a context-blind ablation (degree alone reconstructs edges — the old
  topology-bound AUC 0.90), *not* raw edge AUC. `cell_type` constant (macrophage-only build) = no-op for now.
- **SCENIC generation (in progress):** cells from `_embedding_expressed_scvi/scvi_staging/macrophage.h5ad`
  (115k cells, raw `counts`, 6,820-gene universe, disease/state present; **tissue absent**, proliferating
  sparse). Open decisions: GRNBoost2-only vs full SCENIC (+cisTarget motif DBs); context granularity
  (disease×tissue×state vs disease×state); install target for pyscenic.
