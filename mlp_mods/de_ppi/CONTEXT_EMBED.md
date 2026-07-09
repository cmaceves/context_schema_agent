# CONTEXT_EMBED.md — sharpening the disease signal in the embedding (TEMPORARY working doc)

We showed the embedding *does* carry context (cell-type strongest, then state/tissue; disease ~2× batch
floor, weakest, Crohn≈UC), but the link-prediction objective never rewards disease separation, so the disease
axis is faint. This is a scratch experiment to test two objective changes that should sharpen it.

Both reuse the SAME staged networks as `crohn_alzheimer_ild_uc_embedding_expressed_combat_loc` (per-state,
neutral edges, ComBat expression) — only the TRAINING OBJECTIVE changes — so results are comparable to that
baseline and to each other. Code: `scripts/context_embed/`.

## Method 1 — context-contrastive / center loss  (dir: `..._context_contrastive`)
Add a center loss on the DISEASE ARM: for each protein present in multiple networks, pull its per-network
embeddings of the SAME arm together and push different-arm centroids apart (hinge, margin). Forces disease/
healthy (and disease-vs-disease) to be an explicit axis instead of a byproduct of edge reconstruction.

## Method 4 — healthy-centered / represent the delta  (dir: `..._healthy_centered`, implemented but never built)
Add an auxiliary head that reconstructs the disease-vs-healthy EXPRESSION change from the embedding DELTA:
`aux(Z_disease[p] − healthy_centroid[p]) ≈ expr_disease[p] − mean_healthy_expr[p]`. Forces the healthy-centered
shift (not absolute position, which is degree/identity-dominated) to encode the disease change.

### Method 4 (refined) — masked differential aux loss  [design; the version to build]
As implemented (`train_context_embed.py --method healthy_centered`), the head is
`aux = Linear(dim,dim)→ReLU→Linear(dim,1)`; per epoch it computes, over healthy nets, a per-protein healthy
embedding centroid `hz[p]` and expression centroid `hef[p]`, then for each disease net adds
`MSE( aux(Z_i[p] − hz[p]),  expr_i[p] − hef[p] )` to the baseline with weight λ. Gradients flow through the
encoder, so the **healthy-centered shift ΔZ_p** (not absolute position) is pushed to decode the disease
expression change.

**Why the as-implemented version is weak — expression passthrough.** With `use_expr_feat` the encoder input
already *is* expression (`Linear(1,dim)`), so `ΔZ_p` already carries `expr_disease − expr_healthy` almost
linearly, and the head can drive the MSE→0 by **reading its own input back out** — no topology learned. This
is the identity-shortcut in expression form and is the likely reason the method came out faint. **The single
change that fixes it: MASK the target protein's own expression** when predicting its own Δexpr, so `ΔZ_p`'s
disease signal must come from **neighbours' shifts**, not p's passed-through value (masked-FM fused with the
differential target). This turns the loss from an identity readout into a graph-using objective.

**Two smaller fixes:** (1) pair each disease net with its **matched-context healthy** (same tissue+state),
not the pooled cross-tissue `hz`/`hef` centroid — the scVI build has exact pairs
(`crohn_colon_macrophage_inflammatory ↔ healthy_colon_macrophage_inflammatory`) — removing cross-tissue
contamination of the Δexpr target; (2) if the single scalar target underpowers the 64-d geometry, widen it
(predict neighbours' Δexpr / a small vector).

**What it does / does not.** It makes the healthy-centered shift an explicitly decodable, neighbour-informed
representation of the disease expression change — a denoised disease-direction feature for the disease-axis and
target-prioritisation goals. It is **co-variation structure in our own cross-context data, not causal/
perturbational** (no interventional data; not attempting causation).

**Evaluation — generalization, NOT the deprecated a–j ladder** (CONTROLS.md): (a) leave-one-study-out aux MSE
vs a predict-the-mean baseline — does `shift → Δexpr` generalize to unseen studies/proteins; (b) does the
resulting `ΔZ` improve the supervised `ΔZ → OpenTargets` head (`embedding_target_cv.py`) under LOSO. If masking
is on and it still only matches passthrough, the graph adds nothing → drop it. Best run **on top of masked-FM
pretrained embeddings** rather than the raw link-prediction encoder.

**First run (2026-07-02) — NULL, but confounded.** `train_masked_delta.py` (base `_coexpr_healthyph`, λ_aux 1,
λ_link 0.25, lr 1e-2, 300 ep, CPU): held-out R²_vs_baseline = **−0.021 recon / −0.014 delta** — neither term
beats predict-the-mean; held-out `crohn_colon_macrophage_inflammatory` delta is worst (−0.092), and `L_aux`
stayed flat (~0.05) in training. Caveat: the primary `L_mask` also came in ≈0, so this is **not** a clean test
of `L_aux` — it's confounded by the link anchor competing (loss ≫ L_mask) and by the **global shared mask set**
(same universe nodes masked in every net → co-masked neighbourhoods). Isolation plan: first **establish whether
plain masked-FM beats predict-the-mean here at all** (the step-8 `_masked` dir was deleted, no surviving number),
then add `L_aux` alone, then the anchor; try lr 1e-3 / per-net masking. Result recorded in HISTORY.md §9;
artifacts in `results/crohn_alzheimer_ild_uc_masked_delta/`.

## Baseline objective (both keep it)
Directed link prediction (bilinear decoder + negative sampling) + edge-weight reconstruction, summed over
networks — i.e. exactly `joint_embed.py`, plus the method loss (weight λ).

## Outputs per method (results/<dir>/)
- `embeddings.npz`, `encoder.pt`                              — trained encoder + per-network embeddings
- `images/pca_networks.png`                                  — PCA of per-network mean embeddings, colored by
                                                               cell type, marker = disease arm (does disease separate?)

## Controls — TBD (deliberately omitted for now)
The standard a–j ladder is CIRCULAR here: both objectives are trained on the arm label, so the disease control
`g` (arm-differing nets) rises by construction, not because disease signal is real. Unsupervised floors
(`a`,`b`,`h`,`i`) stay valid as references, but `g` is not a fair test. New generalization-based controls to
design later: (1) shuffled-arm null (retrain with permuted arms — real must beat it on held-out biology);
(2) external OT shared/unique recovery; (3) leave-one-study-out reproducibility.

## Read-out (what would count as success)
On the control ladder, does the **disease control `g` rise relative to the batch floor `b`** vs the baseline
build? And in PCA, do networks cluster by cell type with **disease arms separating within a cell type**? If
neither moves, the objective change didn't add disease signal.

## Run
```
scripts/context_embed/train_context_embed.py --method contrastive     --res-name crohn_alzheimer_ild_uc_context_contrastive
scripts/context_embed/train_context_embed.py --method healthy_centered --res-name crohn_alzheimer_ild_uc_healthy_centered
# then infer_controls + compare_controls + plot_pca_context.py for each
```

## Copy-arrangement objectives (2026-07-08) — separate a protein's contexts by expression
Goal: make a protein's per-context copies sit close when its expression is similar and far when different, so
contexts (UC↔Crohn) are comparable — without the raw-expression level gradient. All on the scvi macrophage build.

- **`_scvi_exprsep` distance rule** (`train_exprsep.py`): pull a protein's copies so ‖ΔZ‖ ≈ α·|Δexpr|.
  Rigid two-sided → held-out corr **0.998** but IMPOSED (one shared linear expression axis) and **recreates a
  level gradient** (level & change share the single expression input direction). One-sided (≥) → 0.97, floor binding.
- **`_scvi_exprsep` diffvec-cosine** (`train_diffvec_cosine.py`): learned-identity input, NO raw expression;
  match copies' embedding cosine to the cosine of their expression difference-vectors. Removes the level gradient
  but **train corr 0.50 / held-out 0.17 — does not generalize**; link AUC 0.95.
- **Catch-22 (the load-bearing conclusion):** arranging copies by expression can't be *learned from the graph*
  (graph carries no expression). Expression as input → trivial passthrough + level gradient; expression not an
  input → can't generalize (memorizes training proteins, held-out ~0.17). Expression positioning is a *provided
  feature*, not a learnable-from-topology pattern.
- **Reference measurement:** on the plain `_expressed_scvi` build (no arrangement objective), copies already
  track expression at corr **0.52** (embedding distance vs |Δexpr|), within-protein spread ≈ 40% of between-protein.
  So the moderate context signal is present without any special objective; the objectives above either impose it
  trivially or fail to generalize. Full record: HISTORY.md "Expression-signal & copy-arrangement experiments".
