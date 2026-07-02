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

## Method 4 — healthy-centered / represent the delta  (dir: `..._healthy_centered`)
Add an auxiliary head that reconstructs the disease-vs-healthy EXPRESSION change from the embedding DELTA:
`aux(Z_disease[p] − healthy_centroid[p]) ≈ expr_disease[p] − mean_healthy_expr[p]`. Forces the healthy-centered
shift (not absolute position, which is degree/identity-dominated) to encode the disease change.

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
