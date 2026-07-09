#!/usr/bin/env bash
# Full context-embed build: train (method) -> infer baseline control nets through the new encoder ->
# score the a-j control ladder -> PCA. Networks + control nets are REUSED from the coexpr base
# (ComBat + cell states + coexpression + expression features); only the training OBJECTIVE differs.
# See de_ppi/CONTEXT_EMBED.md. Nothing here runs until you invoke this script.
#
# Usage:
#   bash mlp_mods/de_ppi/scripts/context_embed/run_context_build.sh contrastive       crohn_alzheimer_ild_uc_context_contrastive
#   bash mlp_mods/de_ppi/scripts/context_embed/run_context_build.sh healthy_centered  crohn_alzheimer_ild_uc_healthy_centered
set -euo pipefail

METHOD="${1:?method: contrastive|healthy_centered|baseline}"
RES="${2:?output build name under de_ppi/results/}"
BASE="${3:-crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr}"   # source of networks + control nets
PY=.venv/bin/python
R=mlp_mods/de_ppi/results

# NOTE: controls are intentionally OMITTED. The standard a-j ladder's disease control (g) is CIRCULAR for an
# objective trained on the arm label (contrastive) / disease delta (healthy_centered) — we optimized for exactly
# that. New generalization-based controls (shuffled-arm null, OT recovery, leave-one-study-out) are TBD.

echo "== [1/2] train encoder (method=$METHOD, base=$BASE) -> $RES =="
$PY mlp_mods/de_ppi/scripts/context_embed/train_context_embed.py \
    --base "$BASE" --method "$METHOD" --res-name "$RES" --expr-feat

echo "== [2/2] PCA of per-network mean embeddings =="
$PY mlp_mods/de_ppi/scripts/context_embed/plot_pca_context.py --main-name "$RES"

echo "DONE -> $R/$RES  (embeddings.npz, encoder.pt, images/pca_networks.png).  Controls: TBD."
