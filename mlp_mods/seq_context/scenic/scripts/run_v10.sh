#!/bin/bash
# v10 = full expanded set: all 132 cisTarget contexts (existing 40 + heart_valve 12 + T-cell 80).
# ESM encoder, cisTarget labels, hard negatives (no --exclude). Then UMAP + per-disease pooled boxplots.
set -u
cd /home/caceves/context_schema_agent
VENV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
S=mlp_mods/seq_context
ts(){ date +%H:%M:%S; }
echo "[$(ts)] v10 train start (full 132 contexts, incl T cells)"
$VENV $S/scripts/train_link_context.py --run link_v10 --protein-repr esm --labels cistarget --neg hard \
  --epochs 60 > $S/results/link_v10.log 2>&1
echo "[$(ts)] v10 trained -> UMAP"
$VENV $S/scripts/plot_embeddings.py --run link_v10 > $S/images/umap_v10.log 2>&1
echo "[$(ts)] UMAP done -> per-disease boxplots"
$VENV $S/validation/per_disease_target_pooled.py --run link_v10 > $S/validation/per_disease_v10.log 2>&1
echo "[$(ts)] v10 PIPELINE COMPLETE"
