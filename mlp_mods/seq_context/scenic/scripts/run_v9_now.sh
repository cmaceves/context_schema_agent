#!/bin/bash
# v9 NOW on the ready contexts (existing 40 + heart_valve 12 = 52), excluding the still-building T cells.
# ESM encoder, cisTarget labels, hard negatives. Then UMAP + per-disease pooled target boxplots.
set -u
cd /home/caceves/context_schema_agent
VENV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
S=mlp_mods/seq_context
ts(){ date +%H:%M:%S; }
echo "[$(ts)] v9 train start (exclude tcell)"
$VENV $S/scripts/train_link_context.py --run link_v9 --protein-repr esm --labels cistarget --neg hard \
  --epochs 60 --exclude tcell > $S/results/link_v9.log 2>&1
echo "[$(ts)] v9 trained -> UMAP"
$VENV $S/scripts/plot_embeddings.py --run link_v9 > $S/images/umap_v9.log 2>&1
echo "[$(ts)] UMAP done -> per-disease boxplots"
$VENV $S/validation/per_disease_target_pooled.py --run link_v9 > $S/validation/per_disease_v9.log 2>&1
echo "[$(ts)] v9 PIPELINE COMPLETE"
