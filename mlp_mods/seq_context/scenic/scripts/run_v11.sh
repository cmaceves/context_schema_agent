#!/bin/bash
# v11 NOW: all cisTarget contexts currently on disk (v10's 132 + finished endothelial). ESM, cisTarget, hard neg.
# Then the pooled all-target classifier boxplot. No UMAP.
set -u
cd /home/caceves/context_schema_agent
VENV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
S=mlp_mods/seq_context
ts(){ date +%H:%M:%S; }
echo "[$(ts)] v11 train start (all cisTarget contexts incl finished endothelial)"
$VENV $S/scripts/train_link_context.py --run link_v11 --protein-repr esm --labels cistarget --neg hard \
  --epochs 60 > $S/results/link_v11.log 2>&1
echo "[$(ts)] v11 trained -> classifier boxplot"
$VENV $S/validation/v10_classifier_boxplot.py --run link_v11 > $S/validation/v11_classifier.log 2>&1
echo "[$(ts)] v11 COMPLETE"
