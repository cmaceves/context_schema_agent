#!/bin/bash
# Wait for the (orphaned) v10 training to finish, then run the pooled all-target classifier boxplot. No UMAP.
set -u
cd /home/caceves/context_schema_agent
VENV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
S=mlp_mods/seq_context
ts(){ date +%H:%M:%S; }
echo "[$(ts)] waiting for v10 training to finish..."
while ps -eo cmd | grep -q '[t]rain_link_context.py --run link_v10'; do sleep 60; done
if [ ! -f $S/results/link_v10/embeddings.npz ]; then echo "[$(ts)] ERROR: no v10 embeddings.npz"; exit 1; fi
echo "[$(ts)] v10 training done -> classifier boxplot + table"
$VENV $S/validation/v10_classifier_boxplot.py --run link_v10 > $S/validation/v10_classifier.log 2>&1
echo "[$(ts)] v10 classifier boxplot COMPLETE"
