#!/bin/bash
# Overlap T-cell cisTarget with the still-running T-cell GRNBoost2, then chain v9 + UMAP + per-disease boxplots.
# cisTarget skips done + not-yet-ready contexts, so it safely mops up as GRNBoost2 finishes. Run in background.
set -u
cd /home/caceves/context_schema_agent
PYSCENIC=/home/caceves/miniforge3/envs/pyscenic/bin/python
VENV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
NET=mlp_mods/seq_context/scenic/networks
S=mlp_mods/seq_context
ts(){ date +%H:%M:%S; }

echo "[$(ts)] pipeline start"
# 1. cisTarget overlap loop until all 80 T-cell contexts have edges_cistarget.tsv
while true; do
  $PYSCENIC $S/scenic/scripts/cistarget_prune.py --celltype tcell --workers 48 >> $S/scenic/tcell_cistarget.log 2>&1
  n=$(ls $NET/*tcell*/edges_cistarget.tsv 2>/dev/null | wc -l)
  echo "[$(ts)] tcell cisTarget: $n/80"
  [ "$n" -ge 80 ] && break
  sleep 90
done
echo "[$(ts)] tcell cisTarget COMPLETE"

# 2. v9 embedding — ESM, cisTarget, hard negs; trainer auto-discovers all cisTarget contexts (40+12+80=132)
$VENV $S/scripts/train_link_context.py --run link_v9 --protein-repr esm --labels cistarget --neg hard --epochs 60 \
  > $S/results/link_v9.log 2>&1
echo "[$(ts)] v9 trained"

# 3. UMAP
$VENV $S/scripts/plot_embeddings.py --run link_v9 > $S/images/umap_v9.log 2>&1
echo "[$(ts)] UMAP done"

# 4. per-disease pooled target boxplots (LogReg + MLP)
$VENV $S/validation/per_disease_target_pooled.py --run link_v9 > $S/validation/per_disease_v9.log 2>&1
echo "[$(ts)] per-disease boxplots done"
echo "[$(ts)] PIPELINE COMPLETE"
