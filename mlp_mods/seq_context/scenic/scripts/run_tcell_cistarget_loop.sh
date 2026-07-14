#!/bin/bash
# Keep building T-cell cisTarget labels (overlapping the still-running T-cell GRNBoost2) for a later v10.
# Skips done + not-yet-ready contexts each pass; exits when all 80 have edges_cistarget.tsv.
set -u
cd /home/caceves/context_schema_agent
PYSCENIC=/home/caceves/miniforge3/envs/pyscenic/bin/python
S=mlp_mods/seq_context
NET=$S/scenic/networks
ts(){ date +%H:%M:%S; }
while true; do
  $PYSCENIC $S/scenic/scripts/cistarget_prune.py --celltype tcell --workers 48 >> $S/scenic/tcell_cistarget.log 2>&1
  n=$(ls $NET/*tcell*/edges_cistarget.tsv 2>/dev/null | wc -l)
  echo "[$(ts)] tcell cisTarget: $n/80"
  [ "$n" -ge 80 ] && break
  sleep 90
done
echo "[$(ts)] tcell cisTarget COMPLETE (ready for v10)"
