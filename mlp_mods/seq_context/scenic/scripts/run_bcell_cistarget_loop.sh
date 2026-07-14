#!/bin/bash
# Overlap B-cell cisTarget with the running B-cell GRNBoost2. Skips done + not-yet-ready contexts;
# exits when all kept B-cell contexts have edges_cistarget.tsv.
set -u
cd /home/caceves/context_schema_agent
PYSCENIC=/home/caceves/miniforge3/envs/pyscenic/bin/python
S=mlp_mods/seq_context
NET=$S/scenic/networks
ts(){ date +%H:%M:%S; }
N=$(sed -n '2,$p' $S/scenic/inputs/bcell/context_cells.tsv | awk -F'\t' '$3=="True"' | wc -l)
echo "[$(ts)] target B-cell contexts: $N"
while true; do
  $PYSCENIC $S/scenic/scripts/cistarget_prune.py --celltype bcell --workers 48 >> $S/scenic/bcell_cistarget.log 2>&1
  n=$(for d in $NET/*_bcell_*/; do [ -s "$d/edges_cistarget.tsv" ] && echo x; done | wc -l)
  echo "[$(ts)] bcell cisTarget: $n/$N"
  [ "$n" -ge "$N" ] && break
  sleep 90
done
echo "[$(ts)] B-cell cisTarget COMPLETE"
