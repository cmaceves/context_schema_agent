#!/bin/bash
# Overlap endothelial cisTarget with the running endothelial GRNBoost2. Skips done + not-yet-ready contexts;
# exits when all 70 have edges_cistarget.tsv.
set -u
cd /home/caceves/context_schema_agent
PYSCENIC=/home/caceves/miniforge3/envs/pyscenic/bin/python
S=mlp_mods/seq_context
NET=$S/scenic/networks
ts(){ date +%H:%M:%S; }
N=$(sed -n '2,$p' $S/scenic/inputs/endothelial/context_cells.tsv | awk -F'\t' '$3=="True"' | wc -l)
echo "[$(ts)] target endothelial contexts: $N"
while true; do
  $PYSCENIC $S/scenic/scripts/cistarget_prune.py --celltype endothelial --workers 48 >> $S/scenic/endo_cistarget.log 2>&1
  n=$(for d in $NET/*_endo_*/; do [ -s "$d/edges_cistarget.tsv" ] && echo x; done | wc -l)
  echo "[$(ts)] endo cisTarget: $n/$N"
  [ "$n" -ge "$N" ] && break
  sleep 90
done
echo "[$(ts)] endothelial cisTarget COMPLETE"
