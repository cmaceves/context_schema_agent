#!/bin/bash
# Overlap bipolar cisTarget with the running bipolar GRNBoost2. Skips done + not-yet-ready contexts;
# exits when all kept bipolar contexts have edges_cistarget.tsv.
# NOTE: bipolar contexts are <arm>_brain_<celltype>_<state> (arm=bipolar|healthy), so a "*_bipolar_*" glob
# does NOT match them — completion is counted from inputs/bipolar/context_cells.tsv instead.
set -u
cd /home/caceves/context_schema_agent
PYSCENIC=/home/caceves/miniforge3/envs/pyscenic/bin/python
S=mlp_mods/seq_context
NET=$S/scenic/networks
ts(){ date +%H:%M:%S; }
N=$(sed -n '2,$p' $S/scenic/inputs/bipolar/context_cells.tsv | awk -F'\t' '$3=="True"' | wc -l)
echo "[$(ts)] target bipolar contexts: $N"
while true; do
  $PYSCENIC $S/scenic/scripts/cistarget_prune.py --celltype bipolar --workers 48 >> $S/scenic/bipolar_cistarget.log 2>&1
  n=$(sed -n '2,$p' $S/scenic/inputs/bipolar/context_cells.tsv | awk -F'\t' '$3=="True"{print $1}' \
       | while read -r ctx; do [ -s "$NET/$ctx/edges_cistarget.tsv" ] && echo x; done | wc -l)
  echo "[$(ts)] bipolar cisTarget: $n/$N"
  [ "$n" -ge "$N" ] && break
  sleep 90
done
echo "[$(ts)] bipolar cisTarget COMPLETE"
