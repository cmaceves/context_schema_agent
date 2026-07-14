#!/bin/bash
# Rerun v11 with the COMPLETE endothelial set once all 70 endothelial cisTarget contexts are done.
# Overwrites link_v11 with the full-endothelial build (202 contexts). Waits, then retrains + boxplot.
set -u
cd /home/caceves/context_schema_agent
VENV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
S=mlp_mods/seq_context; NET=$S/scenic/networks
ts(){ date +%H:%M:%S; }
echo "[$(ts)] v11 rerun watcher: waiting for endothelial cisTarget 70/70 ..."
while true; do
  n=$(for d in $NET/*_endo_*/; do [ -s "$d/edges_cistarget.tsv" ] && echo x; done | wc -l)
  [ "$n" -ge 70 ] && break
  sleep 120
done
tot=$(ls -d $NET/*/edges_cistarget.tsv 2>/dev/null | wc -l)
echo "[$(ts)] endothelial complete ($n/70) -> v11 RERUN on $tot contexts"
$VENV $S/scripts/train_link_context.py --run link_v11 --protein-repr esm --labels cistarget --neg hard \
  --epochs 60 > $S/results/link_v11.log 2>&1
$VENV $S/validation/v10_classifier_boxplot.py --run link_v11 > $S/validation/v11_classifier.log 2>&1
echo "[$(ts)] v11 RERUN COMPLETE (full endothelial, $tot contexts)"
