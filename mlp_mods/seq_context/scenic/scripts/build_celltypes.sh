#!/bin/bash
# Build SCENIC networks for additional cell types end-to-end: scVI staging -> prep -> GRNBoost2.
# Then re-threshold ALL networks (incl. macrophage) to top-50 positives. See seq_context/SEQ_CONTEXT_EMBED.md.
cd /home/caceves/context_schema_agent || exit 1
SCV=/home/caceves/context_schema_agent/.venv_scvi/bin/python
STAGING=mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi/scvi_staging

for ct in fibroblast microglia stem; do
  echo "########## $ct : scVI staging ##########"
  if [ -f "$STAGING/$ct.h5ad" ]; then
    echo "staging $ct.h5ad exists — skip"
  else
    $SCV mlp_mods/de_ppi/scripts/embed/run_scvi.py --celltype "$ct" || { echo "!! SCVI FAILED $ct"; continue; }
  fi
  echo "########## $ct : prep_contexts ##########"
  $SCV mlp_mods/seq_context/scenic/scripts/prep_contexts.py --celltype "$ct" || { echo "!! PREP FAILED $ct"; continue; }
  echo "########## $ct : GRNBoost2 ##########"
  /home/caceves/miniforge3/envs/scenic/bin/python mlp_mods/seq_context/scenic/scripts/run_grnboost2.py \
      --celltype "$ct" --cap 5000 --workers 8 || { echo "!! GRN FAILED $ct"; continue; }
  echo "########## $ct : DONE ##########"
done

echo "########## threshold top-50 (all networks) ##########"
$SCV mlp_mods/seq_context/scenic/scripts/threshold_topk.py --k 50
echo "########## ALL CELL TYPES DONE ##########"
