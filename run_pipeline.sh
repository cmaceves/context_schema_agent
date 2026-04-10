#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/scripts"

# PYTHONUNBUFFERED=1 ../.venv/bin/python3 -u schema_agent.py \
#     --mode async \
#     --resume \
#     --iterations 6 \
#     --budget 1.50 \
#     2>&1 | tee ../output/run_log.txt

PYTHONUNBUFFERED=1 ../.venv/bin/python3 -u schema_agent.py \
    --mode async \
    --drug_disease_test \
    2>&1 | tee ../output/run_log.txt
