"""Make node membership PER-CONTEXT: keep only genes expressed in each network (not the fixed per-celltype
shared set). Fixes the artifact where a gene expressed in one context (e.g. BEX3 in ileum macrophage) is
carried, at ~0 expression, into every macrophage network (e.g. colon) where its embedding shift is
topology-driven noise.

Clones <src>/networks and <src>/controls/networks -> <dst>, and in each network keeps only protein nodes with
expression >= FLOOR (default log1p(0.5) ≈ 0.405, matching the CP10k>=0.5 expressed cutoff), then re-subsets
edges to the retained nodes. Node feature / edge weights (e.g. coexpression) are preserved for kept nodes.
Different networks therefore get DIFFERENT node sets (membership varies by tissue/celltype/state/arm).

After: retrain the encoder on <dst> (context_embed/train_context_embed.py --base <dst>).

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/apply_expression_filter.py \
        --src crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr \
        --dst crohn_alzheimer_ild_uc_coexpr_exprfilt
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd

RES = Path("mlp_mods/de_ppi/results")


def filter_dir(nets_dir: Path, floor: float):
    tot_before = tot_after = n = 0
    for d in sorted(nets_dir.iterdir()):
        nf, ef = d / "network_nodes.tsv", d / "network_edges.tsv"
        if not (nf.exists() and ef.exists()):
            continue
        nodes = pd.read_csv(nf, sep="\t", keep_default_na=False)
        keep = (nodes["node_type"] != "protein") | (nodes["expression"].astype(float) >= floor)
        kept = nodes[keep]
        ids = set(kept["node_id"])
        e = pd.read_csv(ef, sep="\t", keep_default_na=False)
        e = e[e["source"].isin(ids) & e["target"].isin(ids)]
        kept.to_csv(nf, sep="\t", index=False)
        e.to_csv(ef, sep="\t", index=False)
        tot_before += len(nodes); tot_after += len(kept); n += 1
    return n, tot_before, tot_after


def main(src: str, dst: str, floor: float) -> int:
    s, d = RES / src, RES / dst
    for sub in ("networks", "controls/networks"):
        sp, dp = s / sub, d / sub
        if not sp.exists():
            continue
        if dp.exists():
            shutil.rmtree(dp)
        shutil.copytree(sp, dp)
        n, b, a = filter_dir(dp, floor)
        print(f"{sub}: {n} networks, protein-nodes {b} -> {a} (mean {a/max(n,1):.0f}/net, floor expr>={floor:.3f})",
              flush=True)
    print(f"\nwrote {d}\nnext: train_context_embed.py --base {dst} --method contrastive --res-name <out>")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--dst", default="crohn_alzheimer_ild_uc_coexpr_exprfilt")
    ap.add_argument("--floor", type=float, default=math.log1p(0.5))
    a = ap.parse_args()
    raise SystemExit(main(a.src, a.dst, a.floor))
