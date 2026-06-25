"""Make the crohn_alzheimer_ild_uc_embedding_expressed networks NEUTRAL-weighted: set every
sender_weight and edge weight to 1.0, leaving the node set, edges, source tags, and direction
untouched. The DE rank-shift weights barely move the embedding, and mixing weighted vs neutral
networks (e.g. weighted study-A inflammatory vs neutral study-B) created artifacts — neutral makes
every network apples-to-apples (topology + node-set composition only).

Equivalent to rebuilding every network with --neutral-weights (node sets are unaffected by weights),
but in place and fast. Run, then re-run joint_embed_influence + regenerate plots.

Run: .venv/bin/python mlp_mods/de_ppi/neutralize_weights.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from pathlib import Path
import pandas as pd

NET = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed/networks")


def main():
    changed = 0
    for d in sorted(NET.iterdir()):
        nf, ef = d / "network_nodes.tsv", d / "network_edges.tsv"
        if not nf.exists():
            continue
        n = pd.read_csv(nf, sep="\t", keep_default_na=False)
        e = pd.read_csv(ef, sep="\t", keep_default_na=False)
        was = (n["sender_weight"].astype(float) != 1.0).any() or (e["weight"].astype(float) != 1.0).any()
        n["sender_weight"] = 1.0
        e["weight"] = 1.0
        n.to_csv(nf, sep="\t", index=False)
        e.to_csv(ef, sep="\t", index=False)
        changed += int(was)
        print(f"  {d.name:34s} nodes={len(n)} edges={len(e)}  {'NEUTRALIZED' if was else '(already neutral)'}", flush=True)
    print(f"\ndone: {changed} networks neutralized (weights set to 1.0)", flush=True)


if __name__ == "__main__":
    main()
