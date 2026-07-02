"""Per-context node membership, but disease networks ALSO keep every node expressed in their PAIRED HEALTHY
network (same tissue/celltype/state) as a zero-feature placeholder.

Rationale: a plain per-context expression filter drops a gene from the disease net if it's unexpressed there —
even if it's expressed in healthy — which loses down-regulation (gene silenced in disease) and shrinks the
disease-vs-healthy overlap. Here:
  healthy net  node set = its expressed genes (expr >= FLOOR).
  disease net  node set = (disease-expressed)  UNION  (paired-healthy-expressed);
               expression = the disease value for disease-expressed nodes, 0.0 for healthy-only placeholders.
So healthy ⊆ its paired disease, and the disease-vs-healthy shift is computable for every healthy-expressed gene
(a silenced gene shows up as expr→0). Edges = the source build's edges (e.g. coexpression) over the kept set.

Symmetric complement: each HEALTHY net also gains, as ISOLATED zero-feature placeholders, every gene expressed
in a disease net of the EQUIVALENT tissue/celltype/state (same suffix, union over crohn/uc/alz/ild) but not
expressed in healthy. This makes disease-only genes (e.g. ITGA4, off in healthy) appear in both nets, so their
shift and in-silico-perturbation readout are defined. Healthy-side placeholders carry NO edges (isolated), so
they cannot alter any connected node's message passing — only their own (identity + zero-feature) embedding is
added. Disease-side placeholders keep their source edges as before.

Operates on MAIN networks (networks/); the encoder trains on those. Clones <src> -> <dst>.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/apply_healthy_placeholder.py \
        --src crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr \
        --dst crohn_alzheimer_ild_uc_coexpr_healthyph
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd

RES = Path("mlp_mods/de_ppi/results")
DISEASES = {"crohn", "uc", "alz", "ild"}


def main(src: str, dst: str, floor: float) -> int:
    s, d = RES / src, RES / dst
    snet = s / "networks"
    dnet = d / "networks"
    if dnet.exists():
        shutil.rmtree(dnet)
    dnet.mkdir(parents=True)

    nodefiles = {p.name: pd.read_csv(p / "network_nodes.tsv", sep="\t", keep_default_na=False)
                 for p in snet.iterdir() if (p / "network_nodes.tsv").exists()}
    expr_ids = {t: set(df.loc[(df.node_type == "protein") & (df.expression.astype(float) >= floor), "node_id"])
                for t, df in nodefiles.items()}

    for t, nodes in nodefiles.items():
        arm = t.split("_")[0]
        keep = set(expr_ids[t])
        placeholders = set()
        if arm in DISEASES:
            ht = "healthy_" + t.split("_", 1)[1]           # paired healthy (same tissue/celltype/state)
            if ht in expr_ids:
                placeholders = expr_ids[ht] - expr_ids[t]  # healthy-expressed, not disease-expressed
                keep |= placeholders
        elif arm == "healthy":
            suffix = t.split("_", 1)[1]                     # equivalent tissue/celltype/state only
            disease_expr = set()
            for dtag in (f"{d}_{suffix}" for d in DISEASES):
                if dtag in expr_ids:
                    disease_expr |= expr_ids[dtag]          # union over same-context disease arms
            placeholders = disease_expr - expr_ids[t]       # disease-expressed, not healthy-expressed
            keep |= placeholders
        kept = nodes[nodes.node_id.isin(keep)].copy()
        # zero the feature for placeholders (healthy-only in disease nets; disease-only in healthy nets)
        kept.loc[kept.node_id.isin(placeholders), "expression"] = 0.0
        e = pd.read_csv(snet / t / "network_edges.tsv", sep="\t", keep_default_na=False)
        # disease-side placeholders keep their source edges; healthy-side placeholders are ISOLATED (no edges)
        edge_nodes = (keep - placeholders) if arm == "healthy" else keep
        e = e[e.source.isin(edge_nodes) & e.target.isin(edge_nodes)]
        od = dnet / t; od.mkdir(parents=True, exist_ok=True)
        kept.to_csv(od / "network_nodes.tsv", sep="\t", index=False)
        e.to_csv(od / "network_edges.tsv", sep="\t", index=False)
        kind = "disease-only-isolated" if arm == "healthy" else "healthy-placeholder"
        print(f"  {t:42s} nodes={len(kept):5d} (expressed {len(expr_ids[t]):5d} + {kind:21s} {len(placeholders):4d})  edges={len(e)}",
              flush=True)
    print(f"\nwrote {dnet}\nnext: train_context_embed.py --base {dst} --method contrastive --res-name <out>")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc_coexpr")
    ap.add_argument("--dst", default="crohn_alzheimer_ild_uc_coexpr_healthyph")
    ap.add_argument("--floor", type=float, default=math.log1p(0.5))
    a = ap.parse_args()
    raise SystemExit(main(a.src, a.dst, a.floor))
