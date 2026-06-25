"""Per-network protein-count table: disease / cell type / cell state, and how many proteins came
from the PINNACLE backbone vs were added by differential expression.

For each network_nodes.tsv under results/<out_name>/networks/<tag>/, proteins are tagged in the
`source` column (pipe-joined: pinnacle | de | literature_search | expressed). We report:
  PINNACLE      proteins carrying the 'pinnacle' tag (backbone node set)
  DE            proteins carrying the 'de' tag (DE-significant, padj<0.05)
  added via DE  proteins with 'de' but NOT 'pinnacle' (DE genes not already in the backbone)
  total         all protein nodes in the network

Outputs (results/<out_name>/influence_analysis/):
  network_protein_counts.tsv
  network_protein_counts.png

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/print_network_protein_counts.py \
      --out-name crohn_alzheimer_ild_embedding
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DE_PPI = Path("mlp_mods/de_ppi")
DIS = {"crohn": "Crohn", "alz": "Alzheimer", "alzheimer": "Alzheimer", "ild": "ILD"}
CT = {"mac": "macrophage", "macrophage": "macrophage", "stem": "stem",
      "microglia": "microglia", "fibroblast": "fibroblast"}
DIS_ORDER = {"Crohn": 0, "Alzheimer": 1, "ILD": 2}


def parse_tag(tag: str):
    p = tag.split("_")
    disease = DIS.get(p[0], p[0])
    cell_type = CT.get(p[1], p[1]) if len(p) > 1 else ""
    state = "_".join(p[2:]) if len(p) > 2 else ""
    return disease, cell_type, state


def has(src: pd.Series, tag: str) -> pd.Series:
    return src.fillna("").apply(lambda s: tag in s.split("|"))


def main(out_name: str) -> int:
    net_dir = DE_PPI / "results" / out_name / "networks"
    rows = []
    for nn in sorted(net_dir.glob("*/network_nodes.tsv")):
        tag = nn.parent.name
        disease, cell_type, state = parse_tag(tag)
        d = pd.read_csv(nn, sep="\t")
        p = d[d.node_type == "protein"]
        src = p.source
        is_pin, is_de = has(src, "pinnacle"), has(src, "de")
        rows.append({
            "disease": disease, "cell_type": cell_type, "cell_state": state,
            "PINNACLE": int(is_pin.sum()),
            "DE": int(is_de.sum()),
            "added_via_DE": int((is_de & ~is_pin).sum()),
            "total_proteins": int(len(p)),
        })
    df = pd.DataFrame(rows).sort_values(
        by=["disease", "cell_type", "cell_state"],
        key=lambda c: c.map(DIS_ORDER) if c.name == "disease" else c
    ).reset_index(drop=True)

    print(df.to_string(index=False))
    out_tsv = DE_PPI / "results" / out_name / "network_protein_counts.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}", flush=True)

    # ---- render as PNG table ----
    headers = ["Disease", "Cell type", "Cell state", "PINNACLE", "DE", "Added via DE", "Total proteins"]
    cell_text = df.values.tolist()
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(df) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    dz_color = {"Crohn": "#dbe9f6", "Alzheimer": "#fde6d0", "ILD": "#dcecdc"}
    n_cols = len(headers)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#404040"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(dz_color.get(df.iloc[r - 1]["disease"], "white"))
    ax.set_title(f"Network protein composition by disease / cell type / cell state ({out_name})",
                 fontweight="bold", pad=14)
    out_png = out_tsv.with_suffix(".png")
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-network protein-count table (PINNACLE vs DE)")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_embedding")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name))
