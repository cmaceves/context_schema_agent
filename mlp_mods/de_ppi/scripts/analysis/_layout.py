"""Shared network-tag ordering for all heatmaps: block by CELL TYPE, then within a cell type by
TISSUE, then keep each state as its own row/col. Importing this in every heatmap script guarantees
identical row/col order and block separators across figures.

Sort key = (cell-type rank, tissue rank, tag). Tissues are ordered so the two gut tissues (ileum=Crohn,
colon=UC) are adjacent, then lung (ILD), then brain (Alzheimer).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import re

_SUFFIX = re.compile(r"^(rep|loo|study|s)\d*$")   # replicate / leave-one-out / single-study tag suffix

CT_ORDER = {"macrophage": 0, "microglia": 1, "fibroblast": 2, "stem": 3}
# tissue order: the two gut tissues (ileum, colon) kept adjacent, then lung, brain.
TISSUE_RANK = {"ileum": 0, "colon": 1, "lung": 2, "brain": 3, "other": 9}
DISEASE_TISSUE = {"crohn": "ileum", "uc": "colon", "ild": "lung", "alz": "brain"}


def tag_celltype(t: str) -> str:
    for key, ct in (("mac", "macrophage"), ("microglia", "microglia"),
                    ("fibroblast", "fibroblast"), ("stem", "stem")):
        if key in t:
            return ct
    return "other"


def tag_disease(t: str) -> str:
    p = t.split("_")
    if p[0] == "healthy":                       # healthy_<disease>_<ct>, or healthy_macrophage_split* (Crohn normal arm)
        return p[1] if len(p) > 1 and p[1] in DISEASE_TISSUE else "crohn"
    return p[0] if p[0] in DISEASE_TISSUE else "crohn"


CT_TOKENS = ("mac", "macrophage", "microglia", "fibroblast", "stem")


def tag_state(t: str) -> str:
    """Leiden state label = tag minus its leading disease[/colon]/cell-type tokens.
    e.g. crohn_colon_mac_resident -> resident ; ild_macrophage_monocyte_derived -> monocyte_derived."""
    p = t.split("_")
    if p and p[0] == "healthy":
        p = p[1:]
    if p and p[0] in DISEASE_TISSUE:
        p = p[1:]
    if p and p[0] == "colon":                   # explicit tissue token (Crohn-colon build)
        p = p[1:]
    if p and p[0] in CT_TOKENS:
        p = p[1:]
    if p and _SUFFIX.match(p[-1]):              # replicate / leave-one-out / single-study suffix; same state
        p = p[:-1]
    return "_".join(p)


def tag_tissue(t: str) -> str:
    if "colon" in t:                            # Crohn-COLON build (else Crohn defaults to ileum)
        return "colon"
    return DISEASE_TISSUE.get(tag_disease(t), "other")


def order_tags(tags):
    return sorted(tags, key=lambda t: (CT_ORDER.get(tag_celltype(t), 9),
                                       TISSUE_RANK[tag_tissue(t)], t))


def block_separators(tags_o):
    """Return (cell_type_boundaries, tissue_boundaries) as column indices where a new block starts.
    Tissue boundaries exclude positions that are already cell-type boundaries."""
    cts = [tag_celltype(t) for t in tags_o]
    tis = [tag_tissue(t) for t in tags_o]
    ct_b = [k for k in range(1, len(tags_o)) if cts[k] != cts[k - 1]]
    ti_b = [k for k in range(1, len(tags_o)) if tis[k] != tis[k - 1] and k not in ct_b]
    return ct_b, ti_b


def _spans(tags_o, key):
    labs = [key(t) for t in tags_o]
    spans, s = [], 0
    for k in range(1, len(labs) + 1):
        if k == len(labs) or labs[k] != labs[k - 1]:
            spans.append((labs[s], s, k - 1)); s = k
    return spans


def annotate_hierarchy(ax, tags_o, also_left=False):
    """Draw labeled hierarchy brackets ABOVE a square heatmap: outer = CELL TYPE (bold), inner = TISSUE.
    (Matrix is symmetric, so columns label rows too.) With also_left=True, repeat on the y-axis, left of
    the y tick labels. Call after the matrix is drawn; then set the title with pad>=58 to clear the top
    brackets."""
    xt = ax.get_xaxis_transform()                                 # x in data coords, y in axes fraction
    for key, yf, fs, fw, col in [(tag_tissue, 1.012, 7, "normal", "0.35"),
                                 (tag_celltype, 1.06, 9, "bold", "black")]:
        for lab, s, e in _spans(tags_o, key):
            ax.plot([s - 0.45, e + 0.45], [yf, yf], transform=xt, color=col, lw=1.2, clip_on=False)
            ax.text((s + e) / 2, yf + 0.006, lab, transform=xt, ha="center", va="bottom",
                    fontsize=fs, fontweight=fw, color=col, clip_on=False)
    if also_left:
        yt = ax.get_yaxis_transform()                             # x in axes fraction, y in data coords
        for key, xf, fs, fw, col in [(tag_tissue, -0.27, 7, "normal", "0.35"),
                                     (tag_celltype, -0.34, 9, "bold", "black")]:
            for lab, s, e in _spans(tags_o, key):
                ax.plot([xf, xf], [s - 0.45, e + 0.45], transform=yt, color=col, lw=1.2, clip_on=False)
                ax.text(xf - 0.008, (s + e) / 2, lab, transform=yt, ha="center", va="center", rotation=90,
                        fontsize=fs, fontweight=fw, color=col, clip_on=False)
