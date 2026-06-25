"""Assign every protein a single coarse molecular-function class from GO.

We have the GO ontology locally (mlp_mods/go_enrichment/go-basic.obo) but no gene->GO
annotation, so this script downloads/caches the human GO Annotation file (goa_human.gaf)
and maps each gene SYMBOL to its Molecular Function (aspect F) GO terms. Each protein's MF
terms are propagated up the is_a/part_of hierarchy, then the protein is assigned the FIRST
class in a priority-ordered slim it belongs to. Priority puts informative classes
(transcription regulator, kinase, receptor, channel, transporter) ahead of generic ones
(catalytic, binding), so one legible label per protein.

Output (cached, reusable across embeddings):
  mlp_mods/de_ppi/protein_function.tsv   columns: symbol, func_class

Run:
  .venv/bin/python mlp_mods/de_ppi/influence_analysis/annotate_protein_function.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import gzip
import urllib.request
from collections import defaultdict
from pathlib import Path

DE_PPI = Path("mlp_mods/de_ppi")
GO_DIR = DE_PPI.parent / "go_enrichment"
OBO = GO_DIR / "go-basic.obo"
GAF = GO_DIR / "goa_human.gaf.gz"
GAF_URL = "http://current.geneontology.org/annotations/goa_human.gaf.gz"
OUT = DE_PPI / "protein_function.tsv"

# priority-ordered Molecular Function slim: (label, anchor GO id). First match wins.
ANCHORS = [
    ("transcription_regulator", "GO:0140110"),  # transcription regulator activity
    ("protein_kinase",          "GO:0004672"),  # protein kinase activity
    ("phosphatase",             "GO:0016791"),  # phosphatase activity
    ("signaling_receptor",      "GO:0038023"),  # signaling receptor activity
    ("ion_channel",             "GO:0005216"),  # ion channel activity
    ("transporter",             "GO:0005215"),  # transporter activity
    ("protease",                "GO:0008233"),  # peptidase activity
    ("GTPase_or_switch",        "GO:0003924"),  # GTPase activity
    ("enzyme_regulator",        "GO:0030234"),  # enzyme regulator activity
    ("oxidoreductase",          "GO:0016491"),  # oxidoreductase activity
    ("cytoskeletal_structural", "GO:0005198"),  # structural molecule activity
    ("molecular_adaptor",       "GO:0060090"),  # molecular adaptor activity
    ("nucleic_acid_binding",    "GO:0003676"),  # nucleic acid binding
    ("catalytic_other",         "GO:0003824"),  # catalytic activity
    ("binding_other",           "GO:0005488"),  # binding
]


def parse_obo_parents(path: Path) -> dict[str, set[str]]:
    """GO id -> direct parents (is_a + part_of). Only Molecular Function not enforced; closure handles it."""
    parents: dict[str, set[str]] = defaultdict(set)
    cur = None
    obsolete = False
    for line in path.read_text().splitlines():
        if line == "[Term]":
            cur, obsolete = None, False
        elif line.startswith("id: GO:"):
            cur = line[4:].strip()
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif cur and not obsolete and line.startswith("is_a: GO:"):
            parents[cur].add(line.split()[1])
        elif cur and not obsolete and line.startswith("relationship: part_of GO:"):
            parents[cur].add(line.split()[2])
    return parents


def ancestors(go: str, parents: dict[str, set[str]], cache: dict[str, set[str]]) -> set[str]:
    if go in cache:
        return cache[go]
    out = {go}
    for p in parents.get(go, ()):
        out |= ancestors(p, parents, cache)
    cache[go] = out
    return out


def main() -> int:
    GO_DIR.mkdir(parents=True, exist_ok=True)
    if not GAF.exists():
        print(f"downloading {GAF_URL} ...", flush=True)
        req = urllib.request.Request(GAF_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(GAF, "wb") as f:
            f.write(r.read())
    print(f"GAF: {GAF} ({GAF.stat().st_size/1e6:.1f} MB)", flush=True)

    parents = parse_obo_parents(OBO)
    cache: dict[str, set[str]] = {}

    # symbol -> set of MF GO ids (aspect F)
    sym_mf: dict[str, set[str]] = defaultdict(set)
    with gzip.open(GAF, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[8] != "F":
                continue
            qual = c[3]
            if qual.startswith("NOT"):
                continue
            sym_mf[c[2]].add(c[4])
    print(f"{len(sym_mf)} symbols with MF annotations", flush=True)

    anchor_ids = [a for _, a in ANCHORS]
    rows = []
    counts: dict[str, int] = defaultdict(int)
    for sym, gos in sym_mf.items():
        closure: set[str] = set()
        for g in gos:
            closure |= ancestors(g, parents, cache)
        label = "other_unannotated"
        for lab, aid in ANCHORS:
            if aid in closure:
                label = lab
                break
        rows.append((sym, label))
        counts[label] += 1

    rows.sort()
    with open(OUT, "w") as f:
        f.write("symbol\tfunc_class\n")
        for sym, lab in rows:
            f.write(f"{sym}\t{lab}\n")
    print(f"wrote {OUT} ({len(rows)} symbols)", flush=True)
    print("class distribution (all annotated human symbols):")
    for lab, _ in ANCHORS + [("other_unannotated", "")]:
        if counts.get(lab):
            print(f"  {lab:26s} {counts[lab]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
