"""(disease, cell type) PPI network assembly with LITERATURE + DE-curated edge weights.

Parameterized by --build <name> (resolved via config.load_build from
02_build_ppi/builds_manifest.json). Outputs go to de_ppi/results/<build>/networks/.

  - Node set = PINNACLE cell-type proteins  UNION  DE genes (padj<0.05)  UNION  the
    literature-search genes. OmniPath-orphans are dropped unless they reach a metabolite.
  - Sender (broadcast) weight, two SEPARATE tracks:
      * CellxGene DE genes (padj<0.05): paired reference/disease rank change
            w = min(exp(-(disease_rank - ref_rank)/tau), wmax)   (tau=4000, wmax=5)
      * literature genes (not DE): elevated -> 2.0, suppressed -> 0.5.
    PINNACLE backbone / non-DE / metabolite sinks broadcast at 1.0.
  - Metabolite sinks = HMDB disease metabolites UNION literature-search metabolites,
    wired in via MIND protein->metabolite edges.

Outputs (de_ppi/results/<build>/networks/, or --net-out): network_nodes.tsv,
network_edges.tsv. These manifests are consumed by the embedding/influence code
(embed/joint_embed.py, etc.).

Run with .venv:
  .venv/bin/python mlp_mods/de_ppi/build_ppi_network.py --build macrophage_crohn
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse, csv
from pathlib import Path
import numpy as np, pandas as pd

from config import load_build

W_UP, W_DOWN, W_BASE = 2.0, 0.5, 1.0
TAU, WMAX = 4000.0, 5.0


def main(build: str, lit_only: bool = False, rank_weight_all: bool = False,
         no_lit: bool = False, indra: bool = False, expressed: bool = False,
         expressed_backbone: bool = False, net_out: str | None = None,
         expr_genes_path: str | None = None, neutral_weights: bool = False) -> int:
    cfg = load_build(build)
    expr_path = Path(expr_genes_path) if expr_genes_path else cfg.expressed_genes
    if net_out:                                        # redirect all outputs to a custom dir
        nd = Path(net_out); nd.mkdir(parents=True, exist_ok=True)
        out_nodes, out_edges = nd / "network_nodes.tsv", nd / "network_edges.tsv"
    else:
        cfg.networks_dir.mkdir(parents=True, exist_ok=True)   # only create canonical dir when not redirected
        out_nodes, out_edges = cfg.network_nodes, cfg.network_edges

    # literature markers -> direction (majority vote if conflict). Skipped when --no-lit, or when
    # the build has no literature table (e.g. stem cells: no lit_search panel) -> DE-only.
    direction, lit_up, lit_down = {}, set(), set()
    if no_lit:
        print("no-lit: literature genes EXCLUDED (node set, sender weights all use DE only)", flush=True)
    elif not cfg.lit_genes.exists():
        print(f"no literature gene table ({cfg.lit_genes.name} absent) -> DE-only "
              "node set / sender weights", flush=True)
    else:
        lit = pd.read_csv(cfg.lit_genes, sep="\t")
        for g, sub in lit.groupby("entity"):
            up = (sub.direction == "elevated").sum(); dn = (sub.direction == "suppressed").sum()
            direction[g] = "up" if up >= dn else "down"
        lit_up = {g for g, d in direction.items() if d == "up"}
        lit_down = {g for g, d in direction.items() if d == "down"}
        print(f"literature markers: {len(direction)} ({len(lit_up)} up, {len(lit_down)} down)", flush=True)

    if lit_only:                                       # e.g. GBM: DE is batch-confounded -> exclude it
        de, dysreg = None, set()
        print("lit-only: DE genes EXCLUDED from node set and sender weights", flush=True)
    else:
        de = pd.read_csv(cfg.de_table, sep="\t").set_index("gene")
        dysreg = set(de[de.padj < 0.05].index)
    if cfg.celltype_ppi.exists():                      # cell-type PINNACLE backbone (optional)
        mac = pd.read_csv(cfg.celltype_ppi, sep=" ", header=None, names=["a", "b"])
        ppi_nodes = set(mac.a) | set(mac.b)
    else:                                              # no PINNACLE context (e.g. cortical neuron):
        ppi_nodes = set()                              # backbone comes from DE + --expressed proteins
        print(f"no cell-type PPI ({cfg.celltype_ppi.name} absent) -> backbone = DE "
              "(+ --expressed) proteins, wired by OmniPath", flush=True)
    pinnacle_nodes = set(ppi_nodes)                    # true PINNACLE membership for the 'pinnacle' source tag
    if expressed_backbone:                             # REPLACE backbone with the state's expressed set
        exp0 = {g.strip() for g in expr_path.read_text().split() if g.strip()}
        print(f"expressed-backbone: backbone REPLACED by {len(exp0)} expressed proteins "
              f"(detect>=floor) from {expr_path.name}; PINNACLE dropped "
              f"(removes {len(ppi_nodes - exp0)} non-expressed, adds {len(exp0 - ppi_nodes)} expressed)", flush=True)
        ppi_nodes = set(exp0)
        pinnacle_nodes = set()                         # PINNACLE dropped -> no node carries the 'pinnacle' tag
        expressed = True                               # also tag 'expressed' + union below (no-op)
    node_set = ppi_nodes | dysreg | set(direction)     # + literature markers
    exp: set = set()
    if expressed:                                      # + state-expressed proteins (detect>=floor, ambient-blacklisted)
        exp = {g.strip() for g in expr_path.read_text().split() if g.strip()}
        new = exp - node_set
        node_set |= exp
        print(f"expressed: +{len(new)} expressed proteins unioned into node set "
              f"({len(exp)} expressed, {len(new)} new) from {expr_path.name}", flush=True)

    op = pd.read_csv(cfg.omni, sep="\t")
    if indra:                                          # supplement with high-confidence INDRA causal edges
        ipath = cfg.omni.parent / "indra_directed_edges.tsv"
        iz = pd.read_csv(ipath, sep="\t")
        iz = iz[iz.belief >= 0.5][["src", "dst", "sign"]].copy(); iz["layer"] = "indra"
        op = pd.concat([op, iz], ignore_index=True)    # OmniPath rows first -> win sign on dedup below
        print(f"indra: +{len(iz)} INDRA directed edges (belief>=0.5) merged with OmniPath", flush=True)
    op = op[op.src.isin(node_set) & op.dst.isin(node_set) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    omni_incident = node_set & (set(op.src) | set(op.dst))

    # metabolite node set + MIND protein->metabolite edges, scanned over the WHOLE node
    # set so OmniPath-orphans that still reach a metabolite are kept (other orphans drop).
    def _norm_chebi(c):
        c = str(c).strip()
        return c if c.upper().startswith("CHEBI") else f"CHEBI:{c}"
    if cfg.metabolite_chebi.exists():                  # HMDB disease metabolites (optional)
        chebi = pd.read_csv(cfg.metabolite_chebi, sep="\t", dtype=str)
        hmdb_chebi = {_norm_chebi(c) for c in chebi.chebi if str(c).strip()}
    else:
        hmdb_chebi = set()
    if cfg.lit_metabolites.exists():                   # stage-L metabolite sinks (optional)
        lit_met = pd.read_csv(cfg.lit_metabolites, sep="\t", dtype=str)
        lit_chebi = {_norm_chebi(c) for c in lit_met.chebi_id if str(c).strip()}
    else:
        lit_chebi = set()
    chebi_nodes = hmdb_chebi | lit_chebi
    nodes = pd.read_csv(cfg.nodes, dtype=str, keep_default_na=False)
    gmm = nodes[nodes.label == "MacromolecularMachine"]; sym2id = dict(zip(gmm.name, gmm.id)); id2sym = {v: k for k, v in sym2id.items()}
    node_ids = {sym2id[g]: g for g in node_set if g in sym2id}          # protein id -> symbol, over node_set
    pm = []
    pm_rel = {}                                          # (protein_sym, chebi) -> {MIND relation, ...}
    with open(cfg.edges) as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        for h, k, t in r:
            if h in node_ids and t in chebi_nodes:
                s = node_ids[h]
                pm.append((s, t))
                pm_rel.setdefault((s, t), set()).add(k)
    metab_connected = {s for s, _ in pm}

    # protein nodes: OmniPath-incident  UNION  any node reaching a metabolite via MIND;
    # all other orphans are dropped.
    prot = sorted(omni_incident | metab_connected)
    pidx = {g: i for i, g in enumerate(prot)}; np_ = len(prot)
    kept_orphans = sorted(metab_connected - omni_incident)
    print(f"protein nodes: {np_}  (OmniPath-orphans kept for a metabolite edge: "
          f"{len(kept_orphans)} -> {kept_orphans})", flush=True)
    print(f"metabolite candidates: HMDB {len(hmdb_chebi)} + lit-search {len(lit_chebi)} "
          f"-> union {len(chebi_nodes)}", flush=True)
    met = sorted({c for _, c in pm}); midx = {c: np_ + i for i, c in enumerate(met)}
    nm = len(met); N = np_ + nm
    print(f"metabolite sink nodes: {nm} | protein->metabolite edges: {len(pm)}", flush=True)

    # sender weights, two separate tracks: rank-change gate (DE-significant genes by default,
    # or ANY gene with finite ranks when --rank-weight-all); literature genes get 2.0/0.5; rest 1.0.
    rh, rc = ((de[cfg.ref_rank_col].to_dict(), de[cfg.disease_rank_col].to_dict())
              if not lit_only else ({}, {}))
    w = np.full(N, W_BASE)
    for g, i in pidx.items():
        has_rank = g in rh and g in rc and np.isfinite(rh[g]) and np.isfinite(rc[g])
        if has_rank and (rank_weight_all or g in dysreg):          # rank-change weight
            w[i] = min(np.exp(-(rc[g] - rh[g]) / TAU), WMAX)       # all ranked genes if rank_weight_all
        elif g in lit_up:
            w[i] = W_UP                                            # literature elevated -> 2.0
        elif g in lit_down:
            w[i] = W_DOWN                                          # literature suppressed -> 0.5
    dir_w = w.copy()                                               # direction sign is read off this (preserved even if weights neutralized)
    if neutral_weights:                                           # drop the rank-shift MAGNITUDE from edges
        w = np.full(N, W_BASE)                                    # edge/self weights = 1.0 (topology-only)
        print("neutral-weights: edge/self weights set to 1.0 (no rank-shift magnitude); "
              "direction preserved", flush=True)

    # ---- network manifests: all entities + the edge list ----
    mac_nodes = pinnacle_nodes                         # true PINNACLE-context nodes (empty under --expressed-backbone)

    def sender_attr(node):
        """Edge weight is sender-gated from the node's w (only DE genes are != 1.0; all 1.0 under
        --neutral-weights). Direction is read off dir_w (the rank-shift sign), which is preserved
        even when edge weights are neutralized. Metabolites are sinks (no pidx entry) -> 1.0."""
        i = pidx.get(node)
        wt = W_BASE if i is None else float(w[i])
        dw = W_BASE if i is None else float(dir_w[i])
        ddir = "elevated" if dw > 1.0 else "suppressed" if dw < 1.0 else ""
        return wt, ddir

    node_rows = []
    for g in prot:
        srcs = [s for s, ok in (("pinnacle", g in mac_nodes), ("de", g in dysreg),
                                ("literature_search", g in direction), ("expressed", g in exp)) if ok]
        wt, ddir = sender_attr(g)
        node_rows.append({"node_id": g, "node_type": "protein", "source": "|".join(srcs),
                          "direction": ddir, "sender_weight": wt})
    for c in met:
        srcs = [s for s, ok in (("hmdb", c in hmdb_chebi),
                                ("literature_search", c in lit_chebi)) if ok]
        node_rows.append({"node_id": c, "node_type": "metabolite", "source": "|".join(srcs),
                          "direction": "", "sender_weight": w[midx[c]]})
    pd.DataFrame(node_rows, columns=["node_id", "node_type", "source", "direction",
                                     "sender_weight"]).to_csv(out_nodes, sep="\t", index=False)

    edge_rows = []
    for (s, c), rels in pm_rel.items():                    # MIND protein->metabolite (sorted first)
        wt, ddir = sender_attr(s)
        edge_rows.append({"source": s, "target": c, "edge_origin": "MIND",
                          "edge_property": ",".join(sorted(rels)), "weight": wt, "direction": ddir})
    for s, d in zip(op.src, op.dst):                       # OmniPath protein->protein (no MIND property)
        wt, ddir = sender_attr(s)
        edge_rows.append({"source": s, "target": d, "edge_origin": "OmniPath",
                          "edge_property": "", "weight": wt, "direction": ddir})
    pd.DataFrame(edge_rows, columns=["source", "target", "edge_origin", "edge_property",
                                     "weight", "direction"]).to_csv(out_edges, sep="\t", index=False)
    print(f"wrote {out_nodes} ({len(node_rows)} nodes) and "
          f"{out_edges.name} ({len(edge_rows)} edges)", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="literature+DE-weighted (disease, cell type) PPI network assembly")
    ap.add_argument("--build", default="macrophage_crohn")
    ap.add_argument("--lit-only", action="store_true",
                    help="use ONLY literature genes/metabolites (exclude DE from node set + weights)")
    ap.add_argument("--rank-weight-all", action="store_true",
                    help="apply the rank-change sender weight to ALL genes with finite ranks, "
                         "not just DE-significant ones")
    ap.add_argument("--no-lit", action="store_true",
                    help="EXCLUDE literature genes from the node set and weights (use DE only)")
    ap.add_argument("--expressed", action="store_true",
                    help="UNION state-expressed proteins (detect>=floor, ambient-blacklisted) into node set")
    ap.add_argument("--indra", action="store_true",
                    help="merge high-confidence (belief>=0.5) INDRA directed causal edges with OmniPath")
    ap.add_argument("--expressed-backbone", action="store_true",
                    help="REPLACE the PINNACLE backbone with the state's expressed set "
                         "(detect>=floor): removes non-expressed backbone proteins, adds expressed ones")
    ap.add_argument("--net-out", default=None,
                    help="write network_nodes/network_edges to this dir instead of "
                         "results/<build>/networks (does not clobber the canonical per-build outputs)")
    ap.add_argument("--expressed-genes", default=None,
                    help="path to the expressed-gene list to use (overrides expressed_genes/<build>.txt)")
    ap.add_argument("--neutral-weights", action="store_true",
                    help="set all sender/edge weights to 1.0 (drop the DE rank-shift): topology-only network")
    a = ap.parse_args()
    raise SystemExit(main(a.build, a.lit_only, a.rank_weight_all, a.no_lit, a.indra, a.expressed,
                          a.expressed_backbone, a.net_out, a.expressed_genes, a.neutral_weights))
