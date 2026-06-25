"""(disease, cell type) PPI influence with LITERATURE + DE-curated edge weights.

Parameterized by --build <name> (resolved via config.load_build from
02_build_ppi/builds_manifest.json). Outputs go to de_ppi/results/<build>/.

  - Node set = PINNACLE cell-type proteins  UNION  DE genes (padj<0.05)  UNION  the
    literature-search genes. OmniPath-orphans are dropped unless they reach a metabolite.
  - Sender (broadcast) weight, two SEPARATE tracks:
      * CellxGene DE genes (padj<0.05): paired reference/disease rank change
            w = min(exp(-(disease_rank - ref_rank)/tau), wmax)   (tau=4000, wmax=5)
      * literature genes (not DE): elevated -> 2.0, suppressed -> 0.5.
    PINNACLE backbone / non-DE / metabolite sinks broadcast at 1.0.
  - Target set = DE genes  UNION  literature markers  UNION  metabolite sinks
    (HMDB disease metabolites UNION literature-search metabolites).
  - influence(i) = sum_{k=0..3} (P^k @ m),  P[i,j] = w(i)/Z(j). The k=0 (self-loop)
    term credits a node for being itself in the dysregulated target set m.

Outputs (de_ppi/results/<build>/): P3_influence.tsv, networks/network_nodes.tsv,
networks/network_edges.tsv. Prints the OpenTargets-positive target table.

P3_influence.tsv is ranked by (unsigned) reach (`rank`). It also carries `general_influence`
(reach to all nodes = hubness), `specificity` (degree-matched permutation z-score), and the
signed `effect`/`effect_propagated` (signature-reversal via the signed operator P_sig).
CAVEAT: signed multi-hop propagation is fragile to incomplete sign coverage (a signless edge
zeroes a path); the build prints signed-edge coverage so this can be judged.

Run with .venv:
  .venv/bin/python mlp_mods/de_ppi/build_literature_weighted_influence.py --build macrophage_crohn
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

from config import load_build

HOPS = 3
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
        out_p3, out_nodes, out_edges = nd / "P3_influence.tsv", nd / "network_nodes.tsv", nd / "network_edges.tsv"
    else:
        cfg.networks_dir.mkdir(parents=True, exist_ok=True)   # only create canonical dir when not redirected
        out_p3, out_nodes, out_edges = cfg.p3_influence, cfg.network_nodes, cfg.network_edges

    # literature markers -> direction (majority vote if conflict). Skipped when --no-lit, or when
    # the build has no literature table (e.g. stem cells: no lit_search panel) -> DE-only.
    direction, lit_up, lit_down = {}, set(), set()
    if no_lit:
        print("no-lit: literature genes EXCLUDED (node set, target set m, sender weights, and "
              "Effect signature all use DE only)", flush=True)
    elif not cfg.lit_genes.exists():
        print(f"no literature gene table ({cfg.lit_genes.name} absent) -> DE-only "
              "node set / targets / sender weights", flush=True)
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
        print("lit-only: DE genes EXCLUDED from node set, targets, and sender weights", flush=True)
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
    if expressed_backbone:                             # REPLACE backbone with the state's expressed set
        exp0 = {g.strip() for g in expr_path.read_text().split() if g.strip()}
        print(f"expressed-backbone: backbone REPLACED by {len(exp0)} expressed proteins "
              f"(detect>=floor) from {expr_path.name}; PINNACLE dropped "
              f"(removes {len(ppi_nodes - exp0)} non-expressed, adds {len(exp0 - ppi_nodes)} expressed)", flush=True)
        ppi_nodes = set(exp0)
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
    if neutral_weights:                                            # drop the rank-shift MAGNITUDE from propagation
        w = np.full(N, W_BASE)                                     # edge/self weights = 1.0 (topology-only); dysregulated SET still defined via dir_w
        print("neutral-weights: edge/self weights set to 1.0 (no rank-shift magnitude); "
              "direction / dysregulated set preserved", flush=True)

    src = list(op.src.map(pidx)) + [pidx[s] for s, _ in pm]
    dst = list(op.dst.map(pidx)) + [midx[c] for _, c in pm]
    src, dst = np.array(src), np.array(dst)
    P = sp.coo_matrix((w[src], (src, dst)), shape=(N, N)).tocsc()
    Z = np.asarray(P.sum(0)).ravel()
    invZ = sp.diags(np.divide(1.0, Z, out=np.zeros_like(Z), where=Z > 0))
    P = (P @ invZ).tocsr()

    # target = DE-dysregulated  ∪  literature markers  ∪  metabolites
    target_genes = (dysreg | set(direction))
    m = np.zeros(N)
    for g, i in pidx.items():
        if g in target_genes: m[i] = 1.0
    for c in met: m[midx[c]] = 1.0
    print(f"target set: {int(m[:np_].sum())} gene targets (DE ∪ literature) + {nm} metabolites", flush=True)

    # influence(i) = sum_{k=0..3} (P^k @ m). The k=0 (self-loop / identity) term credits a
    # node for being itself in the dysregulated target set -- targeting a dysregulated node
    # directly perturbs the set even if it broadcasts little (e.g. ITGB7). Reach-only
    # influence is recoverable as (influence - is_target) for protein rows.
    infl = m.copy()                      # k=0 self-loop
    v = m.copy()
    for _ in range(HOPS):
        v = P @ v; infl += v

    # general_influence (reach to ALL nodes = hubness) + degree-matched permutation specificity
    gen = np.ones(N); vg = np.ones(N)
    for _ in range(HOPS):
        vg = P @ vg; gen += vg
    B = 1000
    indeg = np.bincount(dst, minlength=N).astype(float)    # incoming-edge count (target reachability)
    binid = pd.qcut(pd.Series(indeg).rank(method="first"), q=20, labels=False).to_numpy()
    pool = defaultdict(list)
    for j in range(N):
        pool[binid[j]].append(j)
    pool = {b: np.array(v) for b, v in pool.items()}
    m_idx = np.where(m > 0)[0]
    rng = np.random.default_rng(0)
    Mn = np.zeros((N, B))
    for b in range(B):
        for mi in m_idx:
            Mn[rng.choice(pool[binid[mi]]), b] = 1.0        # one random same-degree-bin node per m member
    acc = Mn.copy(); Vn = Mn.copy()
    for _ in range(HOPS):
        Vn = P @ Vn; acc += Vn
    mu, sd = acc.mean(1), acc.std(1)
    spec = np.divide(infl - mu, sd, out=np.zeros(N), where=sd > 0)

    # --- signed Effect (signature-reversal): Effect(p) = sum_{k=0..K} (P_sig^k @ sig) ---
    # P_sig = column-normalized operator carrying OmniPath activation/inhibition signs (MIND
    # metabolite edges neutral=0). sig = signed disease magnitude (DE: log2FC; literature:
    # +/- median|log2FC|). effect_propagated drops the k=0 self term. CAVEAT: signed multi-hop
    # propagation is fragile to incomplete sign coverage (a signless edge zeroes a path).
    sign = np.array(list(op.sign.astype(float)) + [0.0] * len(pm))
    P_sig = (sp.coo_matrix((sign * w[src], (src, dst)), shape=(N, N)).tocsc() @ invZ).tocsr()
    de_lfc = (de.loc[[g for g in dysreg if g in de.index], "log2FoldChange"]
              if (not lit_only and len(dysreg)) else pd.Series(dtype=float))
    mbar = float(np.nanmedian(np.abs(de_lfc))) if len(de_lfc) else 1.0
    sigv = np.zeros(N)
    for g, i in pidx.items():
        if (not lit_only) and g in dysreg and np.isfinite(de.loc[g, "log2FoldChange"]):
            sigv[i] = float(de.loc[g, "log2FoldChange"])
        elif g in lit_up:
            sigv[i] = mbar
        elif g in lit_down:
            sigv[i] = -mbar
    effect = sigv.copy(); ve = sigv.copy()
    for _ in range(HOPS):
        ve = P_sig @ ve; effect += ve
    effect_prop = effect - sigv
    fs = float((sign != 0).sum()) / len(sign) if len(sign) else 0.0
    print(f"signed-edge coverage: {fs:.1%} of {len(sign)} edges", flush=True)

    def wlabel(g):
        return "up(2.0)" if g in lit_up else "down(0.5)" if g in lit_down else "1.0"
    rows = [{"protein": g, "influence": float(infl[i]), "general_influence": float(gen[i]),
             "specificity": round(float(spec[i]), 3),
             "sender_weight": w[i], "lit_marker": wlabel(g), "is_target": int(g in target_genes)}
            for g, i in pidx.items()]
    df = pd.DataFrame(rows).sort_values("influence", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1                                   # ranking on (unsigned) reach
    df["specificity_rank"] = df.specificity.rank(ascending=False).astype(int)
    df["effect"] = df.protein.map({g: round(float(effect[i]), 4) for g, i in pidx.items()})
    df["effect_propagated"] = df.protein.map({g: round(float(effect_prop[i]), 4) for g, i in pidx.items()})
    df["effect_rank"] = df.effect.rank(ascending=False).astype(int)
    df.to_csv(out_p3, sep="\t", index=False)

    # OpenTargets-positive target table (only if this build has an OpenTargets pull)
    print(f"\nN={N} | wrote {out_p3}", flush=True)
    if cfg.opentargets_positive and cfg.opentargets_positive.exists() and cfg.ot_efo:
        targets = set(json.load(open(cfg.opentargets_positive))[cfg.ot_efo][cfg.ot_celltype])
        t = df[df.protein.isin(targets)].copy()
        t["pctile"] = (t["rank"] / N * 100).round(1)
        t = t[["rank", "pctile", "protein", "influence", "sender_weight", "lit_marker"]].sort_values("rank")
        print(f"\n=== {cfg.disease} targets ranked by influence ===")
        print(t.round(3).to_string(index=False))
        print(f"\nbest #{int(t['rank'].min())} | median #{int(t['rank'].median())} (top {100*t['rank'].median()/N:.0f}%)")
    else:
        print("(OpenTargets-positive target table skipped — no OpenTargets pull for this build)", flush=True)

    # ---- network manifests: all entities + the edge list ----
    mac_nodes = ppi_nodes                              # PINNACLE-context nodes (empty if no backbone)

    def sender_attr(node):
        """Edge weight is sender-gated from the node's w (only DE genes are != 1.0; all 1.0 under
        --neutral-weights). Direction is read off dir_w (the rank-shift sign), which is preserved
        even when propagation weights are neutralized, so the dysregulated set stays defined.
        Metabolites are sinks (no pidx entry) -> 1.0."""
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
    ap = argparse.ArgumentParser(description="literature+DE-weighted (disease, cell type) PPI influence")
    ap.add_argument("--build", default="macrophage_crohn")
    ap.add_argument("--lit-only", action="store_true",
                    help="use ONLY literature genes/metabolites as the dysregulated set (exclude DE)")
    ap.add_argument("--rank-weight-all", action="store_true",
                    help="apply the rank-change sender weight to ALL genes with finite ranks, "
                         "not just DE-significant ones")
    ap.add_argument("--no-lit", action="store_true",
                    help="EXCLUDE literature genes from the node set, target set, weights, and "
                         "Effect signature (use DE only)")
    ap.add_argument("--expressed", action="store_true",
                    help="UNION state-expressed proteins (detect>=floor, ambient-blacklisted) into node set")
    ap.add_argument("--indra", action="store_true",
                    help="merge high-confidence (belief>=0.5) INDRA directed causal edges with OmniPath")
    ap.add_argument("--expressed-backbone", action="store_true",
                    help="REPLACE the PINNACLE backbone with the state's expressed set "
                         "(detect>=floor): removes non-expressed backbone proteins, adds expressed ones")
    ap.add_argument("--net-out", default=None,
                    help="write P3_influence/network_nodes/network_edges to this dir instead of "
                         "results/<build>/networks (does not clobber the canonical per-build outputs)")
    ap.add_argument("--expressed-genes", default=None,
                    help="path to the expressed-gene list to use (overrides expressed_genes/<build>.txt)")
    ap.add_argument("--neutral-weights", action="store_true",
                    help="set all sender/edge weights to 1.0 (drop the DE rank-shift): topology-only network")
    a = ap.parse_args()
    raise SystemExit(main(a.build, a.lit_only, a.rank_weight_all, a.no_lit, a.indra, a.expressed,
                          a.expressed_backbone, a.net_out, a.expressed_genes, a.neutral_weights))
