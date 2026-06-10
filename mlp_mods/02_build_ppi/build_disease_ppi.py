"""Build a disease PPI for a (cell-type, disease) and save its Wilcoxon ranks.

Generalized, PINNACLE-based stage 2. For a build named in builds_manifest.json
(`--build <name>`), for EACH arm (e.g. healthy / disease):

  1. load the arm's target cells (cell-type, optionally disease-filtered) + the
     shared body-wide reference; normalize.
  2. resample-Wilcoxon (NUM_CELLS_CUTOFF cells/cluster x ITERATIONS, matched
     seeds), target-vs-rest. Per trial, rank ALL genes by p-value. Record:
       - per-gene mean rank across trials  -> FULL ranks
       - per-gene top-MAX_N_GENES membership -> markers kept in >= KEEP_FRAC trials
  3. induce the arm's markers on the PINNACLE global PPI; take the LCC -> arm pool.

Then union the arm pools -> node_vocab.tsv (global_id + per-arm membership flags),
and write each arm's full ranks to rank_shifts/<build>/<arm>.tsv. Edges are not
stored; they are induced bidirectionally at encode time by stage 4. OmniPath is
NOT used -- nodes and (later) edges are both PINNACLE-derived.

This consolidates the former build_resampled_ppis.py + genome_wide_rank_shift.py
+ 05/build_networks.py (minus OmniPath).

Run with .venv (scanpy/anndata/networkx):
  .venv/bin/python mlp_mods/02_build_ppi/build_disease_ppi.py --build macrophage_crohn
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = HERE / "builds_manifest.json"
RANK_SHIFTS = REPO / "mlp_mods/rank_shifts"
REST_LABEL = "rest"


def load_global_ppi(path: Path) -> nx.Graph:
    g = nx.Graph()
    with path.open() as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 2 and p[0] != p[1]:
                g.add_edge(p[0], p[1])
    return g


def ranked_symbols(adata, target_label, sym_map, max_pval):
    """One Wilcoxon target-vs-rest; return symbols ordered by ascending p-value."""
    sc.tl.rank_genes_groups(adata, groupby="cell_type_label", method="wilcoxon",
                            groups=[target_label], reference=REST_LABEL, pts=False, use_raw=False)
    res = sc.get.rank_genes_groups_df(adata, group=target_label)
    res["gene_symbol"] = res["names"].map(sym_map)
    res = res.dropna(subset=["gene_symbol"])
    res = res.sort_values(["gene_symbol", "pvals_adj"]).drop_duplicates("gene_symbol", keep="first")
    res = res[res["pvals"] <= max_pval].sort_values("pvals", ascending=True)
    return res["gene_symbol"].tolist()


def lcc_nodes(markers, g_global):
    g = nx.Graph()
    for u, v in g_global.edges():
        if u in markers and v in markers:
            g.add_edge(u, v)
    if g.number_of_nodes() == 0:
        return set()
    return set(max(nx.connected_components(g), key=len))


def build_arm(arm, a_rest, g_global, k):
    """Return (mean_rank dict over all genes, marker LCC pool set)."""
    a_t = ad.read_h5ad(REPO / arm["h5ad"])
    if arm["disease_filter"]:
        a_t = a_t[a_t.obs["disease"].astype(str).values == arm["disease_filter"]].copy()
    a_t.obs["cell_type_label"] = arm["target_label"]; a_t.obs["subsample_cluster"] = arm["target_label"]
    rest = a_rest.copy(); rest.obs["cell_type_label"] = REST_LABEL
    rest.obs["subsample_cluster"] = rest.obs["cell_type"].astype(str).values
    comb = ad.concat([a_t, rest], axis=0, join="inner", merge="first")
    comb.obs["cell_type_label"] = comb.obs["cell_type_label"].astype("category")
    sc.pp.normalize_total(comb, target_sum=1e4); sc.pp.log1p(comb)
    sym_map = dict(zip(comb.var.index, comb.var["feature_name"].astype(str)))
    clusters = comb.obs["subsample_cluster"].astype(str)
    idx_by = {c: np.where(clusters.values == c)[0] for c in clusters.unique()}

    rsum, rcnt, mcnt = defaultdict(float), defaultdict(int), defaultdict(int)
    for it in range(k["iterations"]):
        rng = np.random.default_rng(k["seed"] + it)
        take = [rng.choice(idx, size=min(k["num_cells_cutoff"], len(idx)), replace=False)
                for idx in idx_by.values()]
        order = ranked_symbols(comb[np.sort(np.concatenate(take))].copy(),
                               arm["target_label"], sym_map, k["max_pval"])
        for pos, sym in enumerate(order):
            rsum[sym] += pos + 1; rcnt[sym] += 1
            if pos < k["max_n_genes"]:
                mcnt[sym] += 1
        print(f"  [{arm['name']}] trial {it+1}/{k['iterations']}", flush=True)

    mean_rank = {g: rsum[g] / rcnt[g] for g in rsum}
    keep_n = math.ceil(k["keep_frac"] * k["iterations"])
    markers = {g for g, c in mcnt.items() if c >= keep_n}
    pool = lcc_nodes(markers, g_global)
    print(f"  [{arm['name']}] markers={len(markers)}  PINNACLE-LCC pool={len(pool)}", flush=True)
    return mean_rank, pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="build name in builds_manifest.json")
    args = ap.parse_args()
    man = json.load(open(MANIFEST))
    k = man["knobs"]; bdef = man["builds"][args.build]
    g_global = load_global_ppi(REPO / man["global_ppi"])
    a_rest = ad.read_h5ad(REPO / man["reference_h5ad"])
    targets = set()
    if bdef.get("opentargets_positive"):
        d = json.load(open(REPO / bdef["opentargets_positive"]))
        for key in bdef["opentargets_key"]:
            d = d[key]
        targets = set(d)

    arms = bdef["arms"]
    mean_ranks, pools = {}, {}
    for arm in arms:
        mean_ranks[arm["name"]], pools[arm["name"]] = build_arm(arm, a_rest, g_global, k)

    # union node vocab over arm pools
    union = sorted(set().union(*pools.values()))
    gid = {p: i for i, p in enumerate(union)}
    vocab = pd.DataFrame({"protein": union, "global_id": [gid[p] for p in union]})
    for arm in arms:
        vocab[f"in_{arm['name']}"] = [int(p in pools[arm["name"]]) for p in union]
    poolset = set(union)
    out_dir = HERE / "builds" / args.build
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(out_dir / "node_vocab.tsv", sep="\t", index=False)

    # per-arm full ranks -> rank_shifts/<build>/<arm>.tsv
    rs = RANK_SHIFTS / args.build
    rs.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        mr = mean_ranks[arm["name"]]
        df = pd.DataFrame({"gene": list(mr), "mean_rank": [round(v, 1) for v in mr.values()]})
        df["in_pool"] = df["gene"].isin(poolset).astype(int)
        df["is_target"] = df["gene"].isin(targets).astype(int)
        df.sort_values("mean_rank").to_csv(rs / f"{arm['name']}.tsv", sep="\t", index=False)

    json.dump({"build": args.build, "cell_type": bdef["cell_type"], "disease": bdef["disease"],
               "arms": [a["name"] for a in arms], "n_nodes": len(union),
               "global_ppi": man["global_ppi"]},
              open(out_dir / "build_config.json", "w"), indent=2)
    print(f"\nwrote builds/{args.build}/node_vocab.tsv ({len(union)} nodes) and "
          f"rank_shifts/{args.build}/{{{','.join(a['name'] for a in arms)}}}.tsv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
