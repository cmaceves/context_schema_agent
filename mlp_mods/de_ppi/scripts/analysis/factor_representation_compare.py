"""Attribution control: how much do disease / tissue / cell-type / cell-state differences show up in THREE
representations of a network — (1) RAW expression feature, (2) the same feature after plain GRAPH propagation
(no learning), (3) the LEARNED embedding. Lets us separate "does the graph add over raw" from "does the
learned encoder add over plain propagation" — for every factor, not just disease.

For each control network we form three per-protein representations over the protein universe:
  raw[net][p]   = the node expression feature (corrected pseudobulk)            -> scalar per protein
  graph[net][p] = 2 hops of (I+A) message passing on raw, A = row-normalized OmniPath operator (NO learning)
  emb[net][p]   = the learned encoder Z (from control_embeddings.npz)            -> 64-d per protein
(A uses the same self-loops/normalization the encoder uses; "graph" is the encoder with its learned linear
maps replaced by identity, i.e. pure propagation.)

For each FACTOR we pair control nets that differ in exactly that factor (others held), and measure the mean
per-protein distance between the pair in each representation (|Δ| for raw/graph, ||Δ||_2 for emb), over
proteins present in both. We also compute the between-study (same context, different study) distance = the
BATCH FLOOR, and report each factor's distance as a multiple of that floor SEPARATELY PER REPRESENTATION
(so the three are comparable despite different units). If the factor pattern is the same across raw/graph/emb,
the graph and the learning don't change the factor structure; if emb inflates/deflates a factor vs raw, the
encoder is reweighting it.

Output (results/<main>/controls/):
  factor_representation_pairs.tsv    one row per (factor, representation) pair distance
  factor_representation_summary.tsv  per factor x representation: n_pairs, mean_dist, dist_over_batch_floor

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/factor_representation_compare.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "mlp_mods/de_ppi/scripts/embed")
from embedding_utils import build_operator, LAYERS                       # same operator the encoder uses

DISEASES = {"crohn", "uc", "alz", "ild"}
MIN_OVERLAP = 20


def parse(tag: str) -> dict:
    split = None
    for h in ("A", "B"):
        if tag.endswith(f"_split{h}"):
            split, tag = h, tag[: -len(f"_split{h}")]
            break
    p = tag.split("_")
    return dict(arm=p[0], study=p[1], tissue=p[2], ct=p[3], state="_".join(p[4:]),
                split=split, loo=p[1].startswith(("loopool", "loosingle")))


def main(main_name: str, hops: int) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    nets_dir = res / "controls" / "networks"
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres, tags = c["Z"][:, pi, :], c["present"][:, pi], list(c["tags"])
    nid = np.asarray(c["node_id"])[pi]
    idx = {g: i for i, g in enumerate(nid)}
    P = {t: parse(t) for t in tags}
    use = [t for t in tags if not P[t]["loo"] and P[t]["split"] is None]     # primary per-state + allstates nets

    # ---- raw expression vector per net (corrected pseudobulk node feature) ----
    raw = {}
    for t in use:
        d = pd.read_csv(nets_dir / t / "network_nodes.tsv", sep="\t", keep_default_na=False)
        s = pd.Series(d["expression"].astype(float).values, index=d["node_id"].values)
        raw[t] = s.reindex(nid).fillna(0.0).values

    # ---- graph operator A per cell type (edges are identical across a cell type's control nets) ----
    A_ct = {}
    for t in use:
        ct = P[t]["ct"]
        if ct in A_ct:
            continue
        e = pd.read_csv(nets_dir / t / "network_edges.tsv", sep="\t", keep_default_na=False)
        e = e[e["source"].isin(idx) & e["target"].isin(idx)]
        A_ct[ct] = build_operator(e, idx, torch.device("cpu"), self_loops=True)

    def propagate(t):                                                       # plain (I+A)^hops @ raw -- no learning
        h = torch.tensor(raw[t], dtype=torch.float32)
        A = A_ct[P[t]["ct"]]
        for _ in range(hops):
            h = torch.sparse.mm(A, h.unsqueeze(1)).squeeze(1) + h
        return h.numpy()
    graph = {t: propagate(t) for t in use}
    emb = {t: Z[tags.index(t)] for t in use}
    presmask = {t: pres[tags.index(t)] for t in use}

    def dist(a, b):
        m = presmask[a] & presmask[b]
        if m.sum() < MIN_OVERLAP:
            return None
        return {"raw": float(np.abs(raw[a][m] - raw[b][m]).mean()),
                "graph": float(np.abs(graph[a][m] - graph[b][m]).mean()),
                "emb": float(np.linalg.norm(emb[a][m] - emb[b][m], axis=1).mean()),
                "n": int(m.sum())}

    rows = []

    def add(factor, groupkey, vary, pool):
        groups = defaultdict(list)
        for t in pool:
            groups[groupkey(P[t])].append(t)
        for ts in groups.values():
            for ta, tb in itertools.combinations(sorted(ts), 2):
                if P[ta][vary] == P[tb][vary]:
                    continue
                d = dist(ta, tb)
                if d is None:
                    continue
                rows.append(dict(factor=factor, network_a=ta, network_b=tb, n_overlap=d["n"],
                                 raw=d["raw"], graph=d["graph"], emb=d["emb"]))

    perstate = [t for t in use if P[t]["state"] != "allstates"]
    allstates = [t for t in use if P[t]["state"] == "allstates"]
    # batch floor: same (arm,tissue,ct,state), different study
    add("batch_floor", lambda q: (q["arm"], q["tissue"], q["ct"], q["state"]), "study", perstate)
    add("disease",     lambda q: (q["tissue"], q["ct"], q["state"]),           "arm",   [t for t in perstate if P[t]["arm"] in DISEASES])
    add("cell_state",  lambda q: (q["arm"], q["tissue"], q["ct"], q["study"]), "state", perstate)
    add("cell_type",   lambda q: (q["arm"], q["tissue"], q["study"]),          "ct",    allstates)
    add("tissue",      lambda q: (q["arm"], q["ct"], q["state"]),              "tissue", perstate)

    df = pd.DataFrame(rows)
    out = res / "controls"
    df.to_csv(out / "factor_representation_pairs.tsv", sep="\t", index=False)
    # native units: raw/graph = mean |Δ| of the log1p(CP10k) expression feature; emb = mean Euclidean ||ΔZ|| over 64 dims
    UNITS = {"raw": "log1p(CP10k)", "graph": "log1p(CP10k) 2hop", "emb": "embed L2 (64d)"}
    agg = df.groupby("factor").agg(
        n_pairs=("raw", "size"),
        raw_mean=("raw", "mean"), raw_sd=("raw", "std"),
        graph_mean=("graph", "mean"), graph_sd=("graph", "std"),
        emb_mean=("emb", "mean"), emb_sd=("emb", "std"))
    floor = agg.loc["batch_floor"] if "batch_floor" in agg.index else None
    summ_rows = []
    for factor, r in agg.iterrows():
        for rep in ("raw", "graph", "emb"):
            summ_rows.append({
                "factor": factor, "representation": rep, "units": UNITS[rep], "n_pairs": int(r.n_pairs),
                "mean_dist": round(r[f"{rep}_mean"], 4), "sd_dist": round(r[f"{rep}_sd"], 4),
                "dist_over_batch_floor": (round(r[f"{rep}_mean"] / floor[f"{rep}_mean"], 3)
                                          if floor is not None and floor[f"{rep}_mean"] else np.nan)})
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(out / "factor_representation_summary.tsv", sep="\t", index=False)
    print(f"hops={hops}  pairs={len(df)}\n")
    for rep in ("raw", "graph", "emb"):
        print(f"[{rep}]  per-protein distance, units = {UNITS[rep]}")
        sub = summ[summ.representation == rep].set_index("factor")[["n_pairs", "mean_dist", "sd_dist", "dist_over_batch_floor"]]
        print(sub.to_string(), "\n")
    print(f"wrote {out/'factor_representation_summary.tsv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    ap.add_argument("--hops", type=int, default=LAYERS, help="graph-propagation hops (default = encoder layers)")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.hops))
