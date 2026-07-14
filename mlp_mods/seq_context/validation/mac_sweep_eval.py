"""Evaluate a set of runs (per-disease pool + max-agg MLP, all-OT recovery) and ONE shared ESM baseline.
For macrophage-only sweeps: the runs share the same context/pool, so ESM is computed once. Prints H@10/H@100/MRR.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/mac_sweep_eval.py link_v14_mac_l010 link_v14_mac_l025 link_v14_mac_l050
"""
from __future__ import annotations
import sys, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

SEQ = Path("mlp_mods/seq_context")
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())


def mlp():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0))


def pool_disease(d, arm):
    m = np.array([c.startswith(arm + "_") for c in d["context"]]); acc = defaultdict(list)
    E = d["emb"].astype(np.float64); idx = d["prot_idx"]
    for j in np.where(m)[0]:
        acc[PROT[idx[j]]].append(E[j])
    return {g: np.mean(v, 0) for g, v in acc.items()}


def rank(score, pos):
    g = np.array(list(score)); sc = np.array([score[x] for x in g]); rk = {x: i + 1 for i, x in enumerate(g[np.argsort(-sc)])}
    P = [x for x in g if x in pos]
    return (sum(rk[x] <= 10 for x in P), sum(rk[x] <= 100 for x in P), float(np.mean([1 / rk[x] for x in P])))


def combined(d, pos, feat):
    arms = sorted({c.split("_")[0] for c in d["context"]} - {"healthy"})
    cmax = {}
    for arm in arms:
        gp = pool_disease(d, arm); genes = np.array(sorted(set(gp) & set(PROT)))
        if len(genes) < 50:
            continue
        y = np.array([1 if g in pos else 0 for g in genes])
        if y.sum() < 3:
            continue
        X = np.stack([gp[g] for g in genes]) if feat == "EMB" else np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
        P = cross_val_predict(mlp(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        for g, p in zip(genes, P):
            cmax[g] = max(cmax.get(g, -1.0), float(p))
    return cmax, arms


def main():
    runs = sys.argv[1:]
    pos = set()
    for f in glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"):
        pos |= set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))

    print(f"{'run/feature':26s} {'arms':>4s} {'pool':>6s} {'H@10':>5s} {'H@100':>6s} {'MRR':>8s}", flush=True)
    esm_done = False
    for r in runs:
        d = np.load(SEQ / "results" / r / "embeddings.npz", allow_pickle=True)
        cm, arms = combined(d, pos, "EMB")
        h10, h100, mrr = rank(cm, pos)
        print(f"{r + '  EMB':26s} {len(arms):4d} {len(cm):6d} {h10:5d} {h100:6d} {mrr:8.4f}", flush=True)
        if not esm_done:                                   # ESM once (shared pool across the sweep)
            cme, _ = combined(d, pos, "ESM"); h10e, h100e, mrre = rank(cme, pos)
            print(f"{'ESM (shared)':26s} {len(arms):4d} {len(cme):6d} {h10e:5d} {h100e:6d} {mrre:8.4f}", flush=True)
            esm_done = True


if __name__ == "__main__":
    main()
