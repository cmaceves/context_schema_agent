"""Step 3a — centroid leave-one-out retrieval of drug targets from the cisTarget embedding, PER CONTEXT.
See seq_context/SEQ_CONTEXT_EMBED.md.

For each disease context: seed = that disease's known drug targets present; leave-one-out centroid; cosine-rank all
proteins. Controls: degree (hubs?), ESM (sequence?), permutation null (luck?). Prints a per-context table.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/centroid_retrieval.py
     [--context <tag>]  (single context; default = loop all disease contexts)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
DRUG = {"crohn": "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0000384.tsv",
        "uc":    "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0000729.tsv",
        "ild":   "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0004244.tsv"}


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def loo_scores(E, S_idx):
    full = E[S_idx].mean(0); full /= np.linalg.norm(full) + 1e-9
    s = E @ full
    for t in S_idx:
        others = [j for j in S_idx if j != t]
        c = E[others].mean(0); c /= np.linalg.norm(c) + 1e-9
        s[t] = E[t] @ c
    return s


def auroc_recall(scores, y):
    order = np.argsort(-scores); ranks = np.empty(len(scores), int); ranks[order] = np.arange(len(scores))
    tr = ranks[y == 1]
    return roc_auc_score(y, scores), float(np.mean(tr < 100))


def run_ctx(ctx, prot, esm_all, d, rng, nperm):
    m = d["context"] == ctx
    genes = np.array([prot[i] for i in d["prot_idx"][m]])
    EMB = unit(d["emb"][m].astype(np.float64))
    ESM = unit(np.stack([esm_all[g].numpy() for g in genes]).astype(np.float64))
    gidx = {g: i for i, g in enumerate(genes)}
    arm = ctx.split("_")[0]
    kd = pd.read_csv(DRUG[arm], sep="\t")
    S = sorted(set(kd.gene_symbol.astype(str)) & set(genes))
    if len(S) < 5:
        return None
    S_idx = np.array([gidx[g] for g in S]); y = np.zeros(len(genes)); y[S_idx] = 1
    ct = pd.read_csv(NET / ctx / "edges_cistarget.tsv", sep="\t")
    deg = pd.concat([ct.tf, ct.target]).value_counts()
    degv = np.array([deg.get(g, 0) for g in genes], float)
    emb_auc, emb_rec = auroc_recall(loo_scores(EMB, S_idx), y)
    esm_auc, _ = auroc_recall(loo_scores(ESM, S_idx), y)
    deg_auc, _ = auroc_recall(degv, y)
    null = np.empty(nperm)
    for i in range(nperm):
        r = rng.choice(len(genes), size=len(S), replace=False)
        yr = np.zeros(len(genes)); yr[r] = 1
        null[i] = roc_auc_score(yr, loo_scores(EMB, r))
    p = (np.sum(null >= emb_auc) + 1) / (nperm + 1)
    return dict(context=ctx.replace("_macrophage", ""), n_targets=len(S), emb=emb_auc, esm=esm_auc,
                degree=deg_auc, recall100=emb_rec, perm_p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v4_cistarget")
    ap.add_argument("--context", default=None)
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    prot = sorted(torch.load("ESM/protein_embeddings.pt", map_location="cpu").keys())
    esm_all = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)

    if args.context:
        ctxs = [args.context]
    else:
        ctxs = [c for c in sorted(set(d["context"])) if c.split("_")[0] in DRUG]
    rows = []
    for c in ctxs:
        r = run_ctx(c, prot, esm_all, d, rng, args.nperm)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    print("\n=== Step 3a: centroid LOO retrieval per context (drug targets) ===", flush=True)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\nMEDIAN: emb={df.emb.median():.3f}  esm={df.esm.median():.3f}  degree={df.degree.median():.3f} | "
          f"contexts where emb>esm & emb>degree & p<0.05: "
          f"{int(((df.emb>df.esm)&(df.emb>df.degree)&(df.perm_p<0.05)).sum())}/{len(df)}", flush=True)
    df.to_csv(SEQ / "validation" / f"centroid_percontext_{args.run}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
