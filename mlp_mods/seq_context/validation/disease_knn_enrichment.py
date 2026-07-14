"""Disease-enriched-biology validation (Reactome pathway kNN enrichment).

Per disease arm, each protein is MEAN-POOLED over that disease's contexts (disease embedding) and over the
TISSUE/CELL-MATCHED healthy contexts (healthy embedding). Seeds = OT known drug targets. For each of:
  - disease : seeds' k-NN in the disease-pooled embedding
  - random  : matched # of random (non-target) proteins' k-NN in the disease-pooled embedding  [target-specificity control]
  - healthy : seeds' k-NN in the healthy-pooled embedding
we take the neighbor set (seeds excluded) and run Reactome pathway hypergeometric enrichment (BH-FDR) vs the
pooled protein background. Output: one row per disease with top-3 pathways for disease / random / healthy.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/disease_knn_enrichment.py --run link_v11 --k 15
Out: results/<run>/disease_knn_enrichment.tsv
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom

SEQ = Path("mlp_mods/seq_context")
REACTOME = Path("mlp_mods/reactome/ReactomePathways.gmt")
PROT = sorted(torch.load("ESM/protein_embeddings.pt", map_location="cpu").keys())
MIN_TERM, MAX_TERM = 3, 500
DIS_EFO = {"crohn": "EFO_0000384", "uc": "EFO_0000729", "ild": "EFO_0004244", "alz": "MONDO_0004975",
           "hvd": "EFO_0009940", "covid": "MONDO_0100096", "athero": "EFO_0003914",
           "bipolar": "MONDO_0004985"}


def reactome_sets():
    g2t, t2g = defaultdict(set), {}
    for ln in REACTOME.read_text().splitlines():
        f = ln.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        name, genes = f[0], set(f[2:])
        if not (MIN_TERM <= len(genes) <= MAX_TERM):
            continue
        t2g[name] = genes
        for g in genes:
            g2t[g].add(name)
    return dict(g2t), t2g


def pooled(ctxarr, idx, emb, tags):
    m = np.isin(ctxarr, list(tags))
    acc = defaultdict(list)
    for j in np.where(m)[0]:
        acc[PROT[idx[j]]].append(emb[j])
    return {g: np.mean(v, 0) for g, v in acc.items()}


def neighbors(emb_dict, seeds, k):
    genes = np.array(sorted(emb_dict))
    X = np.stack([emb_dict[g] for g in genes])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    gi = {g: i for i, g in enumerate(genes)}
    si = [gi[s] for s in seeds if s in gi]
    nb = set()
    for i in si:
        sim = Xn @ Xn[i]; sim[i] = -9
        for j in np.argsort(-sim)[:k]:
            nb.add(genes[j])
    return nb - set(seeds), set(genes)


def enrich(nbrs, bg, t2g, qthr):
    N, n = len(bg), len(nbrs)
    rows = []
    for t, tg in t2g.items():
        K = len(tg & bg); kk = len(tg & nbrs)
        if kk < 2 or K < 3:
            continue
        rows.append((t, kk, K, float(hypergeom.sf(kk - 1, N, K, n)), (kk / n) / (K / N)))
    rows.sort(key=lambda x: x[3]); m = len(rows)
    return [(t, kk, K, p, min(p * m / (i + 1), 1.0), fold) for i, (t, kk, K, p, fold) in enumerate(rows)
            if min(p * m / (i + 1), 1.0) < qthr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v11")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--qthr", type=float, default=0.05)
    args = ap.parse_args()
    print("loading Reactome ...", flush=True)
    g2t, t2g = reactome_sets(); goset = set(g2t)
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    ctxarr, idx, emb = d["context"], d["prot_idx"], d["emb"].astype(np.float64)
    allctx = set(ctxarr)

    def top(emb_dict, seeds):
        nb, genes = neighbors(emb_dict, [s for s in seeds if s in emb_dict], args.k)
        return enrich(nb & goset, genes & goset, t2g, args.qthr)

    def fmt(sig):
        return " | ".join(f"{t} (q={q:.0e})" for t, kk, K, p, q, fold in sig[:3]) or "—"

    rows = []
    for arm, efo in DIS_EFO.items():
        dctxs = {c for c in allctx if c.startswith(arm + "_")}
        hctxs = {"healthy_" + c[len(arm) + 1:] for c in dctxs if "healthy_" + c[len(arm) + 1:] in allctx}
        if not dctxs or not hctxs:
            print(f"skip {arm}: no matched contexts", flush=True); continue
        demb, hemb = pooled(ctxarr, idx, emb, dctxs), pooled(ctxarr, idx, emb, hctxs)
        drug = set(pd.read_csv(f"mlp_mods/03_opentargets_rebuild/known_drugs_{efo}.tsv", sep="\t").gene_symbol.astype(str))
        seeds = sorted(drug & set(demb))
        rng = np.random.default_rng(abs(hash(arm)) % (2 ** 32))
        pool = sorted(set(demb) - drug)
        rand = list(rng.choice(pool, min(len(seeds), len(pool)), replace=False)) if seeds else []
        rows.append({"disease": arm, "n_disease_seeds": len(seeds), "n_random_seeds": len(rand),
                     "disease_pathways": fmt(top(demb, seeds)),
                     "random_pathways": fmt(top(demb, rand)),
                     "healthy_pathways": fmt(top(hemb, seeds))})
        print(f"  {arm}: {len(dctxs)} disease ctx, {len(hctxs)} matched-healthy ctx, {len(seeds)} drug-target seeds", flush=True)
    tab = pd.DataFrame(rows)
    out = SEQ / "results" / args.run / "disease_knn_enrichment.tsv"
    tab.to_csv(out, sep="\t", index=False)
    print("\n" + tab.to_string(index=False) + f"\n\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
