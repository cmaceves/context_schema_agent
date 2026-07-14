"""Disease-enriched-biology validation, readout #2: TARGET co-clustering (druggability recovery).

Companion to disease_knn_enrichment.py (which does Reactome pathway enrichment). Same per-disease pooling:
each protein is MEAN-POOLED over that disease's contexts (disease embedding) and over the TISSUE/CELL-MATCHED
healthy contexts (healthy embedding). Seeds = that disease's OT known drug targets. For each of:
  - disease : seeds' k-NN in the disease-pooled embedding
  - random  : matched # of random (non-target) proteins' k-NN in disease-pooled embedding   [specificity control]
  - healthy : seeds' k-NN in the healthy-pooled embedding
we take the neighbor set (seeds excluded) and ask whether it is ENRICHED for being an OT drug target of ANY
disease (all-OT union label) vs the pooled protein background, via hypergeometric + fold enrichment.
Readout: do drug targets cluster near OTHER drug targets, and is that stronger in disease contexts?

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/disease_knn_target_enrichment.py --run link_v11 --k 50
Out: results/<run>/disease_knn_target_enrichment.tsv
"""
from __future__ import annotations
import argparse, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom

SEQ = Path("mlp_mods/seq_context")
PROT = sorted(torch.load("ESM/protein_embeddings.pt", map_location="cpu").keys())
DIS_EFO = {"crohn": "EFO_0000384", "uc": "EFO_0000729", "ild": "EFO_0004244", "alz": "MONDO_0004975",
           "hvd": "EFO_0009940", "covid": "MONDO_0100096", "athero": "EFO_0003914",
           "bipolar": "MONDO_0004985"}


def all_ot_union():
    u = set()
    for f in glob.glob("mlp_mods/03_opentargets_rebuild/known_drugs_*.tsv"):
        u |= set(pd.read_csv(f, sep="\t").gene_symbol.astype(str))
    return u


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


def target_enrich(nbrs, bg, targets):
    """Hypergeom: are neighbors (nbrs) enriched for all-OT targets vs background bg?"""
    N = len(bg); K = len(targets & bg); n = len(nbrs); kk = len(nbrs & targets)
    if n == 0 or K == 0:
        return kk, n, K, N, float("nan"), float("nan")
    p = float(hypergeom.sf(kk - 1, N, K, n))
    fold = (kk / n) / (K / N) if K else float("nan")
    return kk, n, K, N, p, fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v11")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    targets = all_ot_union()
    print(f"all-OT union label: {len(targets)} proteins", flush=True)
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    ctxarr, idx, emb = d["context"], d["prot_idx"], d["emb"].astype(np.float64)
    allctx = set(ctxarr)

    def stat(emb_dict, seeds):
        nb, genes = neighbors(emb_dict, [s for s in seeds if s in emb_dict], args.k)
        bg = genes & set(PROT)
        return target_enrich(nb & bg, bg, targets & bg)

    def fmt(s):
        kk, n, K, N, p, fold = s
        return f"{fold:.2f}x (q~{p:.0e}; {kk}/{n} nbr are targets)" if n else "—"

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
                     "disease_target_enrich": fmt(stat(demb, seeds)),
                     "random_target_enrich": fmt(stat(demb, rand)),
                     "healthy_target_enrich": fmt(stat(hemb, seeds))})
        print(f"  {arm}: {len(seeds)} seeds, {len(dctxs)} disease ctx, {len(hctxs)} matched-healthy ctx", flush=True)
    tab = pd.DataFrame(rows)
    out = SEQ / "results" / args.run / "disease_knn_target_enrichment.tsv"
    tab.to_csv(out, sep="\t", index=False)
    print("\n" + tab.to_string(index=False) + f"\n\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
