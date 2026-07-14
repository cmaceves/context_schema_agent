"""Disease-context nearest-neighbor GO enrichment.

Question: in a disease-context embedding, do a protein's k nearest neighbors share its GO
Biological-Process terms more than (a) random proteins and (b) neighbors in raw ESM space?
If EMB > ESM, the co-expression/regulatory training added process-level (pathway) structure
beyond what sequence already encodes.

Two metrics, each for EMB / ESM / random:
  coherence = mean over proteins of [fraction of k-NN sharing >=1 informative GO-BP term]
  pair-AUROC = AUROC(same-GO-BP-term pair | cosine) over sampled protein pairs
Informative GO-BP terms = propagated terms annotating 3..500 genes (drop root/ubiquitous & singletons).

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/nn_enrichment/knn_go_enrichment.py \
        --run link_v7_4ct --context crohn_colon_macrophage_inflammatory --k 15
Out: validation/nn_enrichment/knn_go_<context>.{tsv,png}
"""
from __future__ import annotations
import argparse, gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

SEQ = Path("mlp_mods/seq_context")
GO = Path("mlp_mods/go_enrichment")
OBO = GO / "go-basic.obo"
GAF = GO / "goa_human.gaf.gz"
HERE = SEQ / "validation/nn_enrichment"
MIN_TERM, MAX_TERM = 3, 500                      # informative-term size window


def parse_obo_parents(path):
    parents = defaultdict(set); cur = None; aspect = {}
    for ln in path.read_text().splitlines():
        if ln == "[Term]":
            cur = None
        elif ln.startswith("id: GO:"):
            cur = ln[4:].strip()
        elif ln.startswith("namespace:") and cur:
            aspect[cur] = ln.split(":", 1)[1].strip()
        elif ln.startswith("is_a: GO:") and cur:
            parents[cur].add(ln.split()[1])
        elif ln.startswith("relationship: part_of GO:") and cur:
            parents[cur].add(ln.split()[2])
    return parents, aspect


def ancestors(term, parents, cache):
    if term in cache:
        return cache[term]
    acc = set()
    for p in parents.get(term, ()):
        acc.add(p); acc |= ancestors(p, parents, cache)
    cache[term] = acc
    return acc


def gene_to_bp(parents, aspect):
    direct = defaultdict(set)
    with gzip.open(GAF, "rt") as fh:
        for ln in fh:
            if ln.startswith("!"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 9 or c[8] != "P" or "NOT" in c[3]:
                continue
            direct[c[2]].add(c[4])
    cache = {}
    g2t = {}
    for g, terms in direct.items():
        prop = set(terms)
        for t in terms:
            prop |= ancestors(t, parents, cache)
        prop = {t for t in prop if aspect.get(t) == "biological_process"}
        if prop:
            g2t[g] = prop
    # size filter -> informative terms
    size = defaultdict(int)
    for terms in g2t.values():
        for t in terms:
            size[t] += 1
    ok = {t for t, n in size.items() if MIN_TERM <= n <= MAX_TERM}
    g2t = {g: (terms & ok) for g, terms in g2t.items()}
    return {g: terms for g, terms in g2t.items() if terms}


def knn(M, k):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = Mn @ Mn.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def coherence(nn_idx, termsets):
    frac = []
    for i, ti in enumerate(termsets):
        hits = [len(ti & termsets[j]) > 0 for j in nn_idx[i]]
        frac.append(np.mean(hits))
    return float(np.mean(frac))


def random_coherence(termsets, k, rng, reps=20):
    n = len(termsets); vals = []
    for _ in range(reps):
        idx = np.array([rng.choice(np.delete(np.arange(n), i), k, replace=False) for i in range(n)])
        vals.append(coherence(idx, termsets))
    return float(np.mean(vals)), float(np.std(vals))


def pair_auroc(M, termsets, rng, n_pairs=200000):
    n = len(termsets)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    m = a != b; a, b = a[m], b[m]
    y = np.array([len(termsets[i] & termsets[j]) > 0 for i, j in zip(a, b)])
    cos = np.einsum("ij,ij->i", Mn[a], Mn[b])
    return float(roc_auc_score(y, cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v7_4ct")
    ap.add_argument("--context", default="crohn_colon_macrophage_inflammatory")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("parsing GO ...", flush=True)
    parents, aspect = parse_obo_parents(OBO)
    g2t = gene_to_bp(parents, aspect)
    print(f"  GO-BP annotated genes (informative terms): {len(g2t)}", flush=True)

    esm_all = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
    prot = sorted(esm_all.keys())
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    m = d["context"] == args.context
    genes = np.array([prot[i] for i in d["prot_idx"][m]])
    emb = d["emb"][m].astype(np.float64)
    keep = np.array([g in g2t for g in genes])
    genes, emb = genes[keep], emb[keep]
    esm = np.stack([esm_all[g].numpy() for g in genes]).astype(np.float64)
    termsets = [g2t[g] for g in genes]
    print(f"  context={args.context}  proteins scored={len(genes)}  k={args.k}", flush=True)

    rows = []
    for name, M in [("EMB", emb), ("ESM", esm)]:
        coh = coherence(knn(M, args.k), termsets)
        auc = pair_auroc(M, termsets, rng)
        rows.append({"space": name, "coherence": coh, "pair_auroc": auc})
        print(f"  {name}: coherence={coh:.3f}  pair_AUROC={auc:.3f}", flush=True)
    rc_m, rc_s = random_coherence(termsets, args.k, rng)
    rows.append({"space": "random", "coherence": rc_m, "pair_auroc": 0.5})
    print(f"  random: coherence={rc_m:.3f} +/- {rc_s:.3f}  pair_AUROC=0.500", flush=True)

    res = pd.DataFrame(rows)
    res["n_proteins"] = len(genes); res["k"] = args.k; res["context"] = args.context
    tsv = HERE / f"knn_go_{args.context}.tsv"
    res.to_csv(tsv, sep="\t", index=False); print("wrote", tsv, flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    pal = {"EMB": "#1b9e77", "ESM": "#d95f02", "random": "#999999"}
    for a, col, ttl in [(ax[0], "coherence", f"k-NN share GO-BP term (k={args.k})"),
                        (ax[1], "pair_auroc", "pair AUROC (same-GO | cosine)")]:
        a.bar(res.space, res[col], color=[pal[s] for s in res.space])
        a.set_title(ttl); a.set_ylabel(col)
        if col == "pair_auroc":
            a.axhline(0.5, ls="--", c="k", lw=0.8)
        for i, v in enumerate(res[col]):
            a.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle(f"{args.context}\n{args.run}", fontsize=9)
    fig.tight_layout()
    png = HERE / f"knn_go_{args.context}.png"
    fig.savefig(png, dpi=130); print("wrote", png, flush=True)


if __name__ == "__main__":
    main()
