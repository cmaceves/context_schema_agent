"""DE-seeded Random-Walk-with-Restart control (Cowen/Ideker network-propagation baseline).

Seed the walk from the disease-dysregulated genes (seed weight = |disease - matched-healthy expression|),
propagate over the directed network, restart with prob r. Rank proteins by the steady-state score. This is the
canonical "molecular signal + topology" baseline — it combines DE (the seeds) with the graph (the walk), the two
things our embedding-perturbation tries to do implicitly. Compared to DE-alone, PageRank, and the scvi/signed
perturbations on the same context (crohn_colon_macrophage_inflammatory).

Run: .venv/bin/python .../rwr_target_rank.py [--restart 0.5]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

RES = Path("mlp_mods/de_ppi/results")
SCVI = RES / "crohn_alzheimer_ild_uc_embedding_expressed_scvi" / "networks"
HERE = RES / "de_seeded_rwr_control"
OT = Path("mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv")
DIS, HLT = "crohn_colon_macrophage_inflammatory", "healthy_colon_macrophage_inflammatory"


def main(restart, tol=1e-10, max_iter=1000):
    HERE.mkdir(parents=True, exist_ok=True)
    nd = pd.read_csv(SCVI / DIS / "network_nodes.tsv", sep="\t", keep_default_na=False)
    ed = pd.read_csv(SCVI / DIS / "network_edges.tsv", sep="\t", keep_default_na=False)
    hexpr = pd.read_csv(SCVI / HLT / "network_nodes.tsv", sep="\t", keep_default_na=False).set_index("node_id")["expression"].astype(float).to_dict()
    prot = list(nd.node_id); idx = {g: i for i, g in enumerate(prot)}; N = len(prot)
    # DE seed = |disease - matched-healthy expression| (absent in healthy -> 0)
    seed = np.array([abs(float(e) - hexpr.get(g, 0.0)) for g, e in zip(nd.node_id, nd.expression.astype(float))])
    if seed.sum() == 0:
        seed[:] = 1.0
    s = seed / seed.sum()

    # column-stochastic directed transition: M[dst, src] = 1/outdeg(src)
    src = ed.source.map(idx).to_numpy(); dst = ed.target.map(idx).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst)); src, dst = src[keep].astype(int), dst[keep].astype(int)
    outdeg = np.zeros(N); np.add.at(outdeg, src, 1.0)
    w = 1.0 / np.where(outdeg[src] > 0, outdeg[src], 1.0)
    M = sp.csr_matrix((w, (dst, src)), shape=(N, N))
    dangling = outdeg == 0

    # RWR: p = (1-r)*(M p + dangling_mass*s) + r*s
    p = s.copy()
    for _ in range(max_iter):
        p_new = (1 - restart) * (M @ p + p[dangling].sum() * s) + restart * s
        if np.abs(p_new - p).sum() < tol:
            p = p_new; break
        p = p_new
    df = pd.DataFrame({"protein": prot, "de_seed": seed.round(4), "rwr_score": p})
    ot = dict(zip(*[pd.read_csv(OT, sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))
    df["ot"] = df.protein.map(lambda g: round(ot.get(g, 0), 3))
    df = df.sort_values("rwr_score", ascending=False).reset_index(drop=True); df["rank"] = np.arange(1, N + 1)
    df.to_csv(HERE / "rwr_crohn_colon_inflammatory.tsv", sep="\t", index=False)

    top = int(np.ceil(0.1 * N))
    print(f"DE-seeded RWR (restart={restart}) on {DIS}, N={N}, top10%={top}")
    print(f"{'thr':>4s} {'MRR':>7s} {'H@10':>5s} {'H@50':>5s} {'top10%':>7s}")
    for thr in (0.5, 0.3):
        r = df.loc[df.ot > thr, "rank"].to_numpy()
        print(f"{thr:>4} {(1/r).mean():7.4f} {int((r<=10).sum()):5d} {int((r<=50).sum()):5d} {int((r<=top).sum()):4d}/{len(r)}")
    print("\ncompare (same context): PageRank OT>0.3 MRR 0.015 top10% 23/70, OT>0.5 0.052 5/9")
    print("                        scvi-pert OT>0.3 0.024 29/70 | signed 0.018 28/70 | DE-alone 0.005 13/70")
    print("\ntop 12 by RWR:")
    print(df.head(12)[["rank", "protein", "de_seed", "ot"]].to_string(index=False))
    print(f"\nwrote {HERE/'rwr_crohn_colon_inflammatory.tsv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--restart", type=float, default=0.5)
    main(ap.parse_args().restart)
