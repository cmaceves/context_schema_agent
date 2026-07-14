"""Graph-only PageRank control: rank proteins by a random walk on each network's TOPOLOGY alone (no expression,
no DE seeds), and compare to the scvi in-silico perturbation. This is the sharpest test of "is the embedding
just topology?" — if PageRank recovers OpenTargets targets as well as the perturbation, the expression-informed
embedding adds nothing over graph structure.

PageRank = stationary distribution of a damped random walk on the directed graph (walker follows source→target
edges; damping 0.85; dangling nodes teleport uniformly). Computed per network from network_edges only.

Outputs (results/random_walk_control/):
  pagerank_crohn_colon_inflammatory.tsv          per-protein PageRank + rank
  rank_comparison_OT0.5.tsv                       OT>0.5 targets: pagerank_rank vs perturbation_rank vs DE_rank
  images/pagerank_vs_perturbation_ranks.png       scatter for the OT>0.5 targets
  + printed OT-recovery metrics (MRR / Hits@10,50 / top-decile) for PageRank, side by side with DE + perturbation

Run: .venv/bin/python mlp_mods/de_ppi/results/random_walk_control/rw_target_rank.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt

sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
try:
    from plot_style import apply_style
except Exception:
    def apply_style(): pass

BUILD = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi")
HERE = Path("mlp_mods/de_ppi/results/random_walk_control")
DE_CTRL = Path("mlp_mods/de_ppi/results/differential_expression_control")
OT = Path("mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv")
EVAL_TAG = "crohn_colon_macrophage_inflammatory"
DAMP = 0.85


def pagerank(tag, damping=DAMP, tol=1e-10, max_iter=500):
    """directed PageRank over the network's proteins; returns Series indexed by protein."""
    nodes = pd.read_csv(BUILD / "networks" / tag / "network_nodes.tsv", sep="\t", keep_default_na=False)
    edges = pd.read_csv(BUILD / "networks" / tag / "network_edges.tsv", sep="\t", keep_default_na=False)
    prot = list(nodes.node_id); idx = {g: i for i, g in enumerate(prot)}; n = len(prot)
    s = edges.source.map(idx).to_numpy(); d = edges.target.map(idx).to_numpy()
    keep = ~(pd.isna(s) | pd.isna(d))
    s, d = s[keep].astype(int), d[keep].astype(int)
    outdeg = np.zeros(n); np.add.at(outdeg, s, 1.0)
    w = 1.0 / np.where(outdeg[s] > 0, outdeg[s], 1.0)          # column-stochastic: M[d,s] = 1/outdeg(s)
    M = sp.csr_matrix((w, (d, s)), shape=(n, n))
    dangling = (outdeg == 0)
    p = np.full(n, 1.0 / n)
    tele = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        p_new = (1 - damping) * tele + damping * (M @ p + tele * p[dangling].sum())
        if np.abs(p_new - p).sum() < tol:
            p = p_new; break
        p = p_new
    return pd.Series(p, index=prot)


def rank_metrics(order_desc, ot, thr, N):
    """order_desc: proteins ranked best->worst. returns (MRR, hits10, hits50, top10pct, npos)."""
    rank = {g: i + 1 for i, g in enumerate(order_desc)}
    pos = [g for g in order_desc if ot.get(g, 0) > thr]
    r = np.array([rank[g] for g in pos]); npos = len(r)
    top = int(np.ceil(0.10 * N))
    return (1 / r).mean(), int((r <= 10).sum()), int((r <= 50).sum()), int((r <= top).sum()), npos


def main():
    ot = dict(zip(*[pd.read_csv(OT, sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))
    pr = pagerank(EVAL_TAG).sort_values(ascending=False)
    pr_df = pd.DataFrame({"protein": pr.index, "pagerank": pr.values})
    pr_df["pr_rank"] = np.arange(1, len(pr_df) + 1)
    pr_df.to_csv(HERE / "pagerank_crohn_colon_inflammatory.tsv", sep="\t", index=False)
    N = len(pr_df)

    # perturbation ranks (raw projection) + DE ranks (from the DE control, if present)
    pert = pd.read_csv(BUILD / "insilico_perturb" / f"{EVAL_TAG}_perturbation_results.tsv", sep="\t")
    pert = pert.sort_values("projection", ascending=False).reset_index(drop=True)
    pert_rank = {g: i + 1 for i, g in enumerate(pert.protein)}
    pert_order = list(pert.protein)
    de_rank = {}
    de_file = DE_CTRL / "de_target_rank_crohn_colon_inflammatory.tsv"
    if de_file.exists():
        de = pd.read_csv(de_file, sep="\t")
        de_rank = dict(zip(de.protein, de.de_rank))

    # side-by-side OT recovery
    print(f"OT-target recovery in {EVAL_TAG} (N={N}); top 10% = {int(np.ceil(0.1*N))} ranks")
    print(f"{'ranking':14s} {'thr':>4s} {'MRR':>7s} {'H@10':>5s} {'H@50':>5s} {'top10%':>7s}")
    for thr in (0.5, 0.3):
        for name, order in [("pagerank", list(pr_df.protein)), ("perturbation", pert_order)]:
            mrr, h10, h50, tp, npos = rank_metrics(order, ot, thr, N)
            print(f"{name:14s} {thr:>4} {mrr:7.4f} {h10:5d} {h50:5d} {tp:4d}/{npos}")

    # OT>0.5 rank comparison table + plot (pagerank vs perturbation)
    pos = sorted([g for g in pr_df.protein if ot.get(g, 0) > 0.5], key=lambda g: -ot.get(g, 0))
    comp = pd.DataFrame({"protein": pos})
    comp["ot"] = comp.protein.map(ot).round(3)
    comp["pr_rank"] = comp.protein.map(dict(zip(pr_df.protein, pr_df.pr_rank)))
    comp["pert_rank"] = comp.protein.map(pert_rank)
    if de_rank:
        comp["de_rank"] = comp.protein.map(de_rank)
    comp.to_csv(HERE / "rank_comparison_OT0.5.tsv", sep="\t", index=False)
    print("\nOT>0.5 targets:\n" + comp.to_string(index=False))

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim = N + 30
    ax.plot([1, lim], [1, lim], ls="--", lw=1, color="0.6", zorder=0)
    ax.scatter(comp.pr_rank, comp.pert_rank, s=70, color="#DD8452", edgecolors="0.2", linewidths=0.6, zorder=3)
    for _, r in comp.iterrows():
        ax.annotate(r.protein, (r.pr_rank, r.pert_rank), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("rank by PageRank (graph-only, 1 = best)")
    ax.set_ylabel("rank by perturbation (1 = best)")
    ax.set_title(f"OT>0.5 target ranks: PageRank vs perturbation\n{EVAL_TAG} (N={N})", fontsize=9.5)
    fig.tight_layout()
    out = HERE / "images" / "pagerank_vs_perturbation_ranks.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    from scipy.stats import spearmanr
    rho, p = spearmanr(comp.pr_rank, comp.pert_rank)
    print(f"\nSpearman(pagerank_rank, perturbation_rank) over OT>0.5 = {rho:+.2f} (p={p:.2f})")
    print(f"wrote {out}, {HERE/'rank_comparison_OT0.5.tsv'}, {HERE/'pagerank_crohn_colon_inflammatory.tsv'}")


if __name__ == "__main__":
    main()
