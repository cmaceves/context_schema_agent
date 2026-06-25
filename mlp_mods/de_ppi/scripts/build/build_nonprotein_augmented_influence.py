"""Non-protein-augmented (disease, cell type) PPI influence — one-off augmented analysis.

Extends the lit-only influence with our curated NON-PROTEIN entity layer. Built for
beta_type2diabetes but parameterized by --build.

Network:
  - proteins        = PINNACLE cell-type proteins  UNION  literature dysregulation genes
  - non-protein     = the candidate set (candidate_nonprotein_nodes_min3edges_goenriched.tsv:
                      HMDB-or-blank chemicals + GO-enriched processes)  UNION  lit metabolites
  - protein->protein edges = OmniPath (directed)
  - entity<->protein edges = db/edges.tsv accepted types {activates, inhibits, regulates,
                      in_reaction_with, positively_regulates, negatively_regulates}, taken
                      DIRECTED (head->tail) but UNSIGNED (sign ignored), weight 1.
  - operator P[i,j] = w(i)/Z(j), w=1 (uniform, unsigned reach), influence = sum_{k=0..3} P^k m

Target set m = literature dysregulation GENES + literature dysregulation METABOLITES.
  (GO processes are pure sinks and are NOT in m -> inert for influence, kept as structure.)

Outputs (de_ppi/results/<build>/, prefixed nonprotein_augmented_, nothing overwritten):
  nonprotein_augmented_P3_influence.tsv
  nonprotein_augmented_network_nodes.tsv / _network_edges.tsv
  influence_analysis/nonprotein_augmented_<slug>_drug_influence.tsv
  influence_analysis/nonprotein_augmented_phase_vs_percentile.png

Run:
  .venv/bin/python mlp_mods/de_ppi/build_nonprotein_augmented_influence.py --build beta_type2diabetes
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("mlp_mods/de_ppi")))
from config import load_build

HOPS = 3
ACCEPT = {"activates", "inhibits", "regulates", "in_reaction_with",
          "positively_regulates", "negatively_regulates"}
CANDIDATE = ("mlp_mods/rank_shifts/{build}_paired/"
             "candidate_nonprotein_nodes_min3edges_goenriched.tsv")
# secretion/exocytosis processes treated as dysregulated phenotype sinks (--secretion-targets)
SECRETION_PROCESSES = {
    "GO:0030073": "insulin secretion", "GO:0030072": "peptide hormone secretion",
    "GO:0046879": "hormone secretion", "GO:0009306": "protein secretion",
    "GO:0046903": "secretion", "GO:0006887": "exocytosis",
    "GO:0017156": "calcium ion regulated exocytosis", "GO:0006906": "vesicle fusion",
    "GO:0016079": "synaptic vesicle exocytosis", "GO:0099504": "synaptic vesicle cycle",
    "GO:0036465": "synaptic vesicle recycling", "GO:0006811": "ion transport",
    "GO:0034220": "ion transmembrane transport"}
# same negative-control comparison diseases as the IBD boxplot (their OT drug targets
# scored in THIS build's network): (display name, OT label, point color, box color)
COMPARISONS = [("Gout", "gout", "#e8820c", "#b35900"),
               ("Rheumatoid\narthritis", "rheumatoid arthritis", "#2e9e5b", "#1b6b3a"),
               ("Psoriasis", "psoriasis", "#7b3fa0", "#5e2e7e"),
               ("Systemic lupus\nerythematosus", "systemic lupus erythematosus", "#c0392b", "#922b21"),
               ("Athero-\nsclerosis", "atherosclerosis", "#1f9e9e", "#166b6b")]


def _norm_chebi(c):
    c = str(c).strip()
    return c if c.upper().startswith("CHEBI") else f"CHEBI:{c}"


def main(build: str, secretion_targets: bool = False) -> int:
    cfg = load_build(build)
    repo = Path(".")
    pref = "nonprotein_augmented_secretion_" if secretion_targets else "nonprotein_augmented_"

    # --- literature dysregulation: genes (m) + metabolites (m) ---
    lit = pd.read_csv(cfg.lit_genes, sep="\t")
    lit_genes = set(lit.entity)
    litm = pd.read_csv(cfg.lit_metabolites, sep="\t", dtype=str)
    lit_chebi = {_norm_chebi(c) for c in litm.chebi_id if str(c).strip()}

    # --- protein node set: PINNACLE cell-type proteins UNION lit genes ---
    mac = pd.read_csv(cfg.celltype_ppi, sep=" ", header=None, names=["a", "b"])
    prot_syms = (set(mac.a) | set(mac.b)) | lit_genes

    op = pd.read_csv(cfg.omni, sep="\t")
    op = op[op.src.isin(prot_syms) & op.dst.isin(prot_syms) & (op.src != op.dst)].drop_duplicates(["src", "dst"])
    omni_prots = prot_syms & (set(op.src) | set(op.dst))

    # --- id<->symbol maps; non-protein node set (candidate UNION lit metabolites) ---
    nodes = pd.read_csv(cfg.nodes, dtype=str, keep_default_na=False)
    gmm = nodes[nodes.label == "MacromolecularMachine"]
    sym2id = dict(zip(gmm.name, gmm.id)); id2sym = {v: k for k, v in sym2id.items()}
    cand = pd.read_csv(repo / CANDIDATE.format(build=build), sep="\t", dtype=str, keep_default_na=False)
    np_ids = set(cand.id) | lit_chebi               # chemicals + processes + lit metabolites
    id2label = dict(zip(nodes.id, nodes.label))

    # --- entity<->protein edges from the KG (accepted types, directed, unsigned) ---
    np_edges = []                                   # (head_key, tail_key) in symbol/id space
    rel_of = {}
    with open(cfg.edges) as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        for h, k, t in r:
            if k not in ACCEPT:
                continue
            # protein endpoint -> symbol (must be an in-network protein); entity endpoint -> id (in np set)
            if h in id2sym and id2sym[h] in prot_syms and t in np_ids:        # protein -> entity
                a, b = id2sym[h], t
            elif t in id2sym and id2sym[t] in prot_syms and h in np_ids:      # entity -> protein
                a, b = h, id2sym[t]
            else:
                continue
            np_edges.append((a, b)); rel_of.setdefault((a, b), set()).add(k)

    # entities that actually connect into the network
    np_connected = {e for pair in np_edges for e in pair if e in np_ids}

    # --- assemble node index: proteins (omni-incident OR entity-connected) + connected entities ---
    prots = sorted(omni_prots | {e for pair in np_edges for e in pair if e in prot_syms})
    ents = sorted(np_connected)
    order = prots + ents
    idx = {n: i for i, n in enumerate(order)}; N = len(order)
    nprot = len(prots)
    print(f"proteins: {nprot} | non-protein nodes: {len(ents)} "
          f"(chemicals/processes/lit-metabs connected) | N={N}", flush=True)

    # --- edges -> directed operator (unsigned, w=1) ---
    src = [idx[s] for s in op.src] + [idx[a] for a, b in np_edges]
    dst = [idx[d] for d in op.dst] + [idx[b] for a, b in np_edges]
    src, dst = np.array(src), np.array(dst)
    w = np.ones(N)
    P = sp.coo_matrix((w[src], (src, dst)), shape=(N, N)).tocsc()
    Z = np.asarray(P.sum(0)).ravel()
    P = (P @ sp.diags(np.divide(1.0, Z, out=np.zeros_like(Z), where=Z > 0))).tocsr()

    # --- target set m = lit genes + lit metabolites (present as nodes) ---
    m = np.zeros(N)
    for g in lit_genes:
        if g in idx: m[idx[g]] = 1.0
    for c in lit_chebi:
        if c in idx: m[idx[c]] = 1.0
    n_sec = 0
    if secretion_targets:                                  # add secretion/exocytosis processes as sinks
        for pid in SECRETION_PROCESSES:
            if pid in idx: m[idx[pid]] = 1.0; n_sec += 1
    print(f"target set m: {int(m.sum())} ({sum(1 for g in lit_genes if g in idx)} lit genes + "
          f"{sum(1 for c in lit_chebi if c in idx)} lit metabolites + {n_sec} secretion processes)", flush=True)

    infl = m.copy(); v = m.copy()                          # set_influence: reach to m
    for _ in range(HOPS):
        v = P @ v; infl += v

    gen = np.ones(N); v = np.ones(N)                       # general_influence: reach to ALL nodes (hubness)
    for _ in range(HOPS):
        v = P @ v; gen += v

    # specificity: z-score of reach-to-m above reach to DEGREE-MATCHED random target sets
    B = 1000
    indeg = np.bincount(dst, minlength=N).astype(float)    # incoming-edge count (= reachability of a target)
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
    acc = Mn.copy(); V = Mn.copy()
    for _ in range(HOPS):
        V = P @ V; acc += V
    mu, sd = acc.mean(1), acc.std(1)
    spec = np.divide(infl - mu, sd, out=np.zeros(N), where=sd > 0)
    print(f"specificity: degree-matched null B={B} | proteins z>3: {int((spec[:len(prots)] > 3).sum())}", flush=True)

    # --- P3 influence (protein rows): general_influence / set_influence / specificity ---
    rows = [{"protein": g, "set_influence": float(infl[idx[g]]),
             "general_influence": float(gen[idx[g]]), "specificity": round(float(spec[idx[g]]), 3),
             "is_target": int(g in lit_genes)} for g in prots]
    df = pd.DataFrame(rows).sort_values("set_influence", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["specificity_rank"] = df.specificity.rank(ascending=False).astype(int)
    p3 = cfg.results_dir / f"{pref}P3_influence.tsv"
    p3.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p3, sep="\t", index=False)
    print(f"wrote {p3}", flush=True)

    # --- network manifests ---
    nrows = ([{"node_id": g, "node_type": "protein"} for g in prots] +
             [{"node_id": e, "node_type": id2label.get(e, "non_protein")} for e in ents])
    pd.DataFrame(nrows).to_csv(cfg.results_dir / f"{pref}network_nodes.tsv", sep="\t", index=False)
    erows = ([{"source": s, "target": d, "edge_origin": "OmniPath", "edge_property": ""}
              for s, d in zip(op.src, op.dst)] +
             [{"source": a, "target": b, "edge_origin": "KG",
               "edge_property": ",".join(sorted(rel_of[(a, b)]))} for a, b in np_edges])
    pd.DataFrame(erows).to_csv(cfg.results_dir / f"{pref}network_edges.tsv", sep="\t", index=False)
    print(f"wrote {pref}network_nodes/_edges ({len(nrows)} nodes, {len(erows)} edges)", flush=True)

    # --- drug table (OpenTargets, same join as drug_influence_table.py) ---
    n = len(df)
    rank = dict(zip(df.protein, df["rank"]))
    pct = {g: round(100.0 * (1.0 - (r - 1) / n), 1) for g, r in rank.items()}
    kd = pd.read_parquet(cfg.known_drugs_parquet)
    scope = cfg.drug_scope
    anc = lambda a: isinstance(a, (list, np.ndarray)) and any(s in set(a) for s in scope)
    insc = kd[kd["diseaseId"].isin(scope) | kd["ancestors"].apply(anc)]
    drows = []
    for did, sub in insc.groupby("drugId"):
        tg = sorted({s for s in sub.approvedSymbol.dropna() if s})
        innet = [(rank[g], g) for g in tg if g in rank]
        br, bt = min(innet) if innet else (None, "")
        drows.append({"drug_name": sub.prefName.dropna().iloc[0] if sub.prefName.notna().any() else "",
                      "drugId": did, "drug_phase": float(sub.phase.max()),
                      "indications": "; ".join(sorted({s for s in sub.label.dropna() if s})),
                      "influence_rank": br, "influence_percentile": pct.get(bt, np.nan) if bt else np.nan,
                      "influence_target": bt, "n_targets": len(tg), "all_targets": "; ".join(tg),
                      "drugType": sub.drugType.dropna().iloc[0] if sub.drugType.notna().any() else "",
                      "mechanism_of_action": sub.mechanismOfAction.dropna().iloc[0] if sub.mechanismOfAction.notna().any() else ""})
    dt = pd.DataFrame(drows).sort_values("influence_rank", na_position="last").reset_index(drop=True)
    dt["influence_rank"] = dt["influence_rank"].astype("Int64")
    dtp = cfg.influence_dir / f"{pref}{cfg.disease_slug}_drug_influence.tsv"
    dtp.parent.mkdir(parents=True, exist_ok=True)
    dt.to_csv(dtp, sep="\t", index=False)
    print(f"wrote {dtp} ({len(dt)} drugs, {int(dt.influence_rank.notna().sum())} with in-network target)", flush=True)

    # --- boxplot: ONE point per (drug, mechanism); control targets shared with disease REMOVED ---
    disease_targets = {s for s in insc.approvedSymbol.dropna() if s}

    def moa_points(sub, exclude=frozenset()):              # -> [(pct, target, phase)], one per (drug, MoA)
        pts = []
        for _, dg in sub.groupby("drugId"):
            ph = float(dg.phase.max()) if dg.phase.notna().any() else None
            for _, mg in dg.groupby("mechanismOfAction", dropna=False):
                innet = [(pct[t], t) for t in {s for s in mg.approvedSymbol.dropna() if s}
                         if t in pct and t not in exclude]
                if innet:
                    v, t = max(innet); pts.append((v, t, ph))
        return pts

    phase_pts = {}
    for v, t, ph in moa_points(insc):                      # disease panel: all disease drug-mechanisms
        if ph is not None:
            phase_pts.setdefault(ph, []).append(v)
    phases = sorted(phase_pts)

    fig, ax = plt.subplots(figsize=(13, 5.5)); rng = np.random.default_rng(0)

    def panel(pos, vals, color, boxcolor):
        if vals:
            ax.boxplot([vals], positions=[pos], widths=0.6, showfliers=False,
                       medianprops=dict(color="black", linewidth=2), boxprops=dict(color=boxcolor),
                       whiskerprops=dict(color=boxcolor), capprops=dict(color=boxcolor))
            ax.scatter(pos + rng.uniform(-0.18, 0.18, len(vals)), vals, s=24, alpha=1.0,
                       color=color, edgecolor="black", linewidth=0.3, zorder=3)
        return len(vals)

    xt, xl = [], []
    for i, p in enumerate(phases):
        nn = panel(i, phase_pts[p], "#2b6cb0", "#4a4a4a"); xt.append(i); xl.append(f"Phase {p:g}\n(n={nn})")
    for j, (name, label, ptc, bxc) in enumerate(COMPARISONS):
        pos = len(phases) + 0.6 + j
        vals = [v for v, _, _ in moa_points(kd[kd.label == label], exclude=disease_targets)]
        nn = panel(pos, vals, ptc, bxc); xt.append(pos); xl.append(f"{name}\n(n={nn})")
        print(f"  comparison {label}: {nn} drug-mechanism points (disease-shared targets removed)", flush=True)
    ax.axvline(len(phases) - 0.2, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_xlabel(f"{cfg.disease} drugs by phase  |  control diseases   "
                  f"(one point per drug-mechanism; control targets shared with {cfg.disease} removed)")
    ax.set_ylabel("Network influence percentile\n(higher = more influential)")
    ax.set_title(f"Non-protein-augmented influence: {cfg.disease} / {cfg.cell_type}")
    ax.set_ylim(0, 101); ax.set_yticks(range(0, 101, 10)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pp = cfg.influence_dir / f"{pref}phase_vs_percentile.png"
    fig.savefig(pp, dpi=150)
    print(f"wrote {pp}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="non-protein-augmented PPI influence")
    ap.add_argument("--build", default="beta_type2diabetes")
    ap.add_argument("--secretion-targets", action="store_true",
                    help="add secretion/exocytosis processes to the dysregulated set m (sinks)")
    a = ap.parse_args()
    raise SystemExit(main(a.build, a.secretion_targets))
