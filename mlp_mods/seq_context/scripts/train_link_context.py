"""Context-specific protein embedding via regulatory-neighbor link prediction.
See mlp_mods/seq_context/SEQ_CONTEXT_EMBED.md. Run in .venv_scvi (torch + CUDA).

Model:  z = Encoder( frozen ESM(1280) + [disease|tissue|state|cell_type] learned embeddings )   -> ctx-specific
        protein embedding (EMB_DIM). Directed decoder scores (TF->target): sigmoid( src_head(z_tf) . tgt_head(z_tgt) ).
        BCE on top-k SCENIC edges (positives) vs sampled non-edges (negatives).

Eval (held-out EDGES): AUC/AP for the FULL model (uses context) and the CONTEXT-BLIND model (context zeroed,
retrained); the context-LIFT is FULL - BLIND. Also saves per-(protein,context) FULL embeddings.

Nodes restricted to ESM-covered proteins (both endpoints need a sequence feature).
Out: mlp_mods/seq_context/results/<run>/{metrics.json, embeddings.npz}
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
ESM_PT = Path("ESM/protein_embeddings.pt")
EMB_DIM = 128
CTX_DIMS = {"cell_type": 64, "disease": 32, "tissue": 32, "state": 32}
ESM_PROJ = 256       # project frozen ESM 1280->256 so it doesn't dimensionally dominate the 160-d context
POS_TOPK = 50        # positives = each TF's top-50 targets (edges_topk.tsv)
PROTECT_TOPN = 300   # hard negatives must fall OUTSIDE a context's top-300/TF (avoid sub-threshold real edges)


def parse_tag(tag):
    # <arm>_<tissue>_macrophage_<state>
    a = tag.split("_")
    return {"disease": a[0], "tissue": a[1], "cell_type": a[2], "state": a[3]}


class Encoder(nn.Module):
    def __init__(self, ctx_total, hidden=512):
        super().__init__()
        # protein feature (ESM_PROJ-d) is produced upstream (projected ESM, or learned protein-ID embedding)
        self.net = nn.Sequential(nn.Linear(ESM_PROJ + ctx_total, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, EMB_DIM))

    def forward(self, prot_feat, ctx):
        return self.net(torch.cat([prot_feat, ctx], -1))


class AdditiveEncoder(nn.Module):
    """z = z_protein(ESM only) + z_context(context only). Context enters as a protein-agnostic additive offset
    (same shift for every protein in a context) — a strictly weaker, interpretable decomposition vs concat-MLP."""
    def __init__(self, ctx_total, hidden=512):
        super().__init__()
        self.p = nn.Sequential(nn.Linear(ESM_PROJ, hidden), nn.ReLU(), nn.Linear(hidden, EMB_DIM))   # z_protein
        self.c = nn.Sequential(nn.Linear(ctx_total, hidden), nn.ReLU(), nn.Linear(hidden, EMB_DIM))   # z_context

    def forward(self, prot_feat, ctx):
        return self.p(prot_feat) + self.c(ctx)


class LinkModel(nn.Module):
    def __init__(self, esm_mat, factor_sizes, zero=None, protein_repr="esm", arch="concat", n_pathways=0, has_expr=False):
        # zero: None = full context; "all" = blind; or a factor name to zero just that factor (ablation)
        # protein_repr: "esm" = frozen ESM down-projection; "id" = random-init learnable per-protein embedding
        super().__init__()
        self.zero = zero
        self.protein_repr = protein_repr
        self.register_buffer("esm", esm_mat)               # (P, 1280) frozen
        if protein_repr == "esm":
            self.esm_proj = nn.Linear(esm_mat.shape[1], ESM_PROJ)     # learnable ESM down-projection
        else:
            self.id_emb = nn.Embedding(esm_mat.shape[0], ESM_PROJ)    # random-init learned protein identity (no sequence)
        self.ctx_total = sum(CTX_DIMS.values())
        self.emb = nn.ModuleDict({f: nn.Embedding(n, CTX_DIMS[f]) for f, n in factor_sizes.items()})
        self.factors = list(CTX_DIMS)                       # fixed order
        self.enc = AdditiveEncoder(self.ctx_total) if arch == "additive" else Encoder(self.ctx_total)
        self.src_head = nn.Linear(EMB_DIM, EMB_DIM)
        self.tgt_head = nn.Linear(EMB_DIM, EMB_DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        # architecture B: auxiliary heads read z (per-protein-per-context, 128-d)
        self.pathway_head = nn.Linear(EMB_DIM, n_pathways) if n_pathways else None
        self.expr_head = nn.Linear(EMB_DIM, 1) if has_expr else None   # predicts protein's expression in the context

    def ctx_vec(self, fac_ids):
        parts = []
        for i, f in enumerate(self.factors):
            if self.zero == "all" or self.zero == f:
                parts.append(torch.zeros(fac_ids.shape[0], CTX_DIMS[f], device=fac_ids.device))
            else:
                parts.append(self.emb[f](fac_ids[:, i]))
        return torch.cat(parts, -1)

    def embed(self, prot_idx, fac_ids):
        prot_feat = self.esm_proj(self.esm[prot_idx]) if self.protein_repr == "esm" else self.id_emb(prot_idx)
        return self.enc(prot_feat, self.ctx_vec(fac_ids))

    def forward(self, tf_idx, tgt_idx, fac_ids):
        z_tf = self.embed(tf_idx, fac_ids)
        z_tg = self.embed(tgt_idx, fac_ids)
        return (self.src_head(z_tf) * self.tgt_head(z_tg)).sum(-1) + self.bias


def random_negs(ci, count, per_ctx, rng):
    d = per_ctx[ci]; src, tgt, posset = d["src"], d["tgt"], d["posset"]; out = []
    while len(out) < count:
        a = src[rng.integers(0, len(src), size=count * 2)]
        b = tgt[rng.integers(0, len(tgt), size=count * 2)]
        for x, y in zip(a, b):
            if x != y and (x, y) not in posset:
                out.append((ci, x, y))
                if len(out) >= count:
                    break
    return np.array(out, dtype=np.int64)


def sample_negs(pos_arr, tags, per_ctx, hard_pool, neg_mode, rng):
    """One negative per positive, resampled FRESH (call each epoch for train). hard = from precomputed pool."""
    neg = []
    for ci in range(len(tags)):
        npos = int((pos_arr[:, 0] == ci).sum())
        if npos == 0:
            continue
        pool = hard_pool.get(ci) if neg_mode == "hard" else None
        if pool is not None and len(pool):
            idx = rng.choice(len(pool), size=npos, replace=len(pool) < npos)
            neg.append(np.column_stack([np.full(npos, ci), pool[idx, 0], pool[idx, 1]]))
        else:
            neg.append(random_negs(ci, npos, per_ctx, rng))
    return np.vstack(neg).astype(np.int64)


def build_data(seed, neg_mode="random", labels="topk", exclude=None, edge_weight="inv_ctx", include=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    esm = torch.load(ESM_PT, map_location="cpu")
    prot_list = sorted(esm.keys())
    pidx = {p: i for i, p in enumerate(prot_list)}
    esm_mat = torch.stack([esm[p] for p in prot_list]).float()

    # discover ALL contexts (any cell type) that have the needed label file — integrates all built cell types
    label_file = "edges_cistarget.tsv" if labels == "cistarget" else "edges_topk.tsv"
    tags = sorted(p.name for p in NET.iterdir() if p.is_dir() and (p / label_file).exists())
    if include:                                          # keep ONLY contexts whose tag contains an included substring
        inc = [s for s in include.split(",") if s]
        tags = [t for t in tags if any(s in t for s in inc)]
    if exclude:                                          # drop contexts whose tag contains any excluded substring
        ex = [s for s in exclude.split(",") if s]
        tags = [t for t in tags if not any(s in t for s in ex)]
    fac_vocab = {f: {} for f in CTX_DIMS}
    for tag in tags:
        for f, v in parse_tag(tag).items():
            fac_vocab[f].setdefault(v, len(fac_vocab[f]))
    FAC = np.stack([[fac_vocab[f][parse_tag(tag)[f]] for f in CTX_DIMS] for tag in tags])  # (C,4)

    pos = []; per_ctx = {}
    for ci, tag in enumerate(tags):
        if labels == "cistarget":
            # motif-pruned regulons (all edges; already filtered, no top-k truncation)
            e = pd.read_csv(NET / tag / "edges_cistarget.tsv", sep="\t")
            e = e[e.tf.isin(pidx) & e.target.isin(pidx)]
        else:
            e = pd.read_csv(NET / tag / "edges_topk.tsv", sep="\t")
            e = e[e.tf.isin(pidx) & e.target.isin(pidx)]
            # positives = each TF's top-POS_TOPK targets by importance (independent of the k the file was written at)
            e = e.sort_values("importance", ascending=False).groupby("tf", sort=False).head(POS_TOPK)
        tf = e.tf.map(pidx).to_numpy(); tg = e.target.map(pidx).to_numpy()
        per_ctx[ci] = dict(posset=set(zip(tf.tolist(), tg.tolist())),
                           src=np.unique(tf), tgt=np.unique(tg))
        pos += [(ci, a, b) for a, b in zip(tf, tg)]
    pos = np.array(pos, dtype=np.int64)

    global_pos = {}
    for ci in range(len(tags)):
        for p in per_ctx[ci]["posset"]:
            global_pos.setdefault(p, set()).add(ci)
    all_pairs = list(global_pos.keys())

    # hard-negative POOL per context (precomputed ONCE, sampled fresh each epoch): pairs positive in another
    # context, endpoints present here, and OUTSIDE this context's top-PROTECT_TOPN/TF (no sub-threshold real edges).
    hard_pool = {}
    if neg_mode == "hard":
        for ci, tag in enumerate(tags):
            e = pd.read_csv(NET / tag / "edges.tsv", sep="\t")
            e = e[e.tf.isin(pidx) & e.target.isin(pidx)]
            top = e.sort_values("importance", ascending=False).groupby("tf", sort=False).head(PROTECT_TOPN)
            prot = set(zip(top.tf.map(pidx), top.target.map(pidx)))
            sset = set(per_ctx[ci]["src"].tolist()); tset = set(per_ctx[ci]["tgt"].tolist())
            cand = [p for p in all_pairs if p[0] in sset and p[1] in tset and p not in prot]
            hard_pool[ci] = np.array(cand, dtype=np.int64) if cand else np.empty((0, 2), np.int64)

    # split POSITIVES 80/10/10; fix val/test negatives ONCE (train negatives resampled per epoch in run_split)
    perm = rng.permutation(len(pos)); pos = pos[perm]
    n = len(pos); a, b = int(.8 * n), int(.9 * n)
    tr_pos, va_pos, te_pos = pos[:a], pos[a:b], pos[b:]

    # per-positive edge weight: upweight context-specific edges (inverse to # contexts an edge appears in),
    # normalized to mean 1 so the positive/negative loss balance is preserved.
    if edge_weight == "inv_ctx":
        tpw = np.array([1.0 / len(global_pos[(int(s), int(t))]) for _, s, t in tr_pos], dtype=np.float32)
        tpw /= tpw.mean()
    else:
        tpw = np.ones(len(tr_pos), dtype=np.float32)

    def make_xy(pos_a):
        neg = sample_negs(pos_a, tags, per_ctx, hard_pool, neg_mode, rng)
        X = np.vstack([pos_a, neg])
        y = np.concatenate([np.ones(len(pos_a)), np.zeros(len(neg))]).astype(np.float32)
        return X, y, FAC[X[:, 0]]

    sizes = {f: len(fac_vocab[f]) for f in CTX_DIMS}
    return dict(esm_mat=esm_mat, tags=tags, sizes=sizes, FAC=FAC, per_ctx=per_ctx,
                hard_pool=hard_pool, neg_mode=neg_mode, train_pos=tr_pos, train_pos_w=tpw,
                va=make_xy(va_pos), te=make_xy(te_pos))


def _eval(m, X, y, fac, device):
    with torch.no_grad():
        s = m(torch.tensor(X[:, 1], device=device), torch.tensor(X[:, 2], device=device),
              torch.tensor(fac, device=device)).cpu().numpy()
    return s


def run_split(d, zero, epochs, device, seed, protein_repr="esm", patience=10, arch="concat",
              aux_pathway=0.0, pathway_target=None, pathway_membership=None,
              aux_expression=0.0, expr_target=None, expr_mask=None, link_weight=1.0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + hash(str(zero)) % 997)
    n_pw = pathway_target.shape[1] if pathway_target is not None else 0
    m = LinkModel(d["esm_mat"], d["sizes"], zero=zero, protein_repr=protein_repr, arch=arch,
                  n_pathways=n_pw, has_expr=(expr_target is not None)).to(device)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none")        # per-sample -> apply edge weights below
    tr_pos, FAC, tags = d["train_pos"], d["FAC"], d["tags"]
    pw = d.get("train_pos_w", np.ones(len(tr_pos), dtype=np.float32))   # positive edge weights (negs = 1.0)
    # architecture-B auxiliary pathway loss — PER-PROTEIN-PER-CONTEXT (FULL model only; skip when context zeroed/blind).
    # For endpoint (protein p, context c): predict from z_{p,c} the activity (in c) of p's OWN pathways; push
    # non-member pathways toward 0 so the target is protein-specific (forces z to encode identity x context).
    use_pathway = aux_pathway > 0 and pathway_target is not None and pathway_membership is not None and zero is None
    use_expr = aux_expression > 0 and expr_target is not None and zero is None
    use_aux = use_pathway or use_expr
    if use_aux:
        FAC_t = torch.tensor(FAC, device=device)
    if use_pathway:
        A_t = torch.tensor(pathway_target, dtype=torch.float32, device=device)        # (n_ctx, Np) context activity
        M_t = torch.tensor(pathway_membership, dtype=torch.float32, device=device)     # (P, Np) protein->pathway membership
    if use_expr:
        E_t = torch.tensor(expr_target, dtype=torch.float32, device=device)           # (n_ctx, P) per-gene z-expression
        Me_t = torch.tensor(expr_mask, dtype=torch.float32, device=device)            # (n_ctx, P) 1 where gene measured
    Xva, yva, facva = d["va"]; Xte, yte, facte = d["te"]
    yva_t = torch.tensor(yva, device=device)
    bs = 8192
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_ep, best_state, no_improve = -1.0, 0, None, 0
    for ep in range(epochs):
        # FRESH negatives every epoch (train only); val/test stay fixed
        neg = sample_negs(tr_pos, tags, d["per_ctx"], d["hard_pool"], d["neg_mode"], rng)
        X = np.vstack([tr_pos, neg])
        y = np.concatenate([np.ones(len(tr_pos)), np.zeros(len(neg))]).astype(np.float32)
        fac = FAC[X[:, 0]]
        w = np.concatenate([pw, np.ones(len(neg), dtype=np.float32)])   # align with X = [pos, neg]
        Xt = torch.tensor(X, device=device); yt = torch.tensor(y, device=device); ft = torch.tensor(fac, device=device)
        wt = torch.tensor(w, device=device)
        m.train(); perm = torch.randperm(len(Xt), device=device); tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            logit = m(Xt[j, 1], Xt[j, 2], ft[j])
            loss = link_weight * (lossf(logit, yt[j]) * wt[j]).mean()   # link_weight=0 -> train on aux only
            if use_aux:                                     # per-protein-per-context aux losses on the batch's endpoints
                p_ids = torch.cat([Xt[j, 1], Xt[j, 2]])     # tf + target proteins
                c_ids = torch.cat([Xt[j, 0], Xt[j, 0]])     # their (shared per-edge) context index
                zc = m.embed(p_ids, FAC_t[c_ids])           # z_{p,c} (2B, 128), shared by both heads
                if use_pathway:
                    pr, tg, mb = m.pathway_head(zc), A_t[c_ids], M_t[p_ids]
                    l_mem = (mb * (pr - tg) ** 2).sum() / mb.sum().clamp(min=1)             # activity at p's pathways
                    l_non = ((1 - mb) * pr ** 2).sum() / (1 - mb).sum().clamp(min=1)        # 0 at non-member pathways
                    loss = loss + aux_pathway * (l_mem + l_non)
                if use_expr:
                    pe = m.expr_head(zc).squeeze(-1); te = E_t[c_ids, p_ids]; me = Me_t[c_ids, p_ids]
                    loss = loss + aux_expression * ((me * (pe - te) ** 2).sum() / me.sum().clamp(min=1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        m.eval()
        vs = _eval(m, Xva, yva, facva, device)
        vloss = float(lossf(torch.tensor(vs, device=device), yva_t).mean())   # reduction='none' -> mean here
        va_auc = float(roc_auc_score(yva, vs))
        hist["epoch"].append(ep + 1); hist["train_loss"].append(tot / nb)
        hist["val_loss"].append(vloss); hist["val_auc"].append(va_auc)
        if link_weight > 0:                                  # val-AUC early stopping only when link-pred is the objective
            if va_auc > best_auc + 1e-4:                     # (fair readout; loss overfits). link off -> keep final weights.
                best_auc, best_ep, no_improve = va_auc, ep + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    early stop @ epoch {ep + 1} (best val-AUC {best_auc:.3f} @ ep {best_ep}, patience {patience})", flush=True)
                    break
    if best_state is not None:
        m.load_state_dict(best_state)                         # restore best-val-AUC weights
    m.eval(); s = _eval(m, Xte, yte, facte, device)
    return m, dict(auc=float(roc_auc_score(yte, s)), ap=float(average_precision_score(yte, s)), best_epoch=best_ep), hist


def save_embeddings(m, d, device, out):
    m.eval(); tags = d["tags"]; FAC = d["FAC"]
    ids, names, vecs, vecs_blind = [], [], [], []
    with torch.no_grad():
        for ci, tag in enumerate(tags):
            nodes = np.union1d(d["per_ctx"][ci]["src"], d["per_ctx"][ci]["tgt"])
            nt = torch.tensor(nodes, device=device)
            fac = torch.tensor(np.tile(FAC[ci], (len(nodes), 1)), device=device)
            m.zero = None; z = m.embed(nt, fac).cpu().numpy()                # full (context-on)
            m.zero = "all"; z_blind = m.embed(nt, fac).cpu().numpy()         # context zeroed (ESM-only)
            m.zero = None
            ids += [tag] * len(nodes); names += nodes.tolist(); vecs.append(z); vecs_blind.append(z_blind)
    np.savez(out, context=np.array(ids), prot_idx=np.array(names),
             emb=np.vstack(vecs), emb_blind=np.vstack(vecs_blind))


def plot_curves(hist_full, hist_blind, run, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = hist_full["epoch"]; FULL, BLIND = "#1b9e77", "#d95f02"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, hist_full["train_loss"], color=FULL, label="full · train")
    ax[0].plot(ep, hist_full["val_loss"], color=FULL, ls="--", label="full · val")
    if hist_blind is not None:                              # blind ablation is optional (--blind)
        ax[0].plot(hist_blind["epoch"], hist_blind["train_loss"], color=BLIND, label="blind · train")
        ax[0].plot(hist_blind["epoch"], hist_blind["val_loss"], color=BLIND, ls="--", label="blind · val")
    ax[0].set(xlabel="epoch", ylabel="BCE loss", title=f"{run} — loss"); ax[0].legend(fontsize=8)
    ax[1].plot(ep, hist_full["val_auc"], color=FULL, label="full")
    if hist_blind is not None:
        ax[1].plot(hist_blind["epoch"], hist_blind["val_auc"], color=BLIND, label="blind")
    ax[1].axhline(0.5, color="grey", lw=0.8, ls=":")
    ax[1].set(xlabel="epoch", ylabel="validation AUC", title=f"{run} — val AUC"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg", choices=["random", "hard"], default="random",
                    help="hard = cross-context negatives (positive elsewhere, not here) — tests context-specificity")
    ap.add_argument("--labels", choices=["topk", "cistarget"], default="topk",
                    help="topk = GRNBoost2 top-k co-expression; cistarget = motif-pruned regulons")
    ap.add_argument("--protein-repr", choices=["esm", "id"], default="esm",
                    help="esm = frozen ESM projection (default); id = random-init learned protein-ID embedding (no sequence)")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated substrings; contexts whose tag contains any are dropped (e.g. 'tcell')")
    ap.add_argument("--include", default=None,
                    help="comma-separated substrings; keep ONLY contexts whose tag contains one (e.g. 'macrophage' for a fast dev set)")
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stop patience: stop if val-AUC doesn't improve for this many epochs (best weights restored)")
    ap.add_argument("--blind", action="store_true",
                    help="also train the context-blind ablation for the context-lift number (default OFF; embeddings don't need it)")
    ap.add_argument("--edge-weight", choices=["none", "inv_ctx"], default="none",
                    help="none = equal-weight positives (default). inv_ctx = upweight context-specific positive edges "
                         "(1/#contexts, mean-normalized) — tried in v13, inflated lift but HURT recovery, so not default")
    ap.add_argument("--aux-pathway", type=float, default=0.0,
                    help="architecture B: weight lambda on the auxiliary per-protein-per-context pathway-activity loss "
                         "(0 = off). Reads scenic/pathway_activity.tsv.")
    ap.add_argument("--aux-expression", type=float, default=0.0,
                    help="weight lambda on the auxiliary per-protein-per-context expression-prediction MSE loss "
                         "(0 = off). Reads scenic/expression_activity.tsv.")
    ap.add_argument("--link-weight", type=float, default=1.0,
                    help="weight on the SCENIC link-prediction loss (default 1.0). 0 = train the embedding on the aux "
                         "losses ONLY (no link-pred; early stopping disabled, test AUC then meaningless).")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = SEQ / "results" / args.run; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = build_data(args.seed, neg_mode=args.neg, labels=args.labels, exclude=args.exclude, edge_weight=args.edge_weight, include=args.include)
    print(f"data: {len(d['tags'])} contexts | train_pos={len(d['train_pos'])} "
          f"| labels={args.labels} | neg={args.neg} | edge_weight={args.edge_weight} (resampled/epoch) | device={device}", flush=True)

    pw_target = pw_memb = None
    if args.aux_pathway > 0:                               # architecture B: per-context activity + per-protein membership
        pa = pd.read_csv(SEQ / "scenic/pathway_activity.tsv", sep="\t", index_col=0)
        missing = [t for t in d["tags"] if t not in pa.index]
        if missing:
            print(f"WARN {len(missing)}/{len(d['tags'])} contexts missing pathway activity (zero-filled), e.g. {missing[:3]}", flush=True)
        pw_target = pa.reindex(d["tags"]).fillna(0.0).to_numpy(dtype=np.float32)          # (n_ctx, Np) aligned to tags
        prot_list = sorted(torch.load(ESM_PT, map_location="cpu").keys())                 # SAME order as esm_mat rows
        pidx = {g: i for i, g in enumerate(prot_list)}
        gmt = {}
        for ln in open("mlp_mods/reactome/ReactomePathways.gmt"):
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 4:
                gmt[f[0]] = set(f[2:])
        pw_memb = np.zeros((len(prot_list), pa.shape[1]), dtype=np.float32)               # (P, Np) aligned to pa columns
        for k, name in enumerate(pa.columns):
            for g in gmt.get(name, ()):
                if g in pidx:
                    pw_memb[pidx[g], k] = 1.0
        print(f"aux-pathway per-protein-per-context ON (lambda={args.aux_pathway}): activity {pw_target.shape}, "
              f"membership {pw_memb.shape}, mean pathways/protein={pw_memb.sum(1).mean():.1f}", flush=True)

    ex_target = ex_mask = None
    if args.aux_expression > 0:                            # per-protein-per-context expression labels (n_ctx x P)
        ea = pd.read_csv(SEQ / "scenic/expression_activity.tsv", sep="\t", index_col=0).reindex(d["tags"])
        prot_list = sorted(torch.load(ESM_PT, map_location="cpu").keys()); pidx = {g: i for i, g in enumerate(prot_list)}
        common = [g for g in ea.columns if g in pidx]
        sub = ea[common].to_numpy(dtype=np.float32)                                   # (n_ctx, len(common))
        pos_cols = np.array([pidx[g] for g in common])
        ex_target = np.zeros((len(d["tags"]), len(prot_list)), dtype=np.float32)
        ex_mask = np.zeros_like(ex_target)
        ex_target[:, pos_cols] = np.nan_to_num(sub)
        ex_mask[:, pos_cols] = (~np.isnan(sub)).astype(np.float32)
        print(f"aux-expression ON (lambda={args.aux_expression}): expr {ex_target.shape}, "
              f"genes measured={len(common)}, mean measured/context={ex_mask.sum(1).mean():.0f}", flush=True)

    m_full, full, h_full = run_split(d, None, args.epochs, device, args.seed, args.protein_repr, args.patience,
                                     aux_pathway=args.aux_pathway, pathway_target=pw_target, pathway_membership=pw_memb,
                                     aux_expression=args.aux_expression, expr_target=ex_target, expr_mask=ex_mask,
                                     link_weight=args.link_weight)
    print(f"FULL   test AUC={full['auc']:.3f} AP={full['ap']:.3f}  (protein_repr={args.protein_repr}, best_ep={full['best_epoch']})", flush=True)
    # BLIND run is only for the context-lift number; embeddings come from m_full. Opt-in (--blind); default skip.
    blind, h_blind, lift = None, None, None
    if args.blind:
        _, blind, h_blind = run_split(d, "all", args.epochs, device, args.seed, args.protein_repr, args.patience)
        lift = dict(auc=full["auc"] - blind["auc"], ap=full["ap"] - blind["ap"])
        print(f"BLIND  test AUC={blind['auc']:.3f} AP={blind['ap']:.3f} | CONTEXT-LIFT dAUC={lift['auc']:+.3f} dAP={lift['ap']:+.3f}", flush=True)

    img = SEQ / "images"; img.mkdir(parents=True, exist_ok=True)
    tag = args.run.replace("link_", "")                    # loss-curve naming convention: <tag>_loss_curve.png
    loss_png = img / f"{tag}_loss_curve.png"
    plot_curves(h_full, h_blind, args.run, loss_png)
    cf = pd.DataFrame({"epoch": h_full["epoch"], "full_train_loss": h_full["train_loss"],
                       "full_val_loss": h_full["val_loss"], "full_val_auc": h_full["val_auc"]})
    if h_blind is not None:                                 # blind may stop at a different epoch -> merge on epoch
        cb = pd.DataFrame({"epoch": h_blind["epoch"], "blind_train_loss": h_blind["train_loss"],
                           "blind_val_loss": h_blind["val_loss"], "blind_val_auc": h_blind["val_auc"]})
        cf = cf.merge(cb, on="epoch", how="outer")
    cf.to_csv(out / "curve.csv", index=False)
    print(f"curve -> {loss_png} , {out / 'curve.csv'}", flush=True)

    save_embeddings(m_full, d, device, out / "embeddings.npz")
    json.dump(dict(full=full, blind=blind, context_lift=lift, neg=args.neg, labels=args.labels,
                   protein_repr=args.protein_repr, edge_weight=args.edge_weight,
                   aux_pathway=args.aux_pathway, n_pathways=(0 if pw_target is None else pw_target.shape[1]),
                   aux_expression=args.aux_expression, resample_neg_per_epoch=True,
                   n_contexts=len(d["tags"]), n_train_pos=len(d["train_pos"]), emb_dim=EMB_DIM,
                   epochs=args.epochs, seconds=round(time.time() - t0, 1),
                   history=dict(full=h_full, blind=h_blind)),
              open(out / "metrics.json", "w"), indent=2)
    print(f"saved -> {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
