"""CONTEXT-CONDITIONED joint embedding: joint_embed.py + an additive, factorized context
vector added to every node's input, so the same protein can sit in different places per context.

Each network tag has a context = (disease, tissue, cell_type, state). We learn ONE embedding table per
FACTOR (disease/tissue/cell/state), zero-initialized, and add their sum to the node-identity input of
that network's forward pass:  h0 = X_identity + ctx(disease)+ctx(tissue)+ctx(cell)+ctx(state).
Factorized (not per-network) so strength is shared (the 'macrophage' vector is fit from all macrophage
nets). The context is REWARDED in the loss by an auxiliary context-classification term: linear heads
predict each network's (disease, tissue, cell, state) from its node embeddings, so the geometry (and
hence the context vector) is pushed to encode context instead of being shrunk to ~0 (link prediction
alone gives it no gradient). Total loss = link-prediction BCE + W_RECON*weight-recon + W_CTX*context-CE
+ L2_CTX*||context||. Everything else (shared encoder, Jacobian influence, per-node shift) matches
joint_embed.py.

CAVEAT: disease and tissue are collinear here (UC↔colon, ILD↔lung) so those two factors are NOT
separately identifiable; a 'disease' shift may be carrying study/batch. Validate against controls.

Networks are READ from results/<src_name>/networks/ ; outputs written to results/<out_name>/.
Run:
  .venv/bin/python mlp_mods/de_ppi/joint_embed_context.py \
      --src-name crohn_alzheimer_ild_uc_embedding_expressed --out-name crohn_alzheimer_ild_uc_embedding_context
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HERE
from embed_influence import Encoder, BilinearDecoder, WeightHead, DIM, LAYERS, EPOCHS, LR, NEG_RATIO, HOLDOUT, SEED
from joint_embed import Net, W_RECON
sys.path.insert(0, str(Path("mlp_mods/de_ppi") / "influence_analysis"))
from _layout import tag_celltype, tag_tissue, tag_disease, tag_state

L2_CTX = 1e-3        # mild L2 on context embeddings (don't suppress them now that a term rewards them)
W_CTX = 1.0          # weight of the context-classification loss (rewards using context)
N_CTX_SAMPLE = 256   # nodes sampled per network per epoch for the context-classification loss


def factors_of(tag):
    state = "healthy" if tag.startswith("healthy_") else (tag_state(tag) or "none")
    return {"disease": tag_disease(tag), "tissue": tag_tissue(tag),
            "cell": tag_celltype(tag), "state": state}


class Context(nn.Module):
    """Additive factorized context: one zero-init embedding table per factor; vec(tag) = sum of levels."""
    def __init__(self, levels: dict, dim: int):
        super().__init__()
        self.emb = nn.ModuleDict({f: nn.Embedding(len(v), dim) for f, v in levels.items()})
        for e in self.emb.values():
            nn.init.zeros_(e.weight)
        self.index = {f: {lv: i for i, lv in enumerate(v)} for f, v in levels.items()}

    def vec(self, factors: dict, device):
        v = 0
        for f, lv in factors.items():
            v = v + self.emb[f](torch.tensor(self.index[f][lv], device=device))
        return v                                            # (dim,)

    def l2(self):
        return sum((e.weight ** 2).sum() for e in self.emb.values())


def main(src_name, out_name, dim, layers, epochs, lr, neg_ratio, holdout, seed,
         factor_keys=("disease", "tissue", "cell", "state")) -> int:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = HERE / "results" / src_name / "networks"
    tags = sorted(p.name for p in root.iterdir() if (p / "network_nodes.tsv").exists())

    node_type, order = {}, []
    for t in tags:
        nd = pd.read_csv(root / t / "network_nodes.tsv", sep="\t", keep_default_na=False)
        for nid, nt in zip(nd["node_id"], nd["node_type"]):
            if nid not in node_type:
                node_type[nid] = nt; order.append(nid)
    idx = {g: i for i, g in enumerate(order)}
    n = len(order)
    nets = [Net(t, idx, device, root / t) for t in tags]
    tag_factors = {t: {f: factors_of(t)[f] for f in factor_keys} for t in tags}
    levels = {f: sorted({tag_factors[t][f] for t in tags}) for f in factor_keys}
    print(f"universe N={n} | {len(tags)} networks | context levels: "
          + ", ".join(f"{f}={len(v)}" for f, v in levels.items()) + f" | device={device}", flush=True)

    model = Encoder(n, dim, layers).to(device)
    dec = BilinearDecoder(dim).to(device)
    whead = WeightHead(dim).to(device)
    ctx = Context(levels, dim).to(device)
    heads = nn.ModuleDict({f: nn.Linear(dim, len(levels[f])) for f in levels}).to(device)  # context classifier
    opt = torch.optim.Adam(list(model.parameters()) + list(dec.parameters()) + list(whead.parameters())
                           + list(ctx.parameters()) + list(heads.parameters()), lr=lr)
    cvec = lambda t: ctx.vec(tag_factors[t], device).unsqueeze(0)         # (1, dim)
    present_idx = {net.tag: np.array([idx[g] for g in net.present]) for net in nets}
    tgt_lvl = {(t, f): ctx.index[f][tag_factors[t][f]] for t in tags for f in levels}

    splits = []
    for net in nets:
        perm = rng.permutation(len(net.pos_src)); nh = int(len(perm) * holdout)
        splits.append((perm[nh:], perm[:nh]))

    print("training context-conditioned shared encoder ...", flush=True)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        total = 0.0
        for net, (keep, _) in zip(nets, splits):
            z = model(net.A, x=model.x + cvec(net.tag), w_feat=net.w_feat)
            ts = torch.tensor(net.pos_src[keep], device=device)
            td = torch.tensor(net.pos_dst[keep], device=device)
            ns = ts.repeat(neg_ratio)
            nd = torch.randint(0, n, (len(td) * neg_ratio,), device=device)
            logits = torch.cat([dec(z[ts], z[td]), dec(z[ns], z[nd])])
            labels = torch.cat([torch.ones(len(td), device=device), torch.zeros(len(td) * neg_ratio, device=device)])
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            wr = F.mse_loss(whead(z[ts], z[td]), net.pos_w[keep])
            # context-classification: predict the network's disease/tissue/cell/state from node embeddings
            pi = present_idx[net.tag]
            samp = torch.tensor(rng.choice(pi, size=min(N_CTX_SAMPLE, len(pi)), replace=False), device=device)
            ce = 0.0
            for f in levels:
                tgt = torch.full((len(samp),), tgt_lvl[(net.tag, f)], device=device, dtype=torch.long)
                ce = ce + F.cross_entropy(heads[f](z[samp]), tgt)
            total = total + bce + W_RECON * wr + W_CTX * ce
        total = total + L2_CTX * ctx.l2()
        total.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  loss {float(total.detach()):.4f}", flush=True)

    from sklearn.metrics import roc_auc_score
    model.eval()
    Z = {}
    with torch.no_grad():
        for net, (_, hold) in zip(nets, splits):
            z = model(net.A, x=model.x + cvec(net.tag), w_feat=net.w_feat); Z[net.tag] = z.cpu().numpy()
            if len(hold):
                hs = torch.tensor(net.pos_src[hold], device=device); hd = torch.tensor(net.pos_dst[hold], device=device)
                ndg = torch.randint(0, n, (len(hd),), device=device)
                pos = torch.sigmoid(dec(z[hs], z[hd])).cpu().numpy(); neg = torch.sigmoid(dec(z[hs], z[ndg])).cpu().numpy()
                y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
                print(f"  held-out AUC [{net.tag}]: {roc_auc_score(y, np.concatenate([pos, neg])):.3f}", flush=True)

    # context-vector magnitudes (how much context was actually used, per factor level)
    print("\ncontext-vector L2 norms by factor/level:", flush=True)
    for f, lv in levels.items():
        norms = ctx.emb[f].weight.detach().norm(dim=1).cpu().numpy()
        print(f"  {f}: " + ", ".join(f"{l}={nrm:.2f}" for l, nrm in zip(lv, norms)), flush=True)

    # Jacobian influence with context as a fixed additive offset (differentiate w.r.t. node identity)
    print("\ncomputing per-network Jacobian influence ...", flush=True)
    infl = {}
    for net in nets:
        cv = cvec(net.tag).detach()
        x = model.x.detach().clone().requires_grad_(True)
        z = model(net.A, x=x + cv, w_feat=net.w_feat)
        r = z[torch.tensor(net.target_idx, device=device)].sum(0)
        sq = torch.zeros(n, device=device)
        for c in range(r.shape[0]):
            g, = torch.autograd.grad(r[c], x, retain_graph=(c < r.shape[0] - 1))
            sq += (g ** 2).sum(1)
        infl[net.tag] = torch.sqrt(sq).detach().cpu().numpy()

    res = HERE / "results" / out_name; res.mkdir(parents=True, exist_ok=True)
    present = {net.tag: set(net.present) for net in nets}
    out = pd.DataFrame({"node_id": order, "node_type": [node_type[g] for g in order]})
    for net in nets:
        t = net.tag
        out[f"present_{t}"] = out.node_id.isin(present[t]).astype(int)
        v = infl[t].copy(); v[out[f"present_{t}"].to_numpy() == 0] = np.nan
        out[f"influence_{t}"] = v
        out[f"influence_{t}_rank"] = pd.Series(v).rank(method="first", ascending=False).astype("Int64")
    out.sort_values(f"influence_{nets[0].tag}", ascending=False).to_csv(res / "joint_influence.tsv", sep="\t", index=False)

    shift = pd.DataFrame({"node_id": order, "node_type": [node_type[g] for g in order]})
    for net in nets:
        shift[f"present_{net.tag}"] = shift.node_id.isin(present[net.tag]).astype(int)
    pair_cols = []
    for a, b in combinations([net.tag for net in nets], 2):
        dd = np.linalg.norm(Z[a] - Z[b], axis=1)
        both = shift[f"present_{a}"].to_numpy() & shift[f"present_{b}"].to_numpy()
        col = f"shift_{a}_{b}"; shift[col] = np.where(both == 1, np.round(dd, 4), np.nan); pair_cols.append(col)
    shift["shift_total"] = shift[pair_cols].sum(axis=1, skipna=True).round(4)
    shift["present_all"] = (shift[[f"present_{net.tag}" for net in nets]].sum(1) == len(nets)).astype(int)
    shift.sort_values("shift_total", ascending=False).to_csv(res / "embedding_shift.tsv", sep="\t", index=False)

    np.savez_compressed(res / "embeddings.npz",
                        node_id=np.array(order, dtype=object),
                        node_type=np.array([node_type[g] for g in order], dtype=object),
                        tags=np.array([net.tag for net in nets], dtype=object),
                        Z=np.stack([Z[net.tag] for net in nets]),
                        present=np.stack([np.array([g in present[net.tag] for g in order]) for net in nets]))
    print(f"\nwrote {res/'joint_influence.tsv'}\nwrote {res/'embedding_shift.tsv'}\nwrote {res/'embeddings.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-name", default="crohn_alzheimer_ild_uc_embedding_expressed",
                    help="results/<src_name>/networks/ : where the per-tag networks are read from")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_context")
    ap.add_argument("--dim", type=int, default=DIM); ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO); ap.add_argument("--holdout", type=float, default=HOLDOUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--factors", nargs="+", default=["disease", "tissue", "cell", "state"],
                    choices=["disease", "tissue", "cell", "state"],
                    help="which context factors to condition + classify on (e.g. --factors disease)")
    a = ap.parse_args()
    raise SystemExit(main(a.src_name, a.out_name, a.dim, a.layers, a.epochs, a.lr,
                          a.neg_ratio, a.holdout, a.seed, tuple(a.factors)))
