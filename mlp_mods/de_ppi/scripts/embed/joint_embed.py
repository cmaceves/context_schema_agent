"""JOINT embedding of several PPI networks in ONE shared space (one encoder, a forward pass each).

A multi-network extension of embed_influence.py. Instead of training one encoder per network
(which gives embeddings comparable only up to a rotation), this shares ONE input embedding table
+ ONE encoder + ONE decoder across all networks, with a separate FORWARD PASS on each network's
graph and the link-prediction loss SUMMED over all of them. The shared weights + shared input
table anchor every network's embedding to a common geometry, so:

  * per-network influence on each network's dysregulated set is on a COMMON SCALE, and
  * the per-node EMBEDDING SHIFT ||Z_a[i] - Z_b[i]|| between two networks is meaningful — how
    differently node i's neighbourhood is wired/weighted between them (only interpretable
    because the spaces are shared).

Each network is a TAG = a subdir under results/<out_name>/networks/<tag>/ holding
network_nodes.tsv + network_edges.tsv. The node universe is the UNION over tags; a node absent
from a network is isolated there (shift only reported where present).

Output (results/<out_name>/, or --res-name):
  embeddings.npz        node_id, node_type, tags, Z (n_tags, N, dim), present (n_tags, N)
  joint_influence.tsv   node_id, node_type, present_<tag>, influence_<tag>, influence_<tag>_rank
  embedding_shift.tsv   node_id, present_<tag>..., shift_<a>_<b>... (euclidean), shift_total, present_all

(Reconstructed from bytecode after the original joint_embed_influence.py was removed; functionally
faithful, but a future retrain may differ slightly from embeddings trained by the original.)

Run with .venv (the per-tag networks must already be staged under results/<out_name>/networks/):
  .venv/bin/python mlp_mods/de_ppi/joint_embed.py --out-name crohn_alzheimer_ild_uc_embedding_expressed
  # --tags limits/orders the networks; default = all tag subdirs found
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import HERE
from embed_influence import (Encoder, BilinearDecoder, WeightHead, build_operator,
                             jacobian_influence, DIM, LAYERS, EPOCHS, LR, NEG_RATIO, HOLDOUT, SEED)

W_RECON = 1.0     # weight of the edge-weight reconstruction loss (ties embedding to rank-shift weights)


def networks_root(out_name: str) -> Path:
    return HERE / "results" / out_name / "networks"


def discover_tags(out_name: str) -> list[str]:
    return sorted(p.name for p in networks_root(out_name).iterdir() if (p / "network_nodes.tsv").exists())


class Net:
    """One network (tag) mapped onto the shared node universe."""

    def __init__(self, tag: str, idx: dict[str, int], device, ndir: Path):
        nodes = pd.read_csv(ndir / "network_nodes.tsv", sep="\t", keep_default_na=False)
        edges = pd.read_csv(ndir / "network_edges.tsv", sep="\t", keep_default_na=False)
        self.tag = tag
        n = len(idx)
        # presence over the universe
        self.present = np.zeros(n, dtype=bool)
        node_pos = nodes["node_id"].map(idx).to_numpy()
        self.present[node_pos] = True
        # sender weights placed on the universe (absent nodes -> 1.0, the neutral self-loop default)
        sw = np.ones(n)
        sw[node_pos] = nodes["sender_weight"].astype(float).to_numpy()
        self.A = build_operator(edges, idx, device, self_weight=sw)
        self.w_feat = torch.tensor(np.log(sw), dtype=torch.float32, device=device).unsqueeze(1)
        self.pos_src = edges["source"].map(idx).to_numpy()
        self.pos_dst = edges["target"].map(idx).to_numpy()
        self.pos_w = torch.tensor(np.log(edges["weight"].astype(float).to_numpy()),
                                  dtype=torch.float32, device=device)
        # dysregulated set = directioned (DE/literature) nodes UNION metabolite sinks, as universe indices
        dys = (nodes["direction"].astype(str).str.len() > 0) | (nodes["node_type"] == "metabolite")
        self.target_idx = nodes["node_id"][dys.to_numpy()].map(idx).to_numpy()
        self.n_target = int(len(self.target_idx))


def main(out_name, tags, dim, layers, epochs, lr, neg_ratio, holdout, seed,
         mean_readout=False, res_name=None) -> int:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tags = tags or discover_tags(out_name)
    if not tags:
        raise SystemExit("no network tag subdirs under " + str(networks_root(out_name)))
    root = networks_root(out_name)

    # node universe = UNION of node_ids over tags; node_type by first occurrence
    node_type: dict[str, str] = {}
    for t in tags:
        nd = pd.read_csv(root / t / "network_nodes.tsv", sep="\t", keep_default_na=False)
        for nid, nt in zip(nd["node_id"], nd["node_type"]):
            node_type.setdefault(nid, nt)
    order = list(node_type)
    idx = {g: i for i, g in enumerate(order)}
    N = len(order)
    nets = [Net(t, idx, device, root / t) for t in tags]
    print(f"universe N={N} | tags: {', '.join(tags)} | device={device}", flush=True)

    model = Encoder(N, dim, layers).to(device)
    dec = BilinearDecoder(dim).to(device)
    whead = WeightHead(dim).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(dec.parameters()) + list(whead.parameters()), lr=lr)

    # per-network held-out split (for AUC)
    splits = []
    for net in nets:
        perm = rng.permutation(len(net.pos_src))
        nh = int(len(perm) * holdout)
        splits.append((perm[nh:], perm[:nh]))   # (keep, hold)

    print("training shared encoder (link prediction + edge-weight reconstruction, summed over networks)...",
          flush=True)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = torch.tensor(0.0, device=device); mse_sum = 0.0
        for net, (keep, _) in zip(nets, splits):
            z = model(net.A, w_feat=net.w_feat)
            ts = torch.tensor(net.pos_src[keep], device=device)
            td = torch.tensor(net.pos_dst[keep], device=device)
            ns = ts.repeat(neg_ratio)
            ndst = torch.randint(0, N, (len(td) * neg_ratio,), device=device)   # corrupt target
            pos = dec(z[ts], z[td]); neg = dec(z[ns], z[ndst])
            logits = torch.cat([pos, neg])
            labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
            lp = F.binary_cross_entropy_with_logits(logits, labels)
            wr = F.mse_loss(whead(z[ts], z[td]), net.pos_w[torch.tensor(keep, device=device)])
            loss = loss + lp + W_RECON * wr
            mse_sum += float(wr.detach())
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  summed loss {loss.item():.4f}  (weight-recon MSE {mse_sum:.4f})", flush=True)

    # held-out AUC per network
    from sklearn.metrics import roc_auc_score
    model.eval()
    with torch.no_grad():
        for net, (_, hold) in zip(nets, splits):
            if len(hold) == 0:
                continue
            z = model(net.A, w_feat=net.w_feat)
            hs = torch.tensor(net.pos_src[hold], device=device)
            hd = torch.tensor(net.pos_dst[hold], device=device)
            ndst = torch.randint(0, N, (len(hd),), device=device)
            pos = torch.sigmoid(dec(z[hs], z[hd])).cpu().numpy()
            neg = torch.sigmoid(dec(z[hs], z[ndst])).cpu().numpy()
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            print(f"  held-out AUC [{net.tag}]: {roc_auc_score(y, np.concatenate([pos, neg])):.3f}", flush=True)

    # per-network embeddings Z
    model.eval()
    with torch.no_grad():
        Z = np.stack([model(net.A, w_feat=net.w_feat).detach().cpu().numpy() for net in nets])  # (T, N, dim)
    present = np.stack([net.present for net in nets])                                            # (T, N)

    # per-network Jacobian influence on each net's own dysregulated set
    print(f"computing per-network Jacobian influence ({'mean over set' if mean_readout else 'sum over set'})...",
          flush=True)
    out = pd.DataFrame({"node_id": order, "node_type": [node_type[g] for g in order]})
    for ti, net in enumerate(nets):
        infl = jacobian_influence(net.A, model, net.target_idx, device, net.w_feat)
        if mean_readout and net.n_target:
            infl = infl / net.n_target
        out[f"present_{net.tag}"] = present[ti]
        out[f"influence_{net.tag}"] = infl
        out[f"influence_{net.tag}_rank"] = pd.Series(infl).rank(ascending=False, method="first").astype("Int64")

    res = HERE / "results" / (res_name or out_name); res.mkdir(parents=True, exist_ok=True)
    out.to_csv(res / "joint_influence.tsv", sep="\t", index=False)

    # per-node embedding shift between every tag pair (euclidean, where present in both)
    sh = pd.DataFrame({"node_id": order})
    for ti, net in enumerate(nets):
        sh[f"present_{net.tag}"] = present[ti]
    pair_cols = []
    for a, b in combinations(range(len(nets)), 2):
        col = f"shift_{nets[a].tag}_{nets[b].tag}"
        d = np.linalg.norm(Z[a] - Z[b], axis=1)
        d[~(present[a] & present[b])] = np.nan
        sh[col] = np.round(d, 4); pair_cols.append(col)
    sh["shift_total"] = sh[pair_cols].sum(axis=1, skipna=True).round(4)
    sh["present_all"] = present.all(axis=0)
    sh.to_csv(res / "embedding_shift.tsv", sep="\t", index=False)

    np.savez_compressed(res / "embeddings.npz",
                        node_id=np.array(order, dtype=object),
                        node_type=np.array([node_type[g] for g in order], dtype=object),
                        tags=np.array([net.tag for net in nets], dtype=object),
                        Z=Z, present=present)
    print(f"\nwrote {res/'joint_influence.tsv'}\nwrote {res/'embedding_shift.tsv'}\nwrote {res/'embeddings.npz'}",
          flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="results/<out_name>/ : reads networks/<tag>/, writes joint_influence + shift")
    ap.add_argument("--out-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--tags", nargs="+", default=None, help="subset/order of network tags (default: all tag subdirs found)")
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    ap.add_argument("--holdout", type=float, default=HOLDOUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mean", action="store_true", dest="mean_readout",
                    help="divide each network's influence by its own dysregulated-set size (mean readout) instead of sum")
    ap.add_argument("--res-name", default=None,
                    help="write outputs to results/<res_name>/ (networks still READ from out_name; for variants)")
    a = ap.parse_args()
    raise SystemExit(main(a.out_name, a.tags, a.dim, a.layers, a.epochs, a.lr,
                          a.neg_ratio, a.holdout, a.seed, a.mean_readout, a.res_name))
