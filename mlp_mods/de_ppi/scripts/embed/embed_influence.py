"""(disease, cell type) PPI EMBEDDING influence — learned message-passing encoder + Jacobian.

A learned-embedding complement to build_literature_weighted_influence.py's analytic P^k
reach. Instead of propagating a target indicator through a fixed operator, this trains a
2-layer weighted *directed* message-passing encoder on the per-build PPI (unsupervised
directed link prediction, asymmetric bilinear decoder + negative sampling), then measures
how much each node's input drives the embeddings of the DYSREGULATED SET via the encoder
Jacobian:

    influence(i) = || d r / d x_i ||_F ,   r = sum_{j in dysregulated set} z_j

i.e. the Frobenius norm of the Jacobian of the dysregulated-set embedding readout w.r.t.
node i's input features. A node only scores if message passing carries its input to a
dysregulated node within the encoder's receptive field (here 2 hops + self-loops). This is
the all-pairs Jacobian (the retired R-GCN encoder's idea) collapsed onto the dysregulated
set, parameterized by --build like the rest of de_ppi.

Reuses the network already built by build_literature_weighted_influence.py:
  results/<build>/networks/network_nodes.tsv   (node_id, node_type, source, direction, ...)
  results/<build>/networks/network_edges.tsv   (source, target, weight, ...)
Dysregulated set = nodes with a non-empty `direction` (DE/literature proteins) UNION all
metabolite sinks — the same target set m the analytic builder propagates.

Output (results/<build>/): embedding_influence.tsv, one row per node:
  node_id, node_type, source, direction, is_dysregulated,
  embed_influence, embed_influence_rank
(is_dysregulated flags nodes already in the target set; filter them out for control points
that aren't themselves moving, cf. the analytic reach guidance in CAVEATS.md.)

CAVEAT: this is a *learned* embedding, so it is non-deterministic up to the seed and depends
on training quality (held-out link-prediction AUC is printed). Unlike the analytic reach it
is unsigned and carries no activation/inhibition direction.

Run with .venv (needs torch; build_literature_weighted_influence.py must have run first):
  .venv/bin/python mlp_mods/de_ppi/embed_influence.py --build macrophage_crohn
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_build

# encoder / training defaults
DIM = 64          # embedding dimension
LAYERS = 2        # message-passing layers (also the Jacobian receptive field, +self-loops)
EPOCHS = 300
LR = 1e-2
NEG_RATIO = 1     # negative edges sampled per positive edge
HOLDOUT = 0.10    # fraction of edges held out for AUC
SEED = 3


def build_operator(edges: pd.DataFrame, idx: dict[str, int], device,
                   self_weight: np.ndarray | None = None) -> torch.Tensor:
    """Row-normalized directed aggregation operator with self-loops.

    A[target, source] = edge weight, so (A @ X)[j] aggregates j's in-neighbours (predecessors)
    -> signal flows source -> target, matching message passing along the directed arc. Self-loops
    let a node keep its own features; rows are normalized to sum to 1 (receiver-side).

    self_weight gates the self-loop: default 1.0 (state-invariant identity term). Passing each
    node's SENDER weight w(i) instead makes a node's own dysregulation enter its own embedding
    (the self-loop now carries w(i)*X[i]), so a DE gene's own embedding shifts with its own DE.
    Returns a sparse (N, N) tensor.
    """
    n = len(idx)
    src = edges["source"].map(idx).to_numpy()
    dst = edges["target"].map(idx).to_numpy()
    w = edges["weight"].astype(float).to_numpy()
    self_w = np.ones(n) if self_weight is None else np.asarray(self_weight, dtype=float)
    # rows = target (receiver), cols = source (sender); + self-loops (gated by self_w)
    rows = np.concatenate([dst, np.arange(n)])
    cols = np.concatenate([src, np.arange(n)])
    vals = np.concatenate([w, self_w])
    deg = np.zeros(n)
    np.add.at(deg, rows, vals)                      # in-strength per receiver
    vals = vals / np.where(deg[rows] > 0, deg[rows], 1.0)
    i = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    v = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, (n, n)).coalesce().to(device)


class Encoder(nn.Module):
    """Featureless 2-layer weighted directed message-passing encoder.

    Input is a learnable per-node embedding table X (the node identity we differentiate the
    Jacobian against); each layer is  H' = relu(A @ H W + H W_self)  (self-loops already fold
    into A, the extra W_self keeps a residual identity path).
    """

    def __init__(self, n: int, dim: int, layers: int):
        super().__init__()
        self.x = nn.Parameter(torch.randn(n, dim) * 0.1)
        self.lin = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        self.self_lin = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        self.wlin = nn.Linear(1, dim, bias=False)        # injects each node's (log) sender weight into its input

    def forward(self, A: torch.Tensor, x: torch.Tensor | None = None,
                w_feat: torch.Tensor | None = None) -> torch.Tensor:
        h = self.x if x is None else x
        if w_feat is not None:                           # network-specific: node's own rank-shift enters its input
            h = h + self.wlin(w_feat)
        for k, (lin, slin) in enumerate(zip(self.lin, self.self_lin)):
            h = torch.sparse.mm(A, lin(h)) + slin(h)
            if k < len(self.lin) - 1:
                h = F.relu(h)
        return h


class BilinearDecoder(nn.Module):
    """Asymmetric edge scorer: score(i->j) = z_i^T R z_j (R full -> direction-aware)."""

    def __init__(self, dim: int):
        super().__init__()
        self.r = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)

    def forward(self, zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
        return ((zs @ self.r) * zt).sum(-1)


class WeightHead(nn.Module):
    """Regresses an edge's (log) weight from its endpoint embeddings, so the encoder is trained to
    ENCODE the rank-shift weight (not just edge existence). score(i->j) -> predicted log-weight."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([zs, zt], dim=-1)).squeeze(-1)


def train(A, model, dec, pos_src, pos_dst, n, device, epochs, lr, neg_ratio, holdout, rng,
          whead=None, pos_w=None, w_feat=None, w_recon=1.0):
    # held-out split for AUC
    perm = rng.permutation(len(pos_src))
    n_hold = int(len(perm) * holdout)
    hold, keep = perm[:n_hold], perm[n_hold:]
    tr_s = torch.tensor(pos_src[keep], device=device)
    tr_d = torch.tensor(pos_dst[keep], device=device)
    pw = pos_w[torch.tensor(keep, device=device)] if pos_w is not None else None
    params = list(model.parameters()) + list(dec.parameters()) + (list(whead.parameters()) if whead else [])
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(A, w_feat=w_feat)
        ns = tr_s.repeat(neg_ratio)
        nd = torch.randint(0, n, (len(tr_d) * neg_ratio,), device=device)   # corrupt target
        pos = dec(z[tr_s], z[tr_d])
        neg = dec(z[ns], z[nd])
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if whead is not None and pw is not None:
            loss = loss + w_recon * F.mse_loss(whead(z[tr_s], z[tr_d]), pw)   # reconstruct edge (log) weight
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}", flush=True)
    # held-out AUC
    from sklearn.metrics import roc_auc_score
    model.eval()
    with torch.no_grad():
        z = model(A, w_feat=w_feat)
        hs = torch.tensor(pos_src[hold], device=device)
        hd = torch.tensor(pos_dst[hold], device=device)
        nd = torch.randint(0, n, (len(hd),), device=device)
        pos = torch.sigmoid(dec(z[hs], z[hd])).cpu().numpy()
        neg = torch.sigmoid(dec(z[hs], z[nd])).cpu().numpy()
    if n_hold:
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        auc = roc_auc_score(y, np.concatenate([pos, neg]))
        print(f"  held-out link-prediction AUC: {auc:.3f}  ({n_hold} edges)", flush=True)


def jacobian_influence(A, model, target_idx: np.ndarray, device, w_feat=None) -> np.ndarray:
    """influence(i) = || d (sum_{j in target} z_j) / d x_i ||_F over each node's input embedding.

    One backward pass per output coordinate of the dysregulated-set readout r (dim = DIM),
    accumulating squared gradients into per-node Frobenius norms. DIM backward passes total.
    """
    model.eval()
    x = model.x.detach().clone().requires_grad_(True)
    z = model(A, x, w_feat)
    r = z[torch.tensor(target_idx, device=device)].sum(0)      # (DIM,)
    n, dim = x.shape
    sq = torch.zeros(n, device=device)
    for c in range(r.shape[0]):
        g, = torch.autograd.grad(r[c], x, retain_graph=(c < r.shape[0] - 1))
        sq += (g ** 2).sum(1)
    return torch.sqrt(sq).detach().cpu().numpy()


def main(build: str, dim: int, layers: int, epochs: int, lr: float,
         neg_ratio: int, holdout: float, seed: int) -> int:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_build(build)

    nodes = pd.read_csv(cfg.network_nodes, sep="\t", keep_default_na=False)
    edges = pd.read_csv(cfg.network_edges, sep="\t", keep_default_na=False)
    node_ids = nodes["node_id"].tolist()
    idx = {g: i for i, g in enumerate(node_ids)}
    n = len(node_ids)

    # dysregulated set m = nodes with a direction (DE/literature) UNION metabolite sinks
    dys_mask = (nodes["direction"].astype(str).str.len() > 0) | (nodes["node_type"] == "metabolite")
    target_idx = np.where(dys_mask.to_numpy())[0]
    print(f"build={build} | N={n} nodes, {len(edges)} edges, |dysregulated set|={len(target_idx)} "
          f"| device={device}", flush=True)

    sw = nodes["sender_weight"].astype(float).to_numpy()
    A = build_operator(edges, idx, device, self_weight=sw)
    w_feat = torch.tensor(np.log(sw), dtype=torch.float32, device=device).unsqueeze(1)
    pos_w = torch.tensor(np.log(edges["weight"].astype(float).to_numpy()), dtype=torch.float32, device=device)
    model = Encoder(n, dim, layers).to(device)
    dec = BilinearDecoder(dim).to(device)
    whead = WeightHead(dim).to(device)

    pos_src = edges["source"].map(idx).to_numpy()
    pos_dst = edges["target"].map(idx).to_numpy()
    print("training encoder (link prediction + edge-weight reconstruction)...", flush=True)
    train(A, model, dec, pos_src, pos_dst, n, device, epochs, lr, neg_ratio, holdout, rng,
          whead=whead, pos_w=pos_w, w_feat=w_feat)

    print("computing Jacobian influence on the dysregulated set...", flush=True)
    infl = jacobian_influence(A, model, target_idx, device, w_feat)

    # is_dysregulated flags nodes already in the target set: their influence includes a self
    # contribution (their own embedding is in the readout), so filter them out to find control
    # points that aren't themselves moving (cf. the analytic reach guidance in CAVEATS.md).
    out = nodes[["node_id", "node_type", "source", "direction"]].copy()
    out["is_dysregulated"] = dys_mask.astype(int).to_numpy()
    out["embed_influence"] = infl
    out = out.sort_values("embed_influence", ascending=False).reset_index(drop=True)
    out["embed_influence_rank"] = out.index + 1

    dest = cfg.results_dir / "embedding_influence.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, sep="\t", index=False)
    print(f"\nwrote {dest}", flush=True)
    print(out.head(15).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", default="macrophage_crohn")
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    ap.add_argument("--holdout", type=float, default=HOLDOUT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    raise SystemExit(main(a.build, a.dim, a.layers, a.epochs, a.lr,
                          a.neg_ratio, a.holdout, a.seed))
