"""Shared building blocks for the de_ppi learned PPI embeddings.

A featureless weighted *directed* message-passing encoder trained by unsupervised
directed link prediction (asymmetric bilinear decoder + negative sampling, optional
edge-weight reconstruction head), plus the row-normalized aggregation operator and a
Jacobian readout helper. These are pure modeling utilities with no config/IO
dependency; callers (joint_embed.py, joint_embed_context.py, compare_joint_vs_single.py)
own data loading and output.

Defaults below (DIM, LAYERS, ...) are the shared training hyperparameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder / training defaults
DIM = 64          # embedding dimension
LAYERS = 2        # message-passing layers (also the Jacobian receptive field, +self-loops)
EPOCHS = 300
LR = 1e-2
NEG_RATIO = 1     # negative edges sampled per positive edge
HOLDOUT = 0.10    # fraction of edges held out for AUC
SEED = 3


def build_operator(edges: pd.DataFrame, idx: dict[str, int], device,
                   self_weight: np.ndarray | None = None, self_loops: bool = True) -> torch.Tensor:
    """Row-normalized directed aggregation operator, optionally with self-loops.

    A[target, source] = edge weight, so (A @ X)[j] aggregates j's in-neighbours (predecessors)
    -> signal flows source -> target, matching message passing along the directed arc. Self-loops
    let a node keep its own features; rows are normalized to sum to 1 (receiver-side).

    self_weight gates the self-loop: default 1.0 (state-invariant identity term). Passing each
    node's SENDER weight w(i) instead makes a node's own dysregulation enter its own embedding
    (the self-loop now carries w(i)*X[i]), so a DE gene's own embedding shifts with its own DE.
    self_loops=False drops the self-loop entirely (ablation): a node's embedding is then built ONLY
    from its in-neighbours' messages (plus the encoder's separate residual self_lin path). Nodes with
    in-degree 0 get an all-zero aggregation row.
    Returns a sparse (N, N) tensor.
    """
    n = len(idx)
    src = edges["source"].map(idx).to_numpy()
    dst = edges["target"].map(idx).to_numpy()
    w = edges["weight"].astype(float).to_numpy()
    self_w = np.ones(n) if self_weight is None else np.asarray(self_weight, dtype=float)
    # rows = target (receiver), cols = source (sender); + self-loops (gated by self_w) unless ablated
    if self_loops:
        rows = np.concatenate([dst, np.arange(n)])
        cols = np.concatenate([src, np.arange(n)])
        vals = np.concatenate([w, self_w])
    else:
        rows, cols, vals = dst, src, w
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

    def __init__(self, n: int, dim: int, layers: int, use_self_lin: bool = True, use_expr_feat: bool = False):
        super().__init__()
        self.use_self_lin = use_self_lin
        self.use_expr_feat = use_expr_feat               # input = projected per-network log-expression (replaces self.x)
        self.x = nn.Parameter(torch.randn(n, dim) * 0.1)
        self.explin = nn.Linear(1, dim, bias=False)      # projects a node's (per-network) log-expression to the input
        self.lin = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        # residual identity path (W_self); ablated when use_self_lin=False -> positions come only from
        # neighbour messages (+ the self-loop folded into A, if present)
        self.self_lin = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers)) if use_self_lin else None
        self.wlin = nn.Linear(1, dim, bias=False)        # injects each node's (log) sender weight into its input

    def forward(self, A: torch.Tensor, x: torch.Tensor | None = None,
                w_feat: torch.Tensor | None = None, node_feat: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_expr_feat and node_feat is not None:  # per-network expression IS the input (no learned identity)
            h = self.explin(node_feat)
        else:
            h = self.x if x is None else x
        if w_feat is not None:                           # network-specific: node's own rank-shift enters its input
            h = h + self.wlin(w_feat)
        for k in range(len(self.lin)):
            msg = torch.sparse.mm(A, self.lin[k](h))
            h = msg + self.self_lin[k](h) if self.use_self_lin else msg
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
