"""Masked-feature-modeling pretraining for the PPI encoder (self-supervised, no extra data).

Objective (per network): mask the expression feature of a random subset of nodes, run the encoder, and predict
each masked node's true expression from the resulting embedding (MSE). Because the encoder runs with
use_expr_feat=True, the input IS the (masked) per-network expression and the learned per-node IDENTITY table
(Encoder.x) is bypassed -- so a masked node's prediction must come from its NEIGHBOURS' expression + topology
(context), not from a memorized per-gene mean. This is the Geneformer/scGPT masked-token idea on our graph.

HELD-OUT NODES: within each network the maskable (real-expression) nodes are split once into train (90%) and
eval (10%). Only train nodes are ever masked-and-scored during training; eval nodes stay as unmasked context.
At eval they are masked and reconstructed -- a low eval MSE (vs a predict-the-mean baseline) proves the encoder
learned a transferable context->expression map rather than memorizing training targets.

Placeholders (expr==0: healthy-placeholder / disease-only-isolated nodes) are EXCLUDED from masking/scoring
(their zero is artificial) but kept as context. Trains jointly over all networks with one shared encoder+head.

Writes results/<res-name>/: embeddings.npz + encoder.pt (same schema as the other builds) and masked_metrics.tsv
(per-network + overall train/eval reconstruction R^2 vs baseline).

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/train_masked.py \
        --base crohn_alzheimer_ild_uc_coexpr_healthyph --res-name crohn_alzheimer_ild_uc_masked
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "mlp_mods/de_ppi/scripts/embed")
from embedding_utils import Encoder, DIM, LAYERS, EPOCHS, LR, SEED
from joint_embed import Net, networks_root, discover_tags

HERE = Path("mlp_mods/de_ppi")
MASK_VAL = -1.0          # sentinel: expression is log1p(CP10k) >= 0, so -1 is out-of-range -> a clean "masked" signal
EPS = 1e-6               # placeholder nodes have expression exactly 0.0


def build(base, device):
    tags = discover_tags(base)
    root = networks_root(base)
    node_type = {}
    for t in tags:
        nd = pd.read_csv(root / t / "network_nodes.tsv", sep="\t", keep_default_na=False)
        for nid, nt in zip(nd["node_id"], nd["node_type"]):
            node_type.setdefault(nid, nt)
    order = list(node_type); idx = {g: i for i, g in enumerate(order)}
    nets = [Net(t, idx, device, root / t, self_loops=True) for t in tags]
    return order, node_type, nets


def main(base, res_name, dim, layers, epochs, lr, mask_frac, holdout, seed) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order, node_type, nets = build(base, device)
    N = len(order); T = len(nets)
    print(f"base={base} N={N} tags={T} device={device} mask_frac={mask_frac} holdout={holdout}", flush=True)

    # per-network maskable (real-expression) nodes, split once into train / eval (held-out)
    train_idx, eval_idx = [], []
    for net in nets:
        expr = net.expr.squeeze(-1).cpu().numpy()
        maskable = np.where(net.present & (expr > EPS))[0]
        perm = rng.permutation(maskable); nh = int(len(perm) * holdout)
        eval_idx.append(torch.tensor(perm[:nh], device=device, dtype=torch.long))
        train_idx.append(torch.tensor(perm[nh:], device=device, dtype=torch.long))
    print(f"maskable nodes: train={sum(len(t) for t in train_idx)}, held-out eval={sum(len(e) for e in eval_idx)}",
          flush=True)

    model = Encoder(N, dim, layers, use_self_lin=True, use_expr_feat=True).to(device)
    head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)

    def predict(net, mask_positions):
        node_feat = net.expr.clone()
        node_feat[mask_positions] = MASK_VAL
        z = model(net.A, w_feat=net.w_feat, node_feat=node_feat)
        return head(z[mask_positions]).squeeze(-1), z

    for ep in range(epochs):
        model.train(); head.train(); opt.zero_grad()
        loss = torch.tensor(0.0, device=device); n_terms = 0
        for net, tr in zip(nets, train_idx):
            if len(tr) < 2:
                continue
            k = max(1, int(len(tr) * mask_frac))
            sel = tr[torch.randperm(len(tr), device=device)[:k]]      # random subset masked this step
            pred, _ = predict(net, sel)
            loss = loss + F.mse_loss(pred, net.expr[sel].squeeze(-1)); n_terms += 1
        loss = loss / max(n_terms, 1)
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  train masked MSE {loss.item():.4f}", flush=True)

    # held-out evaluation: mask each net's eval nodes, reconstruct; baseline = predict train-maskable mean
    model.eval(); head.eval(); rows = []
    with torch.no_grad():
        for net, tr, ev in zip(nets, train_idx, eval_idx):
            if len(ev) < 2:
                continue
            true = net.expr[ev].squeeze(-1)
            pred, _ = predict(net, ev)
            mse = F.mse_loss(pred, true).item()
            base_pred = net.expr[tr].squeeze(-1).mean()               # predict-the-mean baseline
            base_mse = F.mse_loss(base_pred.expand_as(true), true).item()
            var = true.var(unbiased=False).item()
            rows.append(dict(tag=net.tag, n_eval=len(ev), eval_mse=round(mse, 4),
                             baseline_mse=round(base_mse, 4),
                             r2=round(1 - mse / var, 4) if var > 0 else float("nan"),
                             r2_vs_baseline=round(1 - mse / base_mse, 4) if base_mse > 0 else float("nan")))
    met = pd.DataFrame(rows)

    # embeddings from FULL (unmasked) expression, same schema as the other builds
    with torch.no_grad():
        Z = np.stack([model(n.A, w_feat=n.w_feat, node_feat=n.expr).cpu().numpy() for n in nets])
    present = np.stack([n.present for n in nets])
    res = HERE / "results" / res_name; res.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res / "embeddings.npz", node_id=np.array(order, dtype=object),
                        node_type=np.array([node_type[g] for g in order], dtype=object),
                        tags=np.array([n.tag for n in nets], dtype=object), Z=Z, present=present)
    torch.save({"encoder": model.state_dict(), "recon_head": head.state_dict(),
                "config": {"N": N, "dim": dim, "layers": layers, "self_loops": True,
                           "use_self_lin": True, "use_expr_feat": True, "mask_val": MASK_VAL}, "node_id": list(order)},
               res / "encoder.pt")
    met.to_csv(res / "masked_metrics.tsv", sep="\t", index=False)

    tot_eval = int(met.n_eval.sum())
    ov_mse = float((met.eval_mse * met.n_eval).sum() / tot_eval)
    ov_base = float((met.baseline_mse * met.n_eval).sum() / tot_eval)
    print(f"\nHELD-OUT reconstruction (n={tot_eval} eval nodes over {len(met)} nets):")
    print(f"  masked MSE={ov_mse:.4f}   baseline(mean) MSE={ov_base:.4f}   R2_vs_baseline={1 - ov_mse/ov_base:+.3f}")
    print(met.sort_values("r2_vs_baseline", ascending=False).head(8).to_string(index=False))
    print(f"\nwrote {res/'embeddings.npz'}, {res/'encoder.pt'}, {res/'masked_metrics.tsv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="crohn_alzheimer_ild_uc_coexpr_healthyph")
    ap.add_argument("--res-name", default="crohn_alzheimer_ild_uc_masked")
    ap.add_argument("--dim", type=int, default=DIM); ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--mask-frac", type=float, default=0.25, help="fraction of train-maskable nodes masked per step")
    ap.add_argument("--holdout", type=float, default=0.10, help="fraction of maskable nodes held out for eval")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    raise SystemExit(main(a.base, a.res_name, a.dim, a.layers, a.epochs, a.lr, a.mask_frac, a.holdout, a.seed))
