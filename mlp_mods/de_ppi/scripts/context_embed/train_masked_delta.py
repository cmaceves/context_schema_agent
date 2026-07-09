"""Masked feature modeling + MASKED DIFFERENTIAL disease-direction aux loss (+ optional topology anchor).

The full objective designed in CONTEXT_EMBED.md ("Method 4 refined"). One shared directed-message-passing
encoder f_theta (use_expr_feat=True: node input is its scalar log-expression via Linear(1,dim); identity table
bypassed). Trained on the _coexpr_healthyph base (matched disease<->healthy pairs per tissue/celltype/state;
zero-feature placeholders present, excluded from masking/scoring). Three terms share ONE masked forward pass:

  L_mask  (masked feature modeling)   -- mask a random subset M of real-expression nodes; predict each masked
          node's true expression from its context-only embedding (recon head g_phi). MSE. Same as train_masked.

  L_aux   (masked differential)       -- for each disease net paired with its MATCHED-context healthy net,
          over proteins in M present-with-real-expression in BOTH:
              dZ_p   = z_disease[p] - z_healthy[p]        (both computed with p MASKED)
              target = expr_disease[p] - expr_healthy[p]  (TRUE, unmasked delta)
              L_aux  = MSE( h_psi(dZ_p), target )
          Masking p in both passes is the load-bearing choice: it kills the expression-passthrough shortcut,
          so dZ_p must be carried by NEIGHBOURS' shifts, not p's own passed-through value.

  L_link  (optional topology anchor)  -- directed link prediction + edge-weight reconstruction (joint_embed's
          objective), downweighted, to keep the shared space graph-grounded. beta default modest.

  Total:  L = L_mask + lam_aux * L_aux + lam_link * L_link

Global per-step mask set M (universe indices) is applied to every network, so a protein masked in a disease net
is also masked in its matched healthy -> the paired dZ is well defined. Held-out universe nodes (never masked in
training) give a generalization eval for BOTH L_mask (reconstruction) and L_aux (delta prediction) vs
predict-the-mean baselines -- NOT the deprecated a-j ladder (CONTROLS.md).

NOTE: co-variation structure in our own cross-context data, NOT causal/perturbational.

Writes results/<res-name>/: embeddings.npz + encoder.pt (same schema as other builds) + masked_delta_metrics.tsv.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/train_masked_delta.py \
        --base crohn_alzheimer_ild_uc_coexpr_healthyph --res-name crohn_alzheimer_ild_uc_masked_delta
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
from embedding_utils import Encoder, BilinearDecoder, WeightHead, DIM, LAYERS, EPOCHS, LR, NEG_RATIO, SEED
from joint_embed import Net, networks_root, discover_tags, W_RECON

HERE = Path("mlp_mods/de_ppi")
MASK_VAL = -1.0          # sentinel: log1p(CP10k) >= 0, so -1 is a clean out-of-range "masked" signal
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


def matched_pairs(nets):
    """disease-net index -> matched healthy-net index (same tissue_celltype_state; arm swapped to 'healthy')."""
    by_tag = {n.tag: i for i, n in enumerate(nets)}
    pairs = []
    for i, n in enumerate(nets):
        arm = n.tag.split("_")[0]
        if arm == "healthy":
            continue
        htag = "healthy_" + n.tag.split("_", 1)[1]
        if htag in by_tag:
            pairs.append((i, by_tag[htag]))
    return pairs


def main(base, res_name, dim, layers, epochs, lr, mask_frac, holdout, lam_aux, lam_link, neg_ratio, seed) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order, node_type, nets = build(base, device)
    N = len(order); T = len(nets)
    pairs = matched_pairs(nets)
    print(f"base={base} N={N} tags={T} device={device} | mask_frac={mask_frac} holdout={holdout} "
          f"lam_aux={lam_aux} lam_link={lam_link}", flush=True)
    print(f"matched disease<->healthy pairs for L_aux: {len(pairs)}", flush=True)

    # per-net real-expression mask over the universe (present AND non-placeholder)
    real = torch.stack([torch.tensor((n.present) & (n.expr.squeeze(-1).cpu().numpy() > EPS),
                                     device=device) for n in nets])                       # (T, N) bool
    exprT = torch.stack([n.expr.squeeze(-1) for n in nets])                                # (T, N)

    # GLOBAL held-out split over universe nodes real in >=1 net (eval nodes never masked in training)
    maskable_any = torch.where(real.any(0))[0].cpu().numpy()
    perm = rng.permutation(maskable_any); nh = int(len(perm) * holdout)
    eval_univ = torch.tensor(perm[:nh], device=device, dtype=torch.long)
    train_univ = torch.tensor(perm[nh:], device=device, dtype=torch.long)
    print(f"maskable universe nodes: train={len(train_univ)}, held-out eval={len(eval_univ)}", flush=True)

    model = Encoder(N, dim, layers, use_self_lin=True, use_expr_feat=True).to(device)
    recon = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)).to(device)     # g_phi
    auxh = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)).to(device)      # h_psi
    dec = BilinearDecoder(dim).to(device); whead = WeightHead(dim).to(device)
    params = (list(model.parameters()) + list(recon.parameters()) + list(auxh.parameters())
              + list(dec.parameters()) + list(whead.parameters()))
    opt = torch.optim.Adam(params, lr=lr)

    def forward_masked(mask_bool):
        """run every net with the global mask set applied to its real nodes; return z list + per-net masked-bool."""
        zs, mpos = [], []
        for ti, net in enumerate(nets):
            mb = mask_bool & real[ti]                                  # mask only present real nodes here
            node_feat = net.expr.clone()
            node_feat[mb] = MASK_VAL
            zs.append(model(net.A, w_feat=net.w_feat, node_feat=node_feat))
            mpos.append(mb)
        return zs, mpos

    def link_term():
        L = torch.tensor(0.0, device=device)
        for net in nets:
            if len(net.pos_src) == 0:
                continue
            z = model(net.A, w_feat=net.w_feat, node_feat=net.expr)
            ts = torch.tensor(net.pos_src, device=device); td = torch.tensor(net.pos_dst, device=device)
            ndst = torch.randint(0, N, (len(td) * neg_ratio,), device=device)
            pos = dec(z[ts], z[td]); neg = dec(z[ts.repeat(neg_ratio)], z[ndst])
            lp = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                     torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
            wr = F.mse_loss(whead(z[ts], z[td]), net.pos_w)
            L = L + lp + W_RECON * wr
        return L / max(T, 1)

    for ep in range(epochs):
        model.train(); recon.train(); auxh.train(); opt.zero_grad()
        k = max(1, int(len(train_univ) * mask_frac))
        M = train_univ[torch.randperm(len(train_univ), device=device)[:k]]
        mask_bool = torch.zeros(N, dtype=torch.bool, device=device); mask_bool[M] = True
        zs, mpos = forward_masked(mask_bool)

        # L_mask: reconstruct masked real nodes per net
        lmask = torch.tensor(0.0, device=device); nm = 0
        for ti in range(T):
            sel = torch.where(mpos[ti])[0]
            if len(sel) < 2:
                continue
            lmask = lmask + F.mse_loss(recon(zs[ti][sel]).squeeze(-1), exprT[ti][sel]); nm += 1
        lmask = lmask / max(nm, 1)

        # L_aux: masked differential on matched pairs
        laux = torch.tensor(0.0, device=device); na = 0
        for di, hi in pairs:
            common = torch.where(mask_bool & real[di] & real[hi])[0]     # masked & real in BOTH
            if len(common) < 2:
                continue
            dz = zs[di][common] - zs[hi][common]
            target = exprT[di][common] - exprT[hi][common]
            laux = laux + F.mse_loss(auxh(dz).squeeze(-1), target); na += 1
        laux = laux / max(na, 1)

        llink = link_term() if lam_link > 0 else torch.tensor(0.0, device=device)
        loss = lmask + lam_aux * laux + lam_link * llink
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  L_mask {float(lmask):.4f}  L_aux {float(laux):.4f}  "
                  f"L_link {float(llink):.4f}  total {float(loss):.4f}", flush=True)

    # ---- held-out eval (generalization): mask ALL eval nodes, score recon + delta vs predict-the-mean ----
    model.eval(); recon.eval(); auxh.eval()
    eval_bool = torch.zeros(N, dtype=torch.bool, device=device); eval_bool[eval_univ] = True
    train_bool = torch.zeros(N, dtype=torch.bool, device=device); train_bool[train_univ] = True
    with torch.no_grad():
        zs, mpos = forward_masked(eval_bool)
        # reconstruction (L_mask) per net
        rows = []
        for ti, net in enumerate(nets):
            sel = torch.where(mpos[ti])[0]
            if len(sel) < 2:
                continue
            true = exprT[ti][sel]; pred = recon(zs[ti][sel]).squeeze(-1)
            mse = F.mse_loss(pred, true).item()
            tr = torch.where(train_bool & real[ti])[0]
            base = exprT[ti][tr].mean() if len(tr) else true.mean()
            bmse = F.mse_loss(base.expand_as(true), true).item()
            rows.append(dict(tag=net.tag, kind="recon", n=len(sel), mse=round(mse, 4),
                             baseline_mse=round(bmse, 4),
                             r2_vs_baseline=round(1 - mse / bmse, 4) if bmse > 0 else float("nan")))
        # delta (L_aux) per matched pair
        for di, hi in pairs:
            common = torch.where(eval_bool & real[di] & real[hi])[0]
            if len(common) < 2:
                continue
            dz = zs[di][common] - zs[hi][common]
            target = exprT[di][common] - exprT[hi][common]
            pred = auxh(dz).squeeze(-1)
            mse = F.mse_loss(pred, target).item()
            trc = torch.where(train_bool & real[di] & real[hi])[0]
            base = (exprT[di][trc] - exprT[hi][trc]).mean() if len(trc) else target.mean()
            bmse = F.mse_loss(base.expand_as(target), target).item()
            rows.append(dict(tag=nets[di].tag, kind="delta", n=len(common), mse=round(mse, 4),
                             baseline_mse=round(bmse, 4),
                             r2_vs_baseline=round(1 - mse / bmse, 4) if bmse > 0 else float("nan")))
    met = pd.DataFrame(rows)

    # embeddings from FULL (unmasked) expression, same schema as the other builds
    with torch.no_grad():
        Z = np.stack([model(n.A, w_feat=n.w_feat, node_feat=n.expr).cpu().numpy() for n in nets])
    present = np.stack([n.present for n in nets])
    res = HERE / "results" / res_name; res.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res / "embeddings.npz", node_id=np.array(order, dtype=object),
                        node_type=np.array([node_type[g] for g in order], dtype=object),
                        tags=np.array([n.tag for n in nets], dtype=object), Z=Z, present=present)
    torch.save({"encoder": model.state_dict(), "recon_head": recon.state_dict(),
                "aux_head": auxh.state_dict(), "decoder": dec.state_dict(), "weight_head": whead.state_dict(),
                "config": {"N": N, "dim": dim, "layers": layers, "self_loops": True, "use_self_lin": True,
                           "use_expr_feat": True, "mask_val": MASK_VAL,
                           "lam_aux": lam_aux, "lam_link": lam_link}, "node_id": list(order)},
               res / "encoder.pt")
    met.to_csv(res / "masked_delta_metrics.tsv", sep="\t", index=False)

    for kind in ("recon", "delta"):
        sub = met[met.kind == kind]
        if len(sub):
            tot = int(sub.n.sum())
            ov = float((sub.mse * sub.n).sum() / tot); ob = float((sub.baseline_mse * sub.n).sum() / tot)
            print(f"\nHELD-OUT {kind}: n={tot} over {len(sub)} nets | MSE={ov:.4f} baseline={ob:.4f} "
                  f"R2_vs_baseline={1 - ov/ob:+.3f}")
    print(f"\nwrote {res/'embeddings.npz'}, {res/'encoder.pt'}, {res/'masked_delta_metrics.tsv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="crohn_alzheimer_ild_uc_coexpr_healthyph")
    ap.add_argument("--res-name", default="crohn_alzheimer_ild_uc_masked_delta")
    ap.add_argument("--dim", type=int, default=DIM); ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--mask-frac", type=float, default=0.25, help="fraction of train-maskable universe masked/step")
    ap.add_argument("--holdout", type=float, default=0.10, help="fraction of maskable nodes held out for eval")
    ap.add_argument("--lam-aux", type=float, default=1.0, help="weight of the masked differential aux loss")
    ap.add_argument("--lam-link", type=float, default=0.25, help="weight of the topology anchor (0 disables)")
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    raise SystemExit(main(a.base, a.res_name, a.dim, a.layers, a.epochs, a.lr, a.mask_frac, a.holdout,
                          a.lam_aux, a.lam_link, a.neg_ratio, a.seed))
