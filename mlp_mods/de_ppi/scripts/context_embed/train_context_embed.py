"""Train the joint PPI encoder with an added DISEASE-signal objective (see de_ppi/CONTEXT_EMBED.md).

Reuses joint_embed's network loading (Net) + encoder/decoder + baseline objective (directed link prediction +
edge-weight reconstruction), and adds one method-specific loss (weight --lam):

  --method contrastive       CENTER LOSS on disease arm: pull a protein's same-arm per-network embeddings
                             together, push different-arm centroids apart (hinge, --margin). Makes disease an axis.
  --method healthy_centered  AUX head reconstructing the disease-vs-healthy EXPRESSION delta from the embedding
                             delta:  aux(Z_disease[p] - healthy_centroid[p]) ~= expr_disease[p] - mean_healthy_expr[p].
  --method baseline          no extra loss (sanity = joint_embed).

Networks are READ from results/<base>/networks (default the combat_loc build); embeddings/encoder are WRITTEN to
results/<res-name>/ (embeddings.npz + encoder.pt), same schema as joint_embed so infer_controls / compare_controls
/ PCA work unchanged. Arm is parsed as tag.split('_')[0] (healthy/crohn/uc/alz/ild).

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/train_context_embed.py \
        --method contrastive --res-name crohn_alzheimer_ild_uc_context_contrastive --expr-feat
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "mlp_mods/de_ppi/scripts/embed")
from embedding_utils import Encoder, BilinearDecoder, WeightHead, DIM, LAYERS, EPOCHS, LR, NEG_RATIO, HOLDOUT, SEED
from joint_embed import Net, networks_root, discover_tags, W_RECON

HERE = Path("mlp_mods/de_ppi")


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
    arms = [t.split("_")[0] for t in tags]
    return order, node_type, nets, arms


def center_loss(Z, Pf, arms, margin):
    """Z (T,N,dim), Pf (T,N) float present; pull same-arm embeddings together, push arm centroids apart."""
    uniq = sorted(set(arms)); a_id = torch.tensor([uniq.index(a) for a in arms])
    cent, pres_a = {}, {}
    for k, a in enumerate(uniq):
        m = a_id == k
        cnt = Pf[m].sum(0).clamp(min=1.0)
        cent[a] = (Z[m] * Pf[m].unsqueeze(-1)).sum(0) / cnt.unsqueeze(-1)     # (N,dim)
        pres_a[a] = Pf[m].sum(0) > 0
    within = torch.stack([(((Z[i] - cent[arms[i]]) ** 2).sum(1) * Pf[i]).sum() / Pf[i].sum()
                          for i in range(len(arms))]).mean()
    pushes = []
    for a, b in combinations(uniq, 2):
        both = pres_a[a] & pres_a[b]
        if both.sum() > 0:
            d = torch.norm((cent[a] - cent[b])[both], dim=1)
            pushes.append(F.relu(margin - d).mean())
    between = torch.stack(pushes).mean() if pushes else torch.tensor(0.0, device=Z.device)
    return within + between


def main(base, res_name, method, lam, margin, dim, layers, epochs, lr, neg_ratio, holdout, seed, expr_feat) -> int:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order, node_type, nets, arms = build(base, device)
    N = len(order); T = len(nets)
    Pf = torch.stack([torch.tensor(n.present, dtype=torch.float32, device=device) for n in nets])   # (T,N)
    print(f"base={base} N={N} tags={T} method={method} lam={lam} device={device}", flush=True)

    model = Encoder(N, dim, layers, use_self_lin=True, use_expr_feat=expr_feat).to(device)
    dec = BilinearDecoder(dim).to(device); whead = WeightHead(dim).to(device)
    aux = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)).to(device)   # method 4 head
    params = list(model.parameters()) + list(dec.parameters()) + list(whead.parameters()) + list(aux.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    splits = []
    for net in nets:
        perm = rng.permutation(len(net.pos_src)); nh = int(len(perm) * holdout)
        splits.append((perm[nh:], perm[:nh]))

    healthy_i = [i for i, a in enumerate(arms) if a == "healthy"]
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = torch.tensor(0.0, device=device); Zs = []
        for net, (keep, _) in zip(nets, splits):
            z = model(net.A, w_feat=net.w_feat, node_feat=net.expr); Zs.append(z)
            ts = torch.tensor(net.pos_src[keep], device=device); td = torch.tensor(net.pos_dst[keep], device=device)
            ndst = torch.randint(0, N, (len(td) * neg_ratio,), device=device)
            pos = dec(z[ts], z[td]); neg = dec(z[ts.repeat(neg_ratio)], z[ndst])
            lp = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                     torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
            wr = F.mse_loss(whead(z[ts], z[td]), net.pos_w[torch.tensor(keep, device=device)])
            loss = loss + lp + W_RECON * wr
        Z = torch.stack(Zs)                                                  # (T,N,dim)

        extra = torch.tensor(0.0, device=device)
        if method == "contrastive":
            extra = center_loss(Z, Pf, arms, margin)
        elif method == "healthy_centered" and healthy_i:
            hz = (Z[healthy_i] * Pf[healthy_i].unsqueeze(-1)).sum(0) / Pf[healthy_i].sum(0).clamp(min=1).unsqueeze(-1)
            he = torch.stack([nets[i].expr.squeeze(-1) for i in healthy_i])
            hef = (he * Pf[healthy_i]).sum(0) / Pf[healthy_i].sum(0).clamp(min=1)
            hp = Pf[healthy_i].sum(0) > 0
            terms = []
            for i, a in enumerate(arms):
                if a == "healthy":
                    continue
                m = (Pf[i] > 0) & hp
                if m.sum() == 0:
                    continue
                pred = aux(Z[i] - hz).squeeze(-1)
                terms.append(F.mse_loss(pred[m], (nets[i].expr.squeeze(-1) - hef)[m]))
            if terms:
                extra = torch.stack(terms).mean()
        loss = loss + lam * extra
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}  extra({method}) {float(extra):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        Z = np.stack([model(n.A, w_feat=n.w_feat, node_feat=n.expr).cpu().numpy() for n in nets])
    present = np.stack([n.present for n in nets])
    res = HERE / "results" / res_name; res.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res / "embeddings.npz", node_id=np.array(order, dtype=object),
                        node_type=np.array([node_type[g] for g in order], dtype=object),
                        tags=np.array([n.tag for n in nets], dtype=object), Z=Z, present=present)
    torch.save({"encoder": model.state_dict(), "decoder": dec.state_dict(), "weight_head": whead.state_dict(),
                "config": {"N": N, "dim": dim, "layers": layers, "self_loops": True,
                           "use_self_lin": True, "use_expr_feat": expr_feat}, "node_id": list(order)},
               res / "encoder.pt")
    print(f"wrote {res/'embeddings.npz'} and {res/'encoder.pt'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    ap.add_argument("--res-name", required=True)
    ap.add_argument("--method", choices=("baseline", "contrastive", "healthy_centered"), required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=DIM); ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO); ap.add_argument("--holdout", type=float, default=HOLDOUT)
    ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--expr-feat", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(a.base, a.res_name, a.method, a.lam, a.margin, a.dim, a.layers, a.epochs,
                          a.lr, a.neg_ratio, a.holdout, a.seed, a.expr_feat))
