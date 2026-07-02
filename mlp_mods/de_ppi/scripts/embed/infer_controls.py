"""Forward-pass INFERENCE of control networks through the frozen main-trained encoder.

The joint encoder (joint_embed.py) has a LEARNABLE per-node embedding table x (N, dim) indexed by the
training node universe -- it is transductive over nodes. So a control network is placed in the trained
space by:
  1. loading the saved encoder weights + node universe from encoder.pt,
  2. building the control network's directed operator A and sender-weight feature over THAT fixed
     universe (proteins not in the universe are dropped; universe proteins absent from the control are
     simply not present, self-loop weight 1.0),
  3. a single forward pass model(A, w_feat) -> Z, no gradient, no retraining.

This keeps controls from ever shaping the encoder they are meant to measure. Out-of-universe control
proteins are dropped (they have no trained x row); the count is printed per tag.

Inputs:
  --encoder   results/<main>/encoder.pt          (encoder/decoder/weight-head state + config + node_id)
  --networks  dir of <tag>/network_{nodes,edges}.tsv to infer (default: <controls>/networks)
Output:
  --out       control_embeddings.npz  (node_id, node_type, tags, Z, present)  -- same schema as embeddings.npz

Run:
  .venv/bin/python mlp_mods/de_ppi/scripts/embed/infer_controls.py \
      --encoder mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed/encoder.pt \
      --networks mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed/controls/networks \
      --out      mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed/controls/control_embeddings.npz
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BP
for _sd in ("build", "build/controls", "embed", "analysis"):
    _p = str(_BP("mlp_mods/de_ppi/scripts") / _sd)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embedding_utils import Encoder, build_operator


def infer_one(ndir: Path, idx: dict[str, int], device, self_loops: bool = True, expr_required: bool = False) -> tuple[np.ndarray, np.ndarray, int, "torch.Tensor"]:
    """Return (present mask, sender weights, n_dropped, w_feat) and the operator A for one network dir."""
    nodes = pd.read_csv(ndir / "network_nodes.tsv", sep="\t", keep_default_na=False)
    edges = pd.read_csv(ndir / "network_edges.tsv", sep="\t", keep_default_na=False)
    n = len(idx)
    in_uni = nodes["node_id"].isin(idx)
    n_dropped = int((~in_uni).sum())
    nodes = nodes[in_uni]
    pos = nodes["node_id"].map(idx).to_numpy()
    present = np.zeros(n, dtype=bool); present[pos] = True
    sw = np.ones(n); sw[pos] = nodes["sender_weight"].astype(float).to_numpy()
    e = edges[edges["source"].isin(idx) & edges["target"].isin(idx)]
    A = build_operator(e, idx, device, self_weight=sw, self_loops=self_loops)
    w_feat = torch.tensor(np.log(sw), dtype=torch.float32, device=device).unsqueeze(1)
    if expr_required and "expression" not in nodes.columns:
        raise SystemExit(f"{ndir} has no 'expression' column but encoder was trained with use_expr_feat=True")
    expr = np.zeros(n)
    if "expression" in nodes.columns:
        expr[pos] = nodes["expression"].astype(float).to_numpy()
    node_feat = torch.tensor(expr, dtype=torch.float32, device=device).unsqueeze(1)
    return present, sw, n_dropped, A, w_feat, node_feat


def main(encoder_path, networks_dir, out_path, tags) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    order = list(ckpt["node_id"]); idx = {g: i for i, g in enumerate(order)}
    cfg = ckpt["config"]
    assert cfg["N"] == len(order), "encoder N does not match saved node universe"
    model = Encoder(cfg["N"], cfg["dim"], cfg["layers"], use_self_lin=cfg.get("use_self_lin", True),
                    use_expr_feat=cfg.get("use_expr_feat", False)).to(device)
    model.load_state_dict(ckpt["encoder"]); model.eval()
    self_loops = cfg.get("self_loops", True)
    print(f"encoder self_loops={self_loops} use_self_lin={cfg.get('use_self_lin', True)} "
          f"use_expr_feat={cfg.get('use_expr_feat', False)}", flush=True)

    net_root = Path(networks_dir)
    found = tags or sorted(p.name for p in net_root.iterdir()
                           if (p / "network_nodes.tsv").exists()) if net_root.exists() else []
    if not found:
        raise SystemExit(f"no <tag>/network_nodes.tsv under {net_root}")

    # node_type over the universe (from the encoder's main networks is unknown here; tag with 'protein'
    # for universe nodes -- metabolite sinks in controls are out-of-universe and dropped anyway)
    node_type = np.array(["protein"] * len(order), dtype=object)

    Zs, Ps = [], []
    with torch.no_grad():
        for t in found:
            present, sw, n_drop, A, w_feat, node_feat = infer_one(net_root / t, idx, device, self_loops=self_loops,
                                                                   expr_required=cfg.get("use_expr_feat", False))
            z = model(A, w_feat=w_feat, node_feat=node_feat).detach().cpu().numpy()
            Zs.append(z); Ps.append(present)
            print(f"  {t:48s} present={int(present.sum()):5d}  dropped(out-of-universe)={n_drop}", flush=True)

    Z = np.stack(Zs); present = np.stack(Ps)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, node_id=np.array(order, dtype=object), node_type=node_type,
                        tags=np.array(found, dtype=object), Z=Z, present=present)
    print(f"\nwrote {out}  (Z {Z.shape}, {len(found)} control networks)", flush=True)
    return 0


if __name__ == "__main__":
    MAIN = "mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=f"{MAIN}/encoder.pt")
    ap.add_argument("--networks", default=f"{MAIN}/controls/networks")
    ap.add_argument("--out", default=f"{MAIN}/controls/control_embeddings.npz")
    ap.add_argument("--tags", nargs="+", default=None, help="subset of tags to infer (default: all under --networks)")
    a = ap.parse_args()
    raise SystemExit(main(a.encoder, a.networks, a.out, a.tags))
