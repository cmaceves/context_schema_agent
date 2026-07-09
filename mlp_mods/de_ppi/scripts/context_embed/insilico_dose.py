"""Dose-sweep in-silico test: for each gene in a disease network, set its expression to {0.5x, 1x, 2x} of its
MATCHED-HEALTHY value (scaled in LINEAR CP10k space, not log), re-run the frozen encoder, and measure the
projection onto the disease->healthy axis at each dose. Grade each gene by dose-response monotonicity, then ask
whether monotonic / dose-responsive genes recover OpenTargets targets better than the single-point projection.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/context_embed/insilico_dose.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "mlp_mods/de_ppi/scripts/embed")
from embedding_utils import Encoder
from joint_embed import Net

RES = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi")
DIS, HLT, OT_DISEASE = "crohn_colon_macrophage_inflammatory", "healthy_colon_macrophage_inflammatory", "crohn"
OT = Path("mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv")
DOSES = [0.5, 1.0, 2.0]


def mrr_hits(sub_ranks, n_pos, N):
    rr = 1.0 / sub_ranks
    return rr.mean(), int((sub_ranks <= 10).sum()), int((sub_ranks <= 50).sum())


def main():
    dev = torch.device("cpu")
    ck = torch.load(RES / "encoder.pt", map_location=dev, weights_only=False)
    order = ck["node_id"]; idx = {g: i for i, g in enumerate(order)}; cfg = ck["config"]
    m = Encoder(cfg["N"], cfg["dim"], cfg["layers"], use_self_lin=cfg["use_self_lin"], use_expr_feat=cfg["use_expr_feat"]).to(dev)
    m.load_state_dict(ck["encoder"]); m.eval()
    dn = Net(DIS, idx, dev, RES / "networks" / DIS, self_loops=cfg["self_loops"])
    hn = Net(HLT, idx, dev, RES / "networks" / HLT, self_loops=cfg["self_loops"])

    def emb(net, expr):
        with torch.no_grad(): return m(net.A, w_feat=net.w_feat, node_feat=expr).cpu().numpy()
    Zd, Zh = emb(dn, dn.expr), emb(hn, hn.expr)
    present = dn.present & hn.present
    axis = Zh - Zd; unit = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)

    def proj(expr):
        Zp = emb(dn, expr)
        return float((((Zp - Zd) * unit).sum(1) * present).sum())

    ot = dict(zip(*[pd.read_csv(OT, sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))
    edf = pd.read_csv(RES / "networks" / DIS / "network_edges.tsv", sep="\t", keep_default_na=False)
    outdeg = edf.source.value_counts().to_dict()

    genes = [order[i] for i in np.where(dn.present)[0]]
    rows = []
    for k, g in enumerate(genes):
        xi = idx[g]; hf = float(hn.expr[xi])                       # healthy log1p(CP10k) value
        hx = np.expm1(hf)                                          # -> linear CP10k
        p = {}
        for d in DOSES:
            e = dn.expr.clone(); e[xi] = float(np.log1p(d * hx))   # d x healthy expression (linear), back to log
            p[d] = proj(e)
        ph, pm, pd2 = p[0.5], p[1.0], p[2.0]
        mono = (ph < pm < pd2) or (ph > pm > pd2)
        rows.append(dict(protein=g, ot=round(ot.get(g, 0), 3), out_degree=int(outdeg.get(g, 0)),
                         healthy_expr=round(hf, 3), proj_half=round(ph, 4), proj_match=round(pm, 4),
                         proj_double=round(pd2, 4), slope=round(pd2 - ph, 4), monotonic=mono))
        if (k + 1) % 500 == 0: print(f"  {k+1}/{len(genes)}", flush=True)
    df = pd.DataFrame(rows)
    out = RES / "insilico_perturb" / f"{DIS}_dose_sweep.tsv"
    df.sort_values("proj_match", key=lambda s: s.abs(), ascending=False).to_csv(out, sep="\t", index=False)

    N = len(df)
    for thr in (0.5, 0.3):
        pos = df.ot > thr; npos = int(pos.sum())
        print(f"\n===== OT > {thr}  (npos={npos}/{N}) =====")
        # enrichment: are OT targets more monotonic?
        mono_pos = df.monotonic[pos].mean(); mono_neg = df.monotonic[~pos].mean()
        print(f"  monotonic fraction: OT+={mono_pos:.2f}  vs  OT-={mono_neg:.2f}  ({df.monotonic.mean():.2f} overall)")
        for name, key, absval in [("proj_match (single point)", "proj_match", True),
                                   ("slope |2x-0.5x|", "slope", True)]:
            d = df.assign(_s=df[key].abs() if absval else df[key]).sort_values("_s", ascending=False).reset_index(drop=True)
            d["rank"] = np.arange(1, N + 1)
            r = d.loc[d.ot > thr, "rank"].to_numpy()
            mrr, h10, h50 = mrr_hits(r, npos, N)
            print(f"  rank by {name:26s}: MRR={mrr:.4f} (rand {(np.log(N)+.577)/N:.4f})  Hits@10={h10}  Hits@50={h50}")
        # among monotonic-only, rank by proj_match
        dm = df[df.monotonic].assign(_s=df.proj_match.abs()).sort_values("_s", ascending=False).reset_index(drop=True)
        dm["rank"] = np.arange(1, len(dm) + 1)
        rm = dm.loc[dm.ot > thr, "rank"].to_numpy()
        if len(rm):
            print(f"  MONOTONIC-only ({len(dm)} genes) rank by proj_match: MRR={(1/rm).mean():.4f}  "
                  f"Hits@10={int((rm<=10).sum())}  Hits@50={int((rm<=50).sum())}  (OT+ kept={len(rm)}/{npos})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
