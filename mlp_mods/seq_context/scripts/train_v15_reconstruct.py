"""v15 — reconstruct-the-network from identity+context labels (no ESM).

Encoder input for each (protein, context) = learned embeddings of [protein_id, disease, tissue, cell_type, state]
-> MLP -> latent z (the context-specific protein embedding, the deliverable) -> decoder -> reconstruct that
protein's regulatory-neighbor set (indicator over the gene universe), BCE. The network is only the OUTPUT, so the
label embeddings are forced to carry the network structure. Macrophage only.

Out: results/link_v15/embeddings.npz (z, context, prot, disease, tissue, state) + prints Crohn disease->healthy
driver reversal (top genes whose role shifts most, influence-weighted).
Run: .venv_scvi/bin/python mlp_mods/seq_context/scripts/train_v15_reconstruct.py
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SEQ = Path("mlp_mods/seq_context"); NET = SEQ / "scenic/networks"
PID_DIM, CTX_DIM, LATENT, HIDDEN, EPOCHS, BS, SEED = 128, 32, 64, 256, 50, 512, 0
DIS_TISSUES = ("colon", "ileum")


class V15(nn.Module):
    def __init__(self, nP, nD, nT, nC, nS, G):
        super().__init__()
        self.p = nn.Embedding(nP, PID_DIM); self.d = nn.Embedding(nD, CTX_DIM)
        self.t = nn.Embedding(nT, CTX_DIM); self.c = nn.Embedding(nC, CTX_DIM); self.s = nn.Embedding(nS, CTX_DIM)
        din = PID_DIM + 4 * CTX_DIM
        self.enc = nn.Sequential(nn.Linear(din, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, G))

    def embed(self, pid, did, tid, cid, sid):
        x = torch.cat([self.p(pid), self.d(did), self.t(tid), self.c(cid), self.s(sid)], -1)
        return self.enc(x)

    def forward(self, pid, did, tid, cid, sid):
        return self.dec(self.embed(pid, did, tid, cid, sid))


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tags = sorted(d.name for d in NET.glob("*macrophage*") if (d / "edges_cistarget.tsv").exists())
    conn = {}; nodes = set()
    for t in tags:
        e = pd.read_csv(NET / t / "edges_cistarget.tsv", sep="\t"); c = defaultdict(set)
        for tf, tg in zip(e.tf.astype(str), e.target.astype(str)):
            c[tf].add(tg); c[tg].add(tf); nodes.add(tf); nodes.add(tg)
        conn[t] = c
    nodes = sorted(nodes); nidx = {g: i for i, g in enumerate(nodes)}; G = len(nodes)
    # factor vocabularies
    def parts(t): a = t.split("_"); return a[0], a[1], a[2], a[3]  # disease, tissue, celltype, state
    provoc = {g: i for i, g in enumerate(nodes)}
    dv = {v: i for i, v in enumerate(sorted({parts(t)[0] for t in tags}))}
    tv = {v: i for i, v in enumerate(sorted({parts(t)[1] for t in tags}))}
    cv = {v: i for i, v in enumerate(sorted({parts(t)[2] for t in tags}))}
    sv = {v: i for i, v in enumerate(sorted({parts(t)[3] for t in tags}))}
    # samples
    pid, did, tid, cid, sid, ctx, prot, ii, jj, r = [], [], [], [], [], [], [], [], [], 0
    for t in tags:
        dd, tt, cc, ss = parts(t)
        for p, nbrs in conn[t].items():
            pid.append(provoc[p]); did.append(dv[dd]); tid.append(tv[tt]); cid.append(cv[cc]); sid.append(sv[ss])
            ctx.append(t); prot.append(p)
            for n in nbrs:
                ii.append(r); jj.append(nidx[n])
            r += 1
    Y = sp.csr_matrix((np.ones(len(ii), np.float32), (ii, jj)), shape=(r, G))
    pid = np.array(pid); did = np.array(did); tid = np.array(tid); cid = np.array(cid); sid = np.array(sid)
    ctx = np.array(ctx); prot = np.array(prot)
    print(f"v15: samples={r}, proteins={len(nodes)}, G={G}, factors: dis={len(dv)} tis={len(tv)} ct={len(cv)} st={len(sv)}", flush=True)

    m = V15(len(nodes), len(dv), len(tv), len(cv), len(sv), G).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_w = torch.tensor([(G - Y.nnz / r) / (Y.nnz / r)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    P = torch.tensor(pid, device=dev); D = torch.tensor(did, device=dev); T = torch.tensor(tid, device=dev)
    C = torch.tensor(cid, device=dev); S = torch.tensor(sid, device=dev)
    idx = np.arange(r)
    for ep in range(EPOCHS):
        np.random.shuffle(idx); tot = 0.0
        for s0 in range(0, r, BS):
            b = idx[s0:s0 + BS]; bt = torch.tensor(b, device=dev)
            yb = torch.tensor(Y[b].toarray(), device=dev)
            logit = m(P[bt], D[bt], T[bt], C[bt], S[bt]); loss = lossf(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1}/{EPOCHS} recon-loss {tot/(r//BS+1):.4f}", flush=True)

    # reconstruction quality: rank true neighbors vs non-neighbors for a sample of rows
    m.eval()
    with torch.no_grad():
        samp = np.random.choice(r, 400, replace=False); aucs = []
        for k in samp:
            lg = m(P[k:k+1], D[k:k+1], T[k:k+1], C[k:k+1], S[k:k+1]).cpu().numpy().ravel()
            y = np.zeros(G); y[Y[k].indices] = 1
            if 0 < y.sum() < G:
                aucs.append(roc_auc_score(y, lg))
        print(f"reconstruction AUC (true neighbors vs rest, from labels only): {np.mean(aucs):.3f}", flush=True)
        Z = np.zeros((r, LATENT), np.float32)
        for s0 in range(0, r, 4096):
            e = min(s0 + 4096, r)
            Z[s0:e] = m.embed(P[s0:e], D[s0:e], T[s0:e], C[s0:e], S[s0:e]).cpu().numpy()
    out = SEQ / "results/link_v15"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz", emb=Z, context=ctx, prot_idx=prot,
             disease=np.array([p.split("_")[0] for p in ctx]),
             tissue=np.array([p.split("_")[1] for p in ctx]),
             state=np.array([p.split("_")[3] for p in ctx]))
    print(f"saved -> {out/'embeddings.npz'}", flush=True)

    # Crohn disease->healthy driver reversal (tissue-matched macrophage)
    Zd = Z.astype(np.float64)
    isc = np.array([t.startswith("crohn_") and any(x in t for x in DIS_TISSUES) for t in ctx])
    ish = np.array([t.startswith("healthy_") and any(x in t for x in DIS_TISSUES) for t in ctx])
    def pool(mask):
        acc = defaultdict(list); deg = defaultdict(int)
        for k in np.where(mask)[0]:
            acc[prot[k]].append(Zd[k]); deg[prot[k]] += len(conn[ctx[k]][prot[k]])
        return {g: np.mean(v, 0) for g, v in acc.items()}, deg
    zc, degc = pool(isc); zh, degh = pool(ish)
    both = sorted(set(zc) & set(zh))
    dz = {g: np.linalg.norm(zh[g] - zc[g]) for g in both}
    infl = {g: 0.5 * (degc.get(g, 0) + degh.get(g, 0)) for g in both}
    score = {g: dz[g] * np.log1p(infl[g]) for g in both}
    top = sorted(both, key=lambda g: -score[g])[:10]
    print("\n=== v15 Crohn macrophage: top-10 role-shift (disease -> healthy) ===")
    print(f"{'gene':10s} {'role_shift':>10s} {'deg_crohn':>9s} {'deg_healthy':>11s} {'direction':>9s}")
    for g in top:
        dc, dh = degc.get(g, 0), degh.get(g, 0)
        print(f"{g:10s} {dz[g]:10.3f} {dc:9d} {dh:11d} {'reduce' if dc>dh else 'restore':>9s}")


if __name__ == "__main__":
    main()
