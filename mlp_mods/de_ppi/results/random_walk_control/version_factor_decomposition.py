"""Per protein: what drives its embeddings' separation from their centroid — disease(arm), tissue, or cell state?

For each protein with >=2 context-versions, decompose the total spread (sum of squared distances of its versions
to their centroid, 64-d) by each factor via one-way eta^2 = between-group SS / total SS. The factor with the
highest eta^2 is that protein's 'driver'. (Cell type is constant here — macrophage-only build — so it cannot
drive and is omitted.)

Coupled build. Outputs a per-protein table, a driver-fraction summary (proteins in >=6 contexts, for stability),
and an eta^2 heatmap for the OT>0.5 Crohn targets.

Run: .venv/bin/python mlp_mods/de_ppi/results/random_walk_control/version_factor_decomposition.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

RES = Path("mlp_mods/de_ppi/results")
BUILD = "crohn_alzheimer_ild_uc_embedding_protein_linked"
OT = Path("mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv")
FACTORS = ["arm", "tissue", "state"]


def eta2(V, labels):
    c = V.mean(0); total = ((V - c) ** 2).sum()
    if total <= 0:
        return 0.0
    between = 0.0
    for lv in set(labels):
        idx = [i for i, l in enumerate(labels) if l == lv]
        between += len(idx) * ((V[idx].mean(0) - c) ** 2).sum()
    return float(between / total)


def main():
    z = np.load(RES / BUILD / "embeddings.npz", allow_pickle=True)
    Z, pres, tags = z["Z"], z["present"], [str(t) for t in z["tags"]]
    order = list(z["node_id"])
    fac = {"arm": [t.split("_")[0] for t in tags], "tissue": [t.split("_")[1] for t in tags],
           "state": ["_".join(t.split("_")[3:]) for t in tags]}
    crohn_ot = dict(zip(*[pd.read_csv(OT, sep="\t")[c] for c in ["gene_symbol", "score_indirect"]]))

    rows = []
    for i, g in enumerate(order):
        tis = np.where(pres[:, i])[0]
        if len(tis) < 2:
            continue
        V = Z[tis, i, :]
        e = {f: eta2(V, [fac[f][t] for t in tis]) for f in FACTORS}
        driver = max(FACTORS, key=lambda f: e[f])
        rows.append(dict(protein=g, n_ctx=len(tis), ot=round(crohn_ot.get(g, 0), 3),
                         **{f"eta2_{f}": round(e[f], 3) for f in FACTORS}, driver=driver))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "random_walk_control" / "version_factor_decomposition.tsv", sep="\t", index=False)

    stable = df[df.n_ctx >= 6]
    print(f"driver factor across all proteins with >=6 contexts (n={len(stable)}):")
    print((stable.driver.value_counts(normalize=True).round(3)).to_string())
    print(f"\nmean eta^2 by factor (>=6 ctx): " +
          "  ".join(f"{f}={stable[f'eta2_{f}'].mean():.2f}" for f in FACTORS))

    ot = df[df.ot > 0.5].sort_values("ot", ascending=False)
    print(f"\nOT>0.5 targets — eta^2 per factor + driver:")
    print(ot[["protein", "n_ctx", "ot", "eta2_arm", "eta2_tissue", "eta2_state", "driver"]].to_string(index=False))
    print(f"\nOT>0.5 driver counts: {dict(ot.driver.value_counts())}")

    # heatmap for OT>0.5
    M = ot.set_index("protein")[[f"eta2_{f}" for f in FACTORS]].values
    fig, ax = plt.subplots(figsize=(5, 0.42 * len(ot) + 1.5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(FACTORS)
    ax.set_yticks(range(len(ot))); ax.set_yticklabels(ot.protein, fontsize=8)
    for a in range(len(ot)):
        for b in range(3):
            ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[a, b] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="eta^2 (variance explained)")
    ax.set_title("OT>0.5 targets: what drives version separation", fontsize=9)
    fig.tight_layout()
    out = RES / "random_walk_control" / "images"; out.mkdir(exist_ok=True)
    fig.savefig(out / "ot_factor_decomposition.png", dpi=150); plt.close(fig)
    print(f"\nwrote {out}/ot_factor_decomposition.png and version_factor_decomposition.tsv")
    print("NOTE: arm/tissue/state are partially confounded here (e.g. lung only in ILD, ileum only in Crohn), "
          "so eta^2 overlaps between factors; read as approximate attribution.")


if __name__ == "__main__":
    main()
