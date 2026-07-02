"""Scatter: mean per-protein cosine(disease, own-study healthy) vs % shared proteins, per disease network.

For each disease per-(dataset,state) control network, match its OWN-study healthy network (same
tissue/celltype/state, donor-overlap study group) and compute:
  mean_cos      = mean over shared (present in both) proteins of cos(Z_disease[p], Z_healthy[p])
  pct_shared    = 100 * |present_dis & present_heal| / |present_dis | present_heal|   (Jaccard %)

Output: images/disease_vs_healthy_similarity.png + tables behind it (controls/disease_vs_healthy_similarity.tsv).
Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/disease_vs_healthy_similarity.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RS = Path("mlp_mods/rank_shifts")
NORMAL = {"normal", "healthy"}
SOURCES = {"microglia_alzheimers": ("brain", "microglia"), "fibroblast_alzheimers": ("brain", "fibroblast"),
           "macrophage_crohn": ("ileum", "macrophage"), "macrophage_crohn_colon": ("colon", "macrophage"),
           "macrophage_crohn_rep": ("colon", "macrophage"), "fibroblast_crohn": ("gut", "fibroblast"),
           "stem_crohn": ("ileum", "stem"), "macrophage_uc_smillie": ("colon", "macrophage"),
           "macrophage_ild": ("lung", "macrophage")}
COLORS = {"crohn": "#d62728", "uc": "#1f77b4", "alz": "#2ca02c", "ild": "#9467bd"}


def group_studies(obs):
    ds = list(pd.unique(obs.dataset_id)); parent = {d: d for d in ds}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    d2 = defaultdict(set)
    for don, d in zip(obs.donor_id.astype(str), obs.dataset_id):
        d2[don].add(d)
    for s in d2.values():
        s = list(s)
        for k in s[1:]:
            parent[find(s[0])] = find(k)
    cnt = obs.dataset_id.value_counts(); groups = defaultdict(list)
    for d in ds:
        groups[find(d)].append(d)
    out = {}
    for members in groups.values():
        rep = max(members, key=lambda mm: cnt.get(mm, 0))[:8]
        for mm in members:
            out[mm[:8]] = rep
    return out


def build_ds2study():
    m = {}
    for src, (tissue, ct) in SOURCES.items():
        f = RS / f"{src}_states" / "cell_states.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        h = df[df.disease.astype(str).str.lower().isin(NORMAL)]
        if len(h) == 0:
            continue
        for ds8, st8 in group_studies(h).items():
            m[(tissue, ct, ds8)] = st8
    return m


def parse(tag):
    split = None
    for hh in ("A", "B"):
        if tag.endswith(f"_split{hh}"):
            split, tag = hh, tag[: -len(f"_split{hh}")]; break
    p = tag.split("_")
    return dict(disease=p[0], dataset=p[1], tissue=p[2], celltype=p[3], state="_".join(p[4:]), split=split)


def main(main_name="crohn_alzheimer_ild_uc_embedding_expressed") -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    is_prot = c["node_type"] == "protein"; pi = np.where(is_prot)[0]
    Zc, presc = c["Z"][:, pi, :], c["present"][:, pi]
    tags = list(c["tags"]); ti = {t: i for i, t in enumerate(tags)}
    ds2study = build_ds2study()
    P = {t: parse(t) for t in tags}
    heal = {}
    for t, p in P.items():
        if p["disease"] == "healthy" and p["split"] is None and p["state"] != "allstates":
            heal[(p["dataset"], p["tissue"], p["celltype"], p["state"])] = ti[t]

    rows = []
    for t, p in P.items():
        if p["disease"] == "healthy" or p["split"] is not None or p["state"] == "allstates":
            continue
        st8 = ds2study.get((p["tissue"], p["celltype"], p["dataset"]))
        hi = heal.get((st8, p["tissue"], p["celltype"], p["state"])) if st8 else None
        if hi is None:
            continue
        di = ti[t]
        pd_, ph = presc[di], presc[hi]
        both = pd_ & ph
        if both.sum() < 20:
            continue
        za, zb = Zc[di][both], Zc[hi][both]
        na, nb = np.linalg.norm(za, axis=1), np.linalg.norm(zb, axis=1)
        mean_cos = float(((za * zb).sum(1) / (na * nb + 1e-9)).mean())
        pct_shared = 100.0 * both.sum() / (pd_ | ph).sum()
        rows.append(dict(disease=p["disease"], tissue=p["tissue"], celltype=p["celltype"], state=p["state"],
                         network=t, study=st8, n_shared=int(both.sum()),
                         pct_shared=round(pct_shared, 2), mean_cos=round(mean_cos, 4)))

    df = pd.DataFrame(rows)
    df.to_csv(res / "controls" / "disease_vs_healthy_similarity.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for dis, g in df.groupby("disease"):
        ax.scatter(g.pct_shared, g.mean_cos, s=70, alpha=0.8, edgecolor="k", linewidth=0.4,
                   color=COLORS.get(dis, "gray"), label=f"{dis} (n={len(g)})")
    ax.set_xlabel("% shared proteins with own-study healthy arm (Jaccard)")
    ax.set_ylabel("mean per-protein cosine(disease, own-study healthy)")
    ax.set_title("Disease network vs its own-study healthy arm\n(high cosine + high overlap = disease ≈ healthy)")
    ax.legend(title="disease", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = res / "images" / "disease_vs_healthy_similarity.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img, dpi=150)
    print(f"wrote {img}\nwrote {res/'controls'/'disease_vs_healthy_similarity.tsv'} ({len(df)} disease networks)\n")
    print(df.groupby("disease").agg(n=("mean_cos", "size"), pct_shared=("pct_shared", "mean"),
          mean_cos=("mean_cos", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    raise SystemExit(main(ap.parse_args().main_name))
