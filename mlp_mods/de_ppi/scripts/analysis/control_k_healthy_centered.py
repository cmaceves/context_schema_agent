"""Control k: healthy-centered between-study. Δ(study)=Z_disease(study)-Z_healthy(own study), per protein;
then repeat the between-study comparison on Δ. Demonstrates batch removal by reporting RAW vs CENTERED
cosine on the SAME study pairs -- if the study/batch baseline is shared by both arms, same-disease
replicates should agree MORE after centering (centered cos > raw between-study cos).

Matching: each disease per-(dataset,state) network is paired with its OWN-STUDY healthy network
(same tissue/celltype/state, same donor-overlap study group). Study groups collapse donor-sharing
depositions (so a 'between-study' pair is never a collection vs itself) -- same rule as build_healthy_controls.

  raw_cos(A,B)      = cos(Z_dA-consensus, Z_dB-consensus)         (uncentered, main-consensus reference)
  centered_cos(A,B) = cos(Δ_A, Δ_B)  where Δ=Z_disease-Z_healthy  (own-study healthy reference)

Pairs: same (disease,tissue,celltype,state), different study group.

Output (controls/): control_k_pairs.tsv, and a printed raw-vs-centered summary by disease.
Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/control_k_healthy_centered.py
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RS = Path("mlp_mods/rank_shifts")
NORMAL = {"normal", "healthy"}
# same (tissue, celltype) per source as the builders
SOURCES = {"microglia_alzheimers": ("brain", "microglia"), "fibroblast_alzheimers": ("brain", "fibroblast"),
           "macrophage_crohn": ("ileum", "macrophage"), "macrophage_crohn_colon": ("colon", "macrophage"),
           "macrophage_crohn_rep": ("colon", "macrophage"), "fibroblast_crohn": ("gut", "fibroblast"),
           "stem_crohn": ("ileum", "stem"), "macrophage_uc_smillie": ("colon", "macrophage"),
           "macrophage_ild": ("lung", "macrophage")}


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
        rep = max(members, key=lambda m: cnt.get(m, 0))[:8]
        for m in members:
            out[m[:8]] = rep
    return out


def build_ds2study():
    """(tissue, celltype, dataset8) -> study8, replicating build_healthy_controls' healthy grouping."""
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
    m = np.load(res / "embeddings.npz", allow_pickle=True)
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    assert list(m["node_id"]) == list(c["node_id"])
    is_prot = m["node_type"] == "protein"; pi = np.where(is_prot)[0]
    Zm, presm = m["Z"][:, pi, :], m["present"][:, pi]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmean(np.where(presm[:, :, None], Zm, np.nan), axis=0)
    valid = ~np.isnan(consensus).any(axis=1)

    tags = list(c["tags"]); Zc, presc = c["Z"][:, pi, :], c["present"][:, pi]
    ti = {t: i for i, t in enumerate(tags)}
    ds2study = build_ds2study()
    P = {t: parse(t) for t in tags}
    healthy_idx = {}  # (study8,tissue,ct,state) -> control index
    for t, p in P.items():
        if p["disease"] == "healthy" and p["split"] is None and p["state"] != "allstates":
            healthy_idx[(p["dataset"], p["tissue"], p["celltype"], p["state"])] = ti[t]

    # disease per-(dataset,state) nets -> Δ vs own-study healthy
    delta = {}  # disease control idx -> (study8, key, Δ vector, present mask)
    for t, p in P.items():
        if p["disease"] == "healthy" or p["split"] is not None or p["state"] == "allstates":
            continue
        st8 = ds2study.get((p["tissue"], p["celltype"], p["dataset"]))
        if st8 is None:
            continue
        hk = (st8, p["tissue"], p["celltype"], p["state"])
        hi = healthy_idx.get(hk)
        if hi is None:
            continue
        di = ti[t]
        # option 1: Δ over DISEASE-present proteins; for proteins absent in the healthy network we use
        # the encoder's Z for that node anyway (isolated node, self-loop only = context-free baseline),
        # so disease-gained proteins (composition signal) are included instead of dropped.
        mask = presc[di]
        d = np.zeros_like(Zc[di]); d[mask] = Zc[di][mask] - Zc[hi][mask]
        delta[di] = dict(study=st8, disease=p["disease"], tissue=p["tissue"], celltype=p["celltype"],
                         state=p["state"], present=mask, dvec=d, didx=di)

    # k pairs: same (disease,tissue,ct,state), different study group; one net per study (largest present)
    by_key = defaultdict(dict)
    for di, info in delta.items():
        k = (info["disease"], info["tissue"], info["celltype"], info["state"])
        cur = by_key[k].get(info["study"])
        if cur is None or info["present"].sum() > delta[cur]["present"].sum():
            by_key[k][info["study"]] = di

    rows = []
    for (dis, tis, ct, state), studies in by_key.items():
        for sa, sb in combinations(sorted(studies), 2):
            ia, ib = studies[sa], studies[sb]
            both = delta[ia]["present"] & delta[ib]["present"] & valid
            if both.sum() < 20:
                continue
            da, db = delta[ia]["dvec"][both], delta[ib]["dvec"][both]
            na, nb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
            centered_cos = float(((da * db).sum(1) / (na * nb + 1e-9)).mean())
            mag_centered = float(np.linalg.norm(da - db, axis=1).mean())
            # raw (uncentered) on the same proteins
            za, zb = Zc[delta[ia]["didx"]][both] - consensus[both], Zc[delta[ib]["didx"]][both] - consensus[both]
            ra, rb = np.linalg.norm(za, axis=1), np.linalg.norm(zb, axis=1)
            raw_cos = float(((za * zb).sum(1) / (ra * rb + 1e-9)).mean())
            rows.append(dict(control="k", control_name="healthy_centered_between_study",
                             disease=dis, tissue=tis, celltype=ct, state=state,
                             study_a=sa, study_b=sb, n_proteins=int(both.sum()),
                             raw_cos=round(raw_cos, 4), centered_cos=round(centered_cos, 4),
                             centered_minus_raw=round(centered_cos - raw_cos, 4),
                             mag_centered=round(mag_centered, 4)))

    df = pd.DataFrame(rows)
    out = res / "controls"
    df.to_csv(out / "control_k_pairs.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_k_pairs.tsv'} ({len(df)} between-study pairs)\n")
    if len(df):
        summ = (df.groupby("disease").agg(n=("raw_cos", "size"), raw_cos=("raw_cos", "mean"),
                centered_cos=("centered_cos", "mean"), delta=("centered_minus_raw", "mean")).round(4))
        allrow = pd.DataFrame({"n": [len(df)], "raw_cos": [df.raw_cos.mean().round(4)],
                               "centered_cos": [df.centered_cos.mean().round(4)],
                               "delta": [df.centered_minus_raw.mean().round(4)]}, index=["ALL"])
        print("=== control k: raw vs healthy-centered between-study cosine ===")
        print(pd.concat([summ, allrow]).to_string())
        print("\n(centered_cos > raw_cos => removing the own-study healthy baseline raised same-context "
              "between-study agreement = batch/cohort variance removed)")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    raise SystemExit(main(ap.parse_args().main_name))
