"""Control l: WITHIN-study Δ noise floor (donor split) -- the control for control k.

For a study with both disease and healthy donor-split networks, compute the healthy-centered disease shift
on each independent donor half and ask whether it reproduces WITHIN one study:
  Δ_A = Z_disease_splitA - Z_healthy_splitA      (disease/healthy donors of half A)
  Δ_B = Z_disease_splitB - Z_healthy_splitB
  centered_cos = cos(Δ_A, Δ_B)   (option-1 masking: over disease-present proteins; absent-in-healthy uses
                                   the encoder's isolated-node Z, same as control k)

Interpretation vs control k (between-study Δ ~ 0):
  - l ~ 0 too  -> Δ is subtraction-noise-dominated; k is uninformative (no usable signal in Δ).
  - l high, k ~ 0 -> the disease shift is real but COHORT-SPECIFIC (reproduces within a study, not across).

raw_cos = cos(Z_disease_splitA - consensus, Z_disease_splitB - consensus) reported for reference (the
uncentered within-study donor floor, i.e. control a recomputed on the same pairs).

Output (controls/): control_l_pairs.tsv + printed raw-vs-centered summary by disease.
Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/control_l_within_study_delta.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RS = Path("mlp_mods/rank_shifts")
NORMAL = {"normal", "healthy"}
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
    m = np.load(res / "embeddings.npz", allow_pickle=True)
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
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

    # index split networks
    dis_split, heal_split = {}, {}
    for t, p in P.items():
        if p["split"] is None:
            continue
        if p["disease"] == "healthy":
            heal_split[(p["dataset"], p["tissue"], p["celltype"], p["state"], p["split"])] = ti[t]
        else:
            dis_split[(p["disease"], p["dataset"], p["tissue"], p["celltype"], p["state"], p["split"])] = ti[t]

    rows = []
    seen = set()
    for (dis, ds8, tis, ct, state, half), di in dis_split.items():
        if (dis, ds8, tis, ct, state) in seen:
            continue
        st8 = ds2study.get((tis, ct, ds8))
        if st8 is None:
            continue
        dA = dis_split.get((dis, ds8, tis, ct, state, "A")); dB = dis_split.get((dis, ds8, tis, ct, state, "B"))
        hA = heal_split.get((st8, tis, ct, state, "A")); hB = heal_split.get((st8, tis, ct, state, "B"))
        if None in (dA, dB, hA, hB):
            continue
        seen.add((dis, ds8, tis, ct, state))
        # option-1 Δ per half
        def delta(d_i, h_i):
            mask = presc[d_i]
            v = np.zeros_like(Zc[d_i]); v[mask] = Zc[d_i][mask] - Zc[h_i][mask]
            return v, mask
        vA, mA = delta(dA, hA); vB, mB = delta(dB, hB)
        both = mA & mB & valid
        if both.sum() < 20:
            continue
        da, db = vA[both], vB[both]
        na, nb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
        centered_cos = float(((da * db).sum(1) / (na * nb + 1e-9)).mean())
        za, zb = Zc[dA][both] - consensus[both], Zc[dB][both] - consensus[both]
        ra, rb = np.linalg.norm(za, axis=1), np.linalg.norm(zb, axis=1)
        raw_cos = float(((za * zb).sum(1) / (ra * rb + 1e-9)).mean())
        rows.append(dict(control="l", control_name="within_study_delta", disease=dis, tissue=tis,
                         celltype=ct, state=state, study=st8, n_proteins=int(both.sum()),
                         raw_cos=round(raw_cos, 4), centered_cos=round(centered_cos, 4),
                         centered_minus_raw=round(centered_cos - raw_cos, 4)))

    df = pd.DataFrame(rows)
    out = res / "controls"
    df.to_csv(out / "control_l_pairs.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_l_pairs.tsv'} ({len(df)} within-study donor-split pairs)\n")
    if len(df):
        summ = (df.groupby("disease").agg(n=("raw_cos", "size"), raw_cos=("raw_cos", "mean"),
                centered_cos=("centered_cos", "mean")).round(4))
        allrow = pd.DataFrame({"n": [len(df)], "raw_cos": [round(df.raw_cos.mean(), 4)],
                               "centered_cos": [round(df.centered_cos.mean(), 4)]}, index=["ALL"])
        print("=== control l: WITHIN-study donor-split, raw vs healthy-centered cosine ===")
        print(pd.concat([summ, allrow]).to_string())
        print("\ncompare centered_cos to control k's ~0.03 (between-study): if l >> k, the disease shift is "
              "real but cohort-specific; if l ~ 0 too, Δ is noise.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    raise SystemExit(main(ap.parse_args().main_name))
