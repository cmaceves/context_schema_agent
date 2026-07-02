"""Second control set: HEALTHY-ANCHORED disease-shift agreement (interpretation A), cosine-only.

For every per-study DISEASE control network D(study, tissue, celltype, state), form its disease-shift vector
against the PAIRED HEALTHY arm of the SAME study (control-k construction -> per-study batch cancels inside
the shift):

    r_D[p] = Z_D[p] - Z_H[p],   H = healthy(study, tissue, celltype, state)  [fallback: healthy(study,...,allstates)]

Then, for each factor, take the COSINE between two shift vectors that differ in exactly that one factor with
the other three held constant. Cosine = mean over shared proteins p of cos(r_a[p], r_b[p]) -- the same
per-protein direction metric used by the a-j ladder. It asks: do these two conditions depart from their own
healthy baseline in the SAME direction?

  factor      hold constant                       vary     study
  disease     tissue, celltype, state             disease  (between; each own-study-centered)
  cell_state  disease, tissue, celltype, study    state    (within-study)
  cell_type   disease, tissue, study (allstates)  celltype (within-study; node sets differ -> intersection)
  tissue      disease, celltype, state            tissue   (between; each own-study-centered)

The healthy reference is ALWAYS the paired healthy of the SAME study (never the mean of the two compared
nets -> non-degenerate), so this is meaningful only where per-study healthy exists; we never fall back to the
pooled healthy main (that would reintroduce the batch we are cancelling). Intended for the ComBat-corrected
build (healthy-centered cosine is what batch correction rescues).

Output (results/<main>/controls/):
  healthy_centered_factor_pairs.tsv    one row per one-factor pair (factor, nets, studies, overlap, cosine)
  healthy_centered_factor_summary.tsv  per factor: n_pairs, mean cosine, sd, mean overlap

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/healthy_centered_factors.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DISEASES = {"crohn", "uc", "alz", "ild"}
MIN_OVERLAP = 20


def parse(tag: str) -> dict:
    split = None
    for h in ("A", "B"):
        if tag.endswith(f"_split{h}"):
            split, tag = h, tag[: -len(f"_split{h}")]
            break
    p = tag.split("_")
    return dict(arm=p[0], study=p[1], tissue=p[2], ct=p[3], state="_".join(p[4:]),
                split=split, loo=p[1].startswith(("loopool", "loosingle")))


def main(main_name: str, healthy_ref: str = "ownstudy") -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres, tags = c["Z"][:, pi, :], c["present"][:, pi], list(c["tags"])
    idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}

    if healthy_ref == "ownstudy":
        # paired-healthy lookup (per study), keyed on exact context and on the allstates fallback
        H: dict[tuple, str] = {}
        for t in tags:
            q = P[t]
            if q["arm"] == "healthy" and not q["loo"] and q["split"] is None:
                H[(q["study"], q["tissue"], q["ct"], q["state"])] = t

        def ref_vec(q: dict):
            rt = H.get((q["study"], q["tissue"], q["ct"], q["state"])) \
                or H.get((q["study"], q["tissue"], q["ct"], "allstates"))
            return (Z[idx[rt]], pres[idx[rt]]) if rt else (None, None)
    else:
        # POOLED healthy: studies-pooled healthy MAIN of the context (embeddings.npz), allstates -> mean of states
        m_ = np.load(res / "embeddings.npz", allow_pickle=True)
        assert list(m_["node_id"]) == list(c["node_id"]), "main/control node universes differ"
        Zm, presm, mtags = m_["Z"][:, pi, :], m_["present"][:, pi], list(m_["tags"])
        HM, HM_ct = {}, defaultdict(list)
        for i, t in enumerate(mtags):
            p = t.split("_")
            arm, tis, ct, st = p[0], p[1], p[2], "_".join(p[3:])
            if arm == "healthy":
                HM[(tis, ct, st)] = i
                HM_ct[(tis, ct)].append(i)

        def ref_vec(q: dict):
            i = HM.get((q["tissue"], q["ct"], q["state"]))
            if i is not None:
                return Zm[i], presm[i]
            rows = HM_ct.get((q["tissue"], q["ct"]))               # allstates disease net -> mean of state mains
            if not rows:
                return None, None
            pm = presm[rows].any(0)
            zc = np.where(presm[rows][:, :, None], Zm[rows], np.nan)
            with np.errstate(invalid="ignore"):
                zmean = np.nan_to_num(np.nanmean(zc, axis=0))
            return zmean, pm

    # disease-shift vectors r_D = Z_D - Z_healthy_ref (over proteins present in both)
    dis = [t for t in tags if P[t]["arm"] in DISEASES and not P[t]["loo"] and P[t]["split"] is None]
    shift: dict[str, tuple] = {}                                  # tag -> (r (N,dim), present mask)
    for t in dis:
        zref, pref = ref_vec(P[t])
        if zref is None:
            continue
        i = idx[t]
        m = pres[i] & pref
        if m.sum() < MIN_OVERLAP:
            continue
        r = np.full(Z.shape[1:], np.nan)
        r[m] = Z[i, m] - zref[m]
        shift[t] = (r, m)

    def cos_pair(ta: str, tb: str):
        ra, ma = shift[ta]
        rb, mb = shift[tb]
        m = ma & mb
        if m.sum() < MIN_OVERLAP:
            return None
        a, b = ra[m], rb[m]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        cos = float(((a * b).sum(1) / (na * nb + 1e-9)).mean())
        sa, sb = set(np.where(ma)[0]), set(np.where(mb)[0])
        return cos, int(m.sum()), len(sa & sb) / len(sa | sb)

    rows = []

    def add(factor: str, groupkey, vary: str, pool: list[str]):
        groups = defaultdict(list)
        for t in pool:
            if t in shift:
                groups[groupkey(P[t])].append(t)
        for ts in groups.values():
            for ta, tb in itertools.combinations(sorted(ts), 2):
                if P[ta][vary] == P[tb][vary]:                   # must differ in the varied factor
                    continue
                r = cos_pair(ta, tb)
                if r is None:
                    continue
                cos, nov, jac = r
                rows.append(dict(factor=factor, network_a=ta, network_b=tb,
                                 study_a=P[ta]["study"], study_b=P[tb]["study"],
                                 study="within" if P[ta]["study"] == P[tb]["study"] else "between",
                                 n_overlap=nov, jaccard=round(jac, 4), cosine=round(cos, 4)))

    perstate = [t for t in dis if P[t]["state"] != "allstates"]
    allstates = [t for t in dis if P[t]["state"] == "allstates"]
    add("disease",    lambda q: (q["tissue"], q["ct"], q["state"]),          "arm",   perstate)
    add("cell_state", lambda q: (q["arm"], q["tissue"], q["ct"], q["study"]), "state", perstate)
    add("cell_type",  lambda q: (q["arm"], q["tissue"], q["study"]),          "ct",    allstates)
    add("tissue",     lambda q: (q["arm"], q["ct"], q["state"]),              "tissue", perstate)

    out = res / "controls"
    sfx = "" if healthy_ref == "ownstudy" else "_pooledref"
    df = pd.DataFrame(rows).sort_values(["factor", "study", "network_a", "network_b"]).reset_index(drop=True)
    df.to_csv(out / f"healthy_centered_factor_pairs{sfx}.tsv", sep="\t", index=False)
    summ = (df.groupby("factor")
            .agg(n_pairs=("cosine", "size"), n_within=("study", lambda s: (s == "within").sum()),
                 mean_cosine=("cosine", "mean"), sd_cosine=("cosine", "std"),
                 mean_overlap=("n_overlap", "mean"), mean_jaccard=("jaccard", "mean")).round(4).reset_index())
    summ.to_csv(out / f"healthy_centered_factor_summary{sfx}.tsv", sep="\t", index=False)
    print(f"[healthy_ref={healthy_ref}] shift vectors built for {len(shift)}/{len(dis)} disease nets\n")
    print(f"wrote {out/'healthy_centered_factor_pairs.tsv'} ({len(df)} pairs)")
    print(f"wrote {out/'healthy_centered_factor_summary.tsv'}\n")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    ap.add_argument("--healthy-ref", choices=["ownstudy", "pooled"], default="ownstudy",
                    help="anchor each disease shift on the paired same-study healthy (ownstudy; batch cancels) "
                         "or on the studies-pooled healthy main (pooled)")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.healthy_ref))
