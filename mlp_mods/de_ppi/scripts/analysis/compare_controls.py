"""Walk the inferred control networks into the a-g control rows (see de_ppi/CONTROLS.md).

Pairs are formed among the CONTROL networks (control_embeddings.npz, produced by infer_controls.py); the
MAIN embedding (embeddings.npz) is used ONLY to build the consensus origin. For each pair we report both
metrics vs that consensus, plus node overlap:

  consensus[p]               = per-protein mean of Z over MAIN networks (where present)
  average_magnitude_shift    = mean ||Z_a[p]-Z_b[p]||                       over proteins in both & >=1 main net
  average_cosine_similarity  = mean cos(Z_a[p]-consensus[p], Z_b[p]-cons.)  over the same proteins
  jaccard / n_overlap        = node overlap of the two control networks     (overlap confound control)

Control classes (by which single factor differs; study label = same dataset -> within_study else between_study):
  a donor_split          two donor halves of one (dataset,state)        within
  b between_study        same (disease,tissue,celltype,state), diff dataset
  c cell_state           same (disease,tissue,celltype,dataset), diff state         within
  d cell_type            two allstates nets, same (disease,tissue,dataset), diff celltype  within
  e tissue               two allstates nets, same (disease,celltype), diff tissue
  g disease_between_study same (tissue,celltype,state), diff disease, diff dataset    between

Tag format: <disease>_<dataset8>_<tissue>_<celltype>_<state>[_split{A,B}] | <...>_allstates  (state may contain '_').

Output (results/<main>/controls/):
  control_pairs.tsv     one row per classified pair (incl. per-pair cosine, the raw direction data)
  control_summary.tsv   per control class: n, mean MAGNITUDE shift, mean jaccard (magnitude-only;
                        direction lives in control_centered.py's control_centered_summary.tsv)

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/compare_controls.py
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS = ["disease", "dataset", "tissue", "celltype", "state"]
CLASS_NAME = {"a": "donor_split", "b": "between_study", "c": "cell_state", "d": "cell_type",
              "e": "tissue", "g": "different_disease",
              "h": "healthy_donor_split", "i": "healthy_between_study", "j": "healthy_cell_type",
              "m": "healthy_loo"}
# what is HELD CONSTANT in each comparison (the remaining factor is the one varied)
CONSTANT = {"a": "disease, study, tissue, celltype, state", "b": "disease, tissue, celltype, state",
            "c": "disease, study, tissue, celltype", "d": "disease, tissue, state",
            "e": "disease, celltype, state", "g": "tissue, celltype, state",
            "h": "healthy, study, tissue, celltype, state", "i": "healthy, tissue, celltype, state",
            "j": "healthy, study, tissue", "m": "healthy, tissue, celltype, state"}
# control_summary.tsv is MAGNITUDE-ONLY (how far each factor shifts the embedding). Per-pair cosines are
# still written to control_pairs.tsv as raw data; the "direction" question lives in control_centered.py's
# standalone control_centered_summary.tsv, not in the headline summary.


def parse_tag(tag: str) -> dict:
    split = None
    for h in ("A", "B"):
        if tag.endswith(f"_split{h}"):
            split, tag = h, tag[: -len(f"_split{h}")]
            break
    p = tag.split("_")
    return {"disease": p[0], "dataset": p[1], "tissue": p[2], "celltype": p[3],
            "state": "_".join(p[4:]), "split": split}


def classify(a: dict, b: dict, allstates_bs: bool = False) -> str | None:
    """Return control-class letter for a pair, or None if it isn't a clean one-factor contrast.

    Healthy networks (disease=='healthy') get h/i/j; disease networks get a-g; mixed
    disease-vs-healthy pairs are not part of the ladder (-> None; that is the healthy-centering axis).

    allstates_bs: for cell-type-only builds (no per-state nets), let ALLSTATES nets form the between-study
    (b/i) and disease (g) controls too (normally those come from per-state nets).
    """
    ha, hb = a["disease"] == "healthy", b["disease"] == "healthy"
    if ha != hb:
        return None
    healthy = ha and hb
    # donor splits: same base, opposite halves
    if a["split"] and b["split"]:
        same_base = all(a[f] == b[f] for f in FACTORS)
        if not (same_base and a["split"] != b["split"]):
            return None
        return "h" if healthy else "a"
    if a["split"] or b["split"]:
        return None
    diffs = {f for f in FACTORS if a[f] != b[f]}
    both_all = a["state"] == "allstates" and b["state"] == "allstates"
    one_all = a["state"] == "allstates" or b["state"] == "allstates"
    if healthy:                                             # h/i/j only
        if both_all and diffs == {"celltype"}:
            return "j"
        if not one_all and diffs == {"dataset"}:
            return "i"
        if allstates_bs and both_all and diffs == {"dataset"}:   # allstates between-study (state-less build)
            return "i"
        return None
    if both_all:                                            # (d) cell type / (e) tissue
        if diffs == {"celltype"}:
            return "d"
        if diffs in ({"tissue"}, {"tissue", "dataset"}):
            return "e"
        if allstates_bs and diffs == {"dataset"}:                # allstates between-study
            return "b"
        if allstates_bs and diffs == {"disease", "dataset"}:     # allstates cross-disease
            return "g"
        return None
    if one_all:
        return None
    if diffs == {"dataset"}:                                # per-(dataset,state) pooled networks
        return "b"
    if diffs == {"state"}:
        return "c"
    if diffs == {"disease", "dataset"}:
        return "g"
    return None


def main(main_name, allstates_bs=False) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    m = np.load(res / "embeddings.npz", allow_pickle=True)
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    assert list(m["node_id"]) == list(c["node_id"]), "control/main node universes differ"

    is_prot = m["node_type"] == "protein"
    pi = np.where(is_prot)[0]
    Zm, presm = m["Z"][:, pi, :], m["present"][:, pi]
    masked = np.where(presm[:, :, None], Zm, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmean(masked, axis=0)
    valid = ~np.isnan(consensus).any(axis=1)

    # LOO networks (control m, built by build_healthy_loo.py) are scored by control_m_healthy_loo.py;
    # exclude them here so their 'loopool/loosingle' tags don't get mis-classified into the a-j ladder.
    all_tags = list(c["tags"])
    keep = [i for i, t in enumerate(all_tags) if "_loo" not in t]
    tags = [all_tags[i] for i in keep]
    Zc, presc = c["Z"][:, pi, :][keep], c["present"][:, pi][keep]
    R = Zc - consensus[None]
    Rn = np.linalg.norm(R, axis=2)
    P = parse_tag  # alias
    parsed = {t: P(t) for t in tags}

    rows = []
    for ia, ib in combinations(range(len(tags)), 2):
        ta, tb = tags[ia], tags[ib]
        cls = classify(parsed[ta], parsed[tb], allstates_bs)
        if cls is None:
            continue
        both = presc[ia] & presc[ib] & valid
        if both.sum() == 0:
            continue
        ra, rb = R[ia, both], R[ib, both]
        na, nb = Rn[ia, both], Rn[ib, both]
        cos = ((ra * rb).sum(1) / (na * nb + 1e-9)).mean()
        shift = np.linalg.norm(Zc[ia, both] - Zc[ib, both], axis=1).mean()
        sa, sb = set(np.where(presc[ia])[0]), set(np.where(presc[ib])[0])
        jac = len(sa & sb) / len(sa | sb)
        pa, pb = parsed[ta], parsed[tb]
        rows.append({
            "control": cls, "control_name": CLASS_NAME[cls],
            "study": "within_study" if pa["dataset"] == pb["dataset"] else "between_study",
            "network_a": ta, "network_b": tb,
            "disease_a": pa["disease"], "disease_b": pb["disease"],
            "tissue_a": pa["tissue"], "tissue_b": pb["tissue"],
            "celltype_a": pa["celltype"], "celltype_b": pb["celltype"],
            "state_a": pa["state"], "state_b": pb["state"],
            "n_overlap": len(sa & sb), "jaccard": round(jac, 4),
            "average_magnitude_shift": round(float(shift), 4),
            "average_cosine_similarity": round(float(cos), 4),
        })

    df = pd.DataFrame(rows).sort_values(["control", "study", "network_a", "network_b"]).reset_index(drop=True)
    out = res / "controls"
    df.to_csv(out / "control_pairs.tsv", sep="\t", index=False)
    summ = (df.groupby(["control", "control_name", "study"])
            .agg(n_pairs=("jaccard", "size"), jaccard=("jaccard", "mean"),
                 avg_magnitude_shift=("average_magnitude_shift", "mean"),
                 sd_magnitude_shift=("average_magnitude_shift", "std")).round(4).reset_index())
    summ["constant"] = summ.control.map(CONSTANT)                  # what's held constant in the comparison
    summ = summ[["control", "control_name", "constant", "study", "n_pairs", "jaccard",
                 "avg_magnitude_shift", "sd_magnitude_shift"]]
    # control m (healthy_loo) is computed by control_m_healthy_loo.py -> control_m_pairs.tsv; fold its
    # summary in here so it sits next to i/h. (Same idea could extend to k/l if desired.)
    m_path = out / "control_m_pairs.tsv"
    if m_path.exists():
        md = pd.read_csv(m_path, sep="\t")
        if len(md):
            summ = pd.concat([summ, pd.DataFrame([{
                "control": "m", "control_name": "healthy_loo", "constant": CONSTANT["m"],
                "study": "between_study", "n_pairs": len(md), "jaccard": 1.0,
                "avg_magnitude_shift": round(md["shift"].mean(), 4),
                "sd_magnitude_shift": round(md["shift"].std(), 4)}])], ignore_index=True)
    summ.to_csv(out / "control_summary.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_pairs.tsv'} ({len(df)} pairs)\nwrote {out/'control_summary.tsv'}\n")
    print(summ.to_string(index=False))

    # standard output: healthy between-study (control i) — % node overlap vs mean per-protein cosine,
    # one point per (tissue, celltype, state) pair of independent healthy studies (tests membership-churn)
    hi = df[df.control == "i"]
    if len(hi):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(hi.jaccard * 100, hi.average_cosine_similarity, s=60, alpha=0.8,
                   edgecolor="k", linewidth=0.4, color="#1f77b4")
        ax.set_xlabel("% node overlap (Jaccard)")
        ax.set_ylabel("mean per-protein cosine (consensus-centered)")
        ax.set_title("Healthy between-study pairs (same tissue/celltype/state, different study)\n"
                     f"n={len(hi)}  pearson(overlap,cos)={np.corrcoef(hi.jaccard, hi.average_cosine_similarity)[0,1]:.2f}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        img = res / "images" / "healthy_between_study_overlap_vs_cosine.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(img, dpi=150)
        print(f"wrote {img}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--allstates-between-study", action="store_true",
                    help="cell-type-only builds: let allstates nets form b/i/g (no per-state nets exist)")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.allstates_between_study))
