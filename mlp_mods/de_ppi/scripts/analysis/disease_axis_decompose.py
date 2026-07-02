"""IBD shared/unique disease-axis decomposition (Crohn vs UC, colon macrophage) -- the worked example.

(Distinct from the older disease_axis.py, which computes a baseline-free consensus-axis cosine across all 4
diseases. This script does the per-protein SHARED vs UNIQUE decomposition for the one confound-free contrast.)

Decomposes each protein's disease perturbation (in the 64-d embedding space) into a SHARED component (common
to both diseases) and a UNIQUE component (what distinguishes them). All shifts are batch-cancelled by
centering each disease network on its OWN-study healthy arm; UC is pooled over its two studies AFTER
own-study centering (so no shared-healthy-anchor inflation).

Per protein p (over proteins present in every contributing net):
  r_C[p] = Z_Crohn(518d9049)[p]              - Z_healthy(518d9049)[p]
  r_U[p] = mean_s( Z_UC(s)[p] - Z_healthy(s)[p] ),  s in UC's studies

  CO-MOVEMENT decomposition (shared requires BOTH diseases to move it the same way -- a one-sided move is
  NOT half-shared, unlike a naive (r_C+r_U)/2 mean):
    chat   = unit(r_C + r_U)                       consensus direction
    shared = max(0, min(r_C.chat, r_U.chat))       common co-moving magnitude along the consensus
    crohn_unique = ||r_C - shared*chat|| ;  uc_unique = ||r_U - shared*chat||
    shared_frac[p] = 2*shared^2 / (||r_C||^2 + ||r_U||^2)   in [0,1]
  movement = max(||r_C||, ||r_U||); below move_pct percentile -> "static"
  label: static | shared (frac>=0.5) | crohn_unique (crohn_unique>uc_unique) | uc_unique
  uc_xstudy_cos[p] = cos(r_UC^study1[p], r_UC^study2[p])   (per-protein UC reproducibility; robustness col)

Sanity: UC doesn't move it (r_U=0) -> shared=0, fully crohn_unique; identical shifts -> shared=||r||, fully
shared; opposite shifts -> shared=0, divergent. Global headline: shared_fraction = sum 2*shared^2 / sum(||r_C||^2+||r_U||^2).

Validation is by EXTERNAL biological recovery (known pan-IBD vs Crohn/UC-specific genes); the UC cross-study
column is a cheap robustness readout, not a gate. NOTE: Crohn colon macrophage is a single study, so
crohn_unique proteins cannot be cross-study validated internally (UC can). See DISEASE_AXIS.md.

Output (results/<main>/disease_axis/):
  disease_axis_proteins.tsv  one row per protein (per-disease mags, shared_mag, shared_frac, xstudy cos, label,
                             rank_shared, rank_unique). movement + *_unique_mag are NOT emitted (derivable).
  disease_axis_summary.tsv   global shared fraction, label counts, mean UC reproducibility, disease cosine

Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/disease_axis_decompose.py \
        --main-name crohn_alzheimer_ild_uc_embedding_expressed_combat_loc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse(tag: str) -> dict:
    split = None
    for h in ("A", "B"):
        if tag.endswith(f"_split{h}"):
            split, tag = h, tag[: -len(f"_split{h}")]
            break
    p = tag.split("_")
    return dict(arm=p[0], study=p[1], tissue=p[2], ct=p[3], state="_".join(p[4:]),
                split=split, loo=p[1].startswith(("loopool", "loosingle")))


def main(main_name, tissue, ct, arm_a, arm_b, move_pct) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    pi = np.where(c["node_type"] == "protein")[0]
    Z, pres = c["Z"][:, pi, :], c["present"][:, pi]
    node_id = np.asarray(c["node_id"])[pi]
    tags = list(c["tags"])
    idx = {t: i for i, t in enumerate(tags)}
    P = {t: parse(t) for t in tags}

    # own-study healthy (allstates) lookup for this (tissue, ct): study -> tag
    H = {P[t]["study"]: t for t in tags
         if P[t]["arm"] == "healthy" and P[t]["tissue"] == tissue and P[t]["ct"] == ct
         and P[t]["state"] == "allstates" and not P[t]["loo"] and P[t]["split"] is None}

    def arm_studies(arm):
        """disease allstates nets for (arm, tissue, ct) that have a same-study healthy ref -> {study: dis_tag}."""
        out = {}
        for t in tags:
            q = P[t]
            if (q["arm"] == arm and q["tissue"] == tissue and q["ct"] == ct and q["state"] == "allstates"
                    and not q["loo"] and q["split"] is None and q["study"] in H):
                out[q["study"]] = t
        return out

    A, B = arm_studies(arm_a), arm_studies(arm_b)
    if not A or not B:
        raise SystemExit(f"need >=1 study each for {arm_a} and {arm_b} in {tissue}/{ct}; got {list(A)}, {list(B)}")

    # common protein mask: present in every contributing disease net AND its own-study healthy
    m = np.ones(Z.shape[1], bool)
    perstudy = {}                                                 # (arm, study) -> r vector (N, dim) full universe
    for arm, grp in ((arm_a, A), (arm_b, B)):
        for study, dt in grp.items():
            di, hi = idx[dt], idx[H[study]]
            m &= pres[di] & pres[hi]
            perstudy[(arm, study)] = Z[di] - Z[hi]

    def arm_shift(grp, arm):
        return np.mean([perstudy[(arm, s)][m] for s in grp], axis=0)     # mean over studies, on common mask

    rC = arm_shift(A, arm_a)
    rU = arm_shift(B, arm_b)
    mag_a = np.linalg.norm(rC, axis=1)
    mag_b = np.linalg.norm(rU, axis=1)

    # CO-MOVEMENT decomposition: shared = the part BOTH diseases move in the SAME direction. Project each
    # shift onto the consensus direction; shared = the common (min, clipped >=0) projection, so a protein
    # moved by only ONE disease -> shared 0 (NOT half-shared). unique = each shift minus that shared vector.
    csum = rC + rU
    chat = csum / (np.linalg.norm(csum, axis=1, keepdims=True) + 1e-12)
    pC = (rC * chat).sum(1)
    pU = (rU * chat).sum(1)
    shared = np.clip(np.minimum(pC, pU), 0.0, None)                # co-moving magnitude along the consensus
    shared_vec = shared[:, None] * chat
    cu = np.linalg.norm(rC - shared_vec, axis=1)                   # arm_a unique magnitude
    uu = np.linalg.norm(rU - shared_vec, axis=1)                   # arm_b unique magnitude
    shared_frac = (2 * shared ** 2) / (mag_a ** 2 + mag_b ** 2 + 1e-12)   # per-protein shared fraction in [0,1]
    disease_cos = (rC * rU).sum(1) / (mag_a * mag_b + 1e-9)               # signed cross-disease direction cosine [-1,1]

    # cross-study reproducibility per protein: mean pairwise cosine across an arm's studies (needs >=2)
    def xstudy(arm, grp):
        studs = list(grp)
        if len(studs) < 2:
            return np.full(int(m.sum()), np.nan)
        cs = []
        for i in range(len(studs)):
            for j in range(i + 1, len(studs)):
                r1, r2 = perstudy[(arm, studs[i])][m], perstudy[(arm, studs[j])][m]
                cs.append((r1 * r2).sum(1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-9))
        return np.mean(cs, axis=0)
    a_xcos = xstudy(arm_a, A)                                    # arm_a (e.g. crohn) cross-study reproducibility
    b_xcos = xstudy(arm_b, B)                                    # arm_b (e.g. uc) cross-study reproducibility

    move = np.maximum(mag_a, mag_b)
    floor = np.percentile(move, move_pct)
    label = np.where(move < floor, "static",
             np.where(shared_frac >= 0.5, "shared",
             np.where(cu > uu, f"{arm_a}_unique", f"{arm_b}_unique")))

    # NOTE: movement (=max(mag_a,mag_b)) and the two *_unique_mag columns are computed locally (used for the
    # static threshold, the label, and rank_unique) but deliberately NOT emitted -- they are derivable from the
    # retained columns and were dropped from the output table.
    df = pd.DataFrame({
        "protein": node_id[m], f"mag_{arm_a}": mag_a.round(4), f"mag_{arm_b}": mag_b.round(4),
        "shared_mag": shared.round(4), "shared_frac": shared_frac.round(4),
        "disease_cos": disease_cos.round(4),                              # +1 shared direction, -1 divergent (idea #1)
        f"{arm_a}_xstudy_cos": np.round(a_xcos, 4), f"{arm_b}_xstudy_cos": np.round(b_xcos, 4),
        "label": label,
    })
    dom_unique = np.maximum(cu, uu)
    df["rank_shared"] = df["shared_mag"].where(df.label != "static").rank(ascending=False, method="min")
    df["rank_unique"] = (pd.Series(dom_unique, index=df.index).where(df.label != "static")
                         .rank(ascending=False, method="min"))
    df = df.iloc[np.argsort(-move)].reset_index(drop=True)        # sort by movement (max-magnitude); col not emitted

    out = res / "disease_axis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "disease_axis_proteins.tsv", sep="\t", index=False)

    moving = move >= floor
    dcos = (float(((rC[moving] * rU[moving]).sum(1) / (mag_a[moving] * mag_b[moving] + 1e-9)).mean())
            if moving.any() else float("nan"))                    # disease cosine over MOVING proteins only

    def repro(xc, which):                                         # mean cross-study cosine over a label group (unsorted `label`)
        msk = (label == which) & ~np.isnan(xc)
        return round(float(xc[msk].mean()), 4) if msk.any() else float("nan")
    counts = df.label.value_counts().to_dict()
    summ = pd.DataFrame([{
        "tissue": tissue, "celltype": ct, "arm_a": arm_a, "arm_b": arm_b,
        f"{arm_a}_studies": len(A), f"{arm_b}_studies": len(B), "n_proteins": int(m.sum()),
        "shared_fraction_global": round(float((2 * shared ** 2).sum() / (mag_a ** 2 + mag_b ** 2).sum()), 4),
        "mean_disease_cos_moving": round(dcos, 4),
        # KEY VALIDATION: do the unique/shared proteins reproduce across that disease's independent studies?
        f"repro_{arm_a}_unique": repro(a_xcos, f"{arm_a}_unique"),
        f"repro_{arm_b}_unique": repro(b_xcos, f"{arm_b}_unique"),
        f"repro_shared_{arm_a}": repro(a_xcos, "shared"),
        f"repro_shared_{arm_b}": repro(b_xcos, "shared"),
        "n_shared": counts.get("shared", 0), f"n_{arm_a}_unique": counts.get(f"{arm_a}_unique", 0),
        f"n_{arm_b}_unique": counts.get(f"{arm_b}_unique", 0), "n_static": counts.get("static", 0),
    }])
    summ.to_csv(out / "disease_axis_summary.tsv", sep="\t", index=False)
    print(f"{arm_a} studies={list(A)}  {arm_b} studies={list(B)}  common proteins={int(m.sum())}\n")
    print(f"wrote {out/'disease_axis_proteins.tsv'} ({len(df)} proteins)")
    print(f"wrote {out/'disease_axis_summary.tsv'}\n")
    print(summ.to_string(index=False))
    print("\ntop shared:", list(df[df.label == "shared"].nlargest(12, "shared_mag").protein))
    print(f"top {arm_a}_unique:", list(df[df.label == f"{arm_a}_unique"].nsmallest(12, "rank_unique").protein))
    print(f"top {arm_b}_unique:", list(df[df.label == f"{arm_b}_unique"].nsmallest(12, "rank_unique").protein))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed_combat_loc")
    ap.add_argument("--tissue", default="colon")
    ap.add_argument("--celltype", default="macrophage")
    ap.add_argument("--arm-a", default="crohn")
    ap.add_argument("--arm-b", default="uc")
    ap.add_argument("--move-pct", type=float, default=50.0,
                    help="proteins below this percentile of max-movement are labelled 'static' (not ranked)")
    a = ap.parse_args()
    raise SystemExit(main(a.main_name, a.tissue, a.celltype, a.arm_a, a.arm_b, a.move_pct))
