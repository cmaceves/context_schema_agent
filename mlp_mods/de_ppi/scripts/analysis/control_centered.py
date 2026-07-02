"""Context-matched (local) centering of the control ladder. Instead of one GLOBAL consensus, each control
pair is centered on the MAIN network(s) of its held-constant context, so the cosine isolates the VARIED
factor (study/disease/state) rather than being dominated by cell-type/state/arm structure.

Centroid is always a MAIN network's Z (embeddings.npz) for the matched context (arm, tissue, celltype, state):
  - healthy controls h,i,m            -> healthy-main of context
  - disease controls a,b,c            -> TWO variants: disease-main of context, AND healthy-main of context
  - c (varies state)                  -> centroid = MEAN of the per-state mains of (arm,tissue,celltype)
  - different_disease (renamed g)     -> healthy-main of context (cross-disease; no disease centroid)
  - d (cell type), e (tissue)         -> direct MAIN-vs-MAIN cosine, no centroid (d over node intersection)

For a centered control: r_X = Z_X - centroid (over proteins present in both compared nets, present in the
centroid, and valid); cosine = mean per-protein cos(r_A, r_B). Pairing reuses compare_controls.classify.

Output: STANDALONE direction table controls/control_centered_summary.tsv (`ctxcos_<centroid>` /
`ctxn_<centroid>` per control), plus per-pair detail in controls/control_centered_pairs.tsv. The headline
controls/control_summary.tsv is magnitude-only and is NOT touched by this script (direction is kept separate).
Run: .venv/bin/python mlp_mods/de_ppi/scripts/analysis/control_centered.py --main-name <build>
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DISEASES = {"crohn", "uc", "alz", "ild"}


def parse(tag):
    """control/main tag -> dict. control: <arm>_<study8>_<tissue>_<ct>_<state>[_split{A,B}|_allstates].
    main:   <arm>_<tissue>_<ct>_<state>.  arm = disease slug or 'healthy'."""
    split = None
    for h in ("A", "B"):
        if tag.endswith(f"_split{h}"):
            split, tag = h, tag[: -len(f"_split{h}")]; break
    p = tag.split("_")
    return p, split


def parse_main(tag):
    p = tag.split("_")
    return p[0], p[1], p[2], "_".join(p[3:])          # arm, tissue, ct, state


def parse_ctrl(tag):
    p, split = parse(tag)
    return dict(arm=p[0], study=p[1], tissue=p[2], celltype=p[3], state="_".join(p[4:]), split=split)


def cos_dev(Za, Zb, ca, cb, mask):
    ra, rb = Za[mask] - ca[mask], Zb[mask] - cb[mask]
    na, nb = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1)
    return float(((ra * rb).sum(1) / (na * nb + 1e-9)).mean())


def main(main_name) -> int:
    res = Path("mlp_mods/de_ppi/results") / main_name
    m = np.load(res / "embeddings.npz", allow_pickle=True)
    c = np.load(res / "controls" / "control_embeddings.npz", allow_pickle=True)
    assert list(m["node_id"]) == list(c["node_id"])
    is_prot = m["node_type"] == "protein"; pi = np.where(is_prot)[0]
    Zm, presm, mtags = m["Z"][:, pi, :], m["present"][:, pi], list(m["tags"])
    valid = presm.any(axis=0)
    # MAIN lookup: (arm,tissue,ct,state) -> row;  and (arm,tissue,ct) -> list of rows (for c mean-of-states)
    main_row, main_ct = {}, {}
    for i, t in enumerate(mtags):
        arm, tis, ct, st = parse_main(t)
        main_row[(arm, tis, ct, st)] = i
        main_ct.setdefault((arm, tis, ct), []).append(i)

    def centroid(arm, tis, ct, st):
        i = main_row.get((arm, tis, ct, st))
        return (Zm[i], presm[i]) if i is not None else (None, None)

    def centroid_states(arm, tis, ct):               # mean of per-state mains (control c)
        rows = main_ct.get((arm, tis, ct))
        if not rows:
            return None, None
        sub, pr = Zm[rows], presm[rows]
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cen = np.nanmean(np.where(pr[:, :, None], sub, np.nan), axis=0)
        return np.nan_to_num(cen), pr.any(axis=0)

    ctags = list(c["tags"]); Zc, presc = c["Z"][:, pi, :], c["present"][:, pi]
    cti = {t: i for i, t in enumerate(ctags)}
    P = {t: parse_ctrl(t) for t in ctags if "_loo" not in t}

    # import the a-j classifier (same pairing) from compare_controls
    import sys as _s
    _s.path.insert(0, str(Path("mlp_mods/de_ppi/scripts/analysis")))
    from compare_controls import classify

    rows = []

    def add(control, centroid_label, ia, ib, ca, cpa, cb, cpb):
        both = presc[ia] & presc[ib] & cpa & cpb & valid
        if both.sum() < 20:
            return
        rows.append(dict(control=control, centroid=centroid_label, network_a=ctags[ia], network_b=ctags[ib],
                         n=int(both.sum()), cos=round(cos_dev(Zc[ia], Zc[ib], ca, cb, both), 4)))

    # ---- pairwise centered controls a,b,c,(different_disease),h,i,m ----
    keys = list(P)
    for ta, tb in combinations(keys, 2):
        a, b = P[ta], P[tb]
        cls = classify(dict(disease=a["arm"], dataset=a["study"], tissue=a["tissue"], celltype=a["celltype"],
                            state=a["state"], split=a["split"]),
                       dict(disease=b["arm"], dataset=b["study"], tissue=b["tissue"], celltype=b["celltype"],
                            state=b["state"], split=b["split"]))
        # a/h: donor-split cosine degenerate (magnitude only). i/m: only an own-arm centroid exists ->
        # degenerate -> blanked. So only cross-arm/meaningful centerings are kept below.
        if cls in (None, "a", "d", "e", "h", "i"):
            continue
        ia, ib = cti[ta], cti[tb]
        if cls in ("g", "b"):                         # different_disease / between_study -> CROSS-ARM healthy-main
            cen, cp = centroid("healthy", a["tissue"], a["celltype"], a["state"])
            if cen is not None:
                add(cls, "healthy_main", ia, ib, cen, cp, cen, cp)
        elif cls == "c":                              # cell_state -> mean of per-state mains (disease + healthy)
            dz, hp = centroid_states(a["arm"], a["tissue"], a["celltype"])
            if dz is not None:
                add(cls, "disease_state_mean", ia, ib, dz, hp, dz, hp)
            cen, cp = centroid_states("healthy", a["tissue"], a["celltype"])
            if cen is not None:
                add(cls, "healthy_state_mean", ia, ib, cen, cp, cen, cp)

    # ---- d (cell type) / e (tissue): direct MAIN-vs-MAIN, no centroid ----
    for ta, tb in combinations(range(len(mtags)), 2):
        aA = parse_main(mtags[ta]); aB = parse_main(mtags[tb])
        armA, tisA, ctA, stA = aA; armB, tisB, ctB, stB = aB
        if armA == "healthy" or armB == "healthy":
            continue
        both = presm[ta] & presm[tb]
        if both.sum() < 20:
            continue
        za, zb = Zm[ta][both], Zm[tb][both]
        cs = float(((za * zb).sum(1) / (np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1) + 1e-9)).mean())
        if armA == armB and tisA == tisB and stA == stB and ctA != ctB:        # d: vary cell type
            rows.append(dict(control="d", centroid="none(direct)", network_a=mtags[ta], network_b=mtags[tb],
                             n=int(both.sum()), cos=round(cs, 4)))
        elif armA == armB and ctA == ctB and stA == stB and tisA != tisB:      # e: vary tissue
            rows.append(dict(control="e", centroid="none(direct)", network_a=mtags[ta], network_b=mtags[tb],
                             n=int(both.sum()), cos=round(cs, 4)))

    # (control m: only an own-arm healthy-main centroid exists -> degenerate -> blanked; its global-consensus
    #  value lives in control_summary via control_m_healthy_loo.py.)

    df = pd.DataFrame(rows)
    out = res / "controls"; df.to_csv(out / "control_centered_pairs.tsv", sep="\t", index=False)
    # pivot context-matched cosines to wide (one column per centroid) -> STANDALONE direction table.
    # This is the "direction" control set; the magnitude-only control_summary.tsv is left untouched.
    piv = df.pivot_table(index="control", columns="centroid", values="cos", aggfunc="mean").round(4)
    piv.columns = [f"ctxcos_{c}" for c in piv.columns]
    npiv = df.pivot_table(index="control", columns="centroid", values="cos", aggfunc="size").astype("Int64")
    npiv.columns = [f"ctxn_{c}" for c in npiv.columns]
    from compare_controls import CLASS_NAME
    ctx = piv.join(npiv).reset_index()
    ctx.insert(1, "control_name", ctx.control.map(CLASS_NAME))
    ctx.to_csv(out / "control_centered_summary.tsv", sep="\t", index=False)
    print(f"wrote {out/'control_centered_summary.tsv'} (direction; magnitude stays in control_summary.tsv)\n")
    print(ctx.to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-name", default="crohn_alzheimer_ild_uc_embedding_expressed")
    raise SystemExit(main(ap.parse_args().main_name))
