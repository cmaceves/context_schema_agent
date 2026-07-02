"""Control networks for the PINNACLE cell-type build (build_pinnacle_ct.py), matching its scheme:
node set = PINNACLE(celltype) ∪ expressed(arm) ∩ OmniPath-incident, OmniPath directed edges, neutral weights,
ComBat-corrected log1p(mean CP10k) feature. CELL-TYPE only (no state), so the a-j ladder minus the cell-STATE
control (c). Per (tissue,celltype) group, per (arm, study):

  <arm>_<study8>_<tissue>_<celltype>_allstates            per-study pooled net  -> b/i between-study, d/j celltype, e tissue, g disease
  <arm>_<study8>_<tissue>_<celltype>_allstates_split{A,B} donor halves          -> a/h inter-donor floor

Expression is ComBat-corrected exactly as in the main build (per (tissue,celltype): each (arm,study) pseudobulk
is a sample, batch=study, arm preserved, location-only). A per-study net gets its own corrected value; a donor
half gets its raw pseudobulk plus that study's per-gene ComBat delta. Node set for an arm is fixed across its
studies (only the feature varies) — the control design.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/build_pinnacle_controls.py \
        --out crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct
Then: infer_controls.py -> compare_controls.py / control_centered.py -> disease_axis_decompose.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "mlp_mods/de_ppi/scripts/build")
sys.path.insert(0, "mlp_mods/de_ppi/scripts/build/controls")
from apply_combat_expression import combat_ls
from build_pinnacle_ct import (GROUPS, LABEL2ARM, MIN_CELLS, CP10K_CUTOFF, UNEXPRESSED_FILL, OMNI, RES,
                               load_atlas, pinn_nodes)

MIN_SPLIT_CELLS = 30


def main(out: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    omni = pd.read_csv(OMNI, sep="\t", usecols=["src", "dst"])
    omni = omni[omni.src != omni.dst].drop_duplicates()
    incident = set(omni.src) | set(omni.dst)
    out_root = RES / out / "controls" / "networks"; out_root.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for (tissue, ct), atlases in GROUPS.items():
        # gather per (arm, study) cell index lists (pool states), deduped by (study, arm) across atlases
        cellsets, seen = [], set()                                 # each: dict(arm, study, base, rowmask, meta_donor)
        for base in atlases:
            meta, cp10k, var = load_atlas(base)
            for lbl, arm in LABEL2ARM.items():
                sel_lbl = meta["disease"].astype(str) == lbl
                if not sel_lbl.any():
                    continue
                for study in meta.loc[sel_lbl, "dataset_id"].astype(str).unique():
                    if (study, arm) in seen:
                        continue
                    mask = (sel_lbl & (meta["dataset_id"].astype(str) == study)).to_numpy()
                    if mask.sum() < MIN_CELLS:
                        continue
                    cellsets.append(dict(arm=arm, study=study, base=base, mask=mask,
                                         donors=meta.loc[mask, "donor_id"].astype(str).to_numpy()))
                    seen.add((study, arm))
        if not cellsets:
            continue

        def pb(base, mask):                                        # log1p(mean CP10k) Series over that atlas' genes
            _, cp10k, var = load_atlas(base)
            return pd.Series(np.log1p(np.asarray(cp10k[mask].mean(0)).ravel()), index=var)

        raw = [pb(c["base"], c["mask"]) for c in cellsets]
        Mexpr = pd.DataFrame({i: r for i, r in enumerate(raw)}).T
        smeta = pd.DataFrame([(c["arm"], c["study"]) for c in cellsets], columns=["arm", "study"])
        arms = sorted(smeta.arm.unique())

        expr_arm = {a: set(np.expm1(Mexpr.loc[smeta.index[smeta.arm == a]]).mean(0)
                           .pipe(lambda s: s.index[s >= CP10K_CUTOFF])) for a in arms}
        pn = pinn_nodes(ct)
        ug = [g for g in (pn | set().union(*expr_arm.values())) & incident if g in Mexpr.columns]

        X = Mexpr[ug].to_numpy()
        Xc = combat_ls(X, smeta.study.to_numpy(), smeta[["arm"]], scale=False) if smeta.study.nunique() > 1 else X
        delta = pd.DataFrame(Xc - X, columns=ug)                  # per-sample per-gene ComBat shift

        nodes_by_arm = {a: sorted((pn | expr_arm[a]) & incident) for a in arms}
        edges_by_arm = {a: omni[omni.src.isin(nodes_by_arm[a]) & omni.dst.isin(nodes_by_arm[a])] for a in arms}

        def write(tag, arm, feat: pd.Series):
            nodes = nodes_by_arm[arm]; e = edges_by_arm[arm]
            d = out_root / tag; d.mkdir(parents=True, exist_ok=True)
            src_col = ["expressed" if g in expr_arm[arm] else "pinnacle" for g in nodes]
            fv = feat.reindex(nodes).fillna(0.0).values
            expr_vals = [float(v) if s == "expressed" else UNEXPRESSED_FILL for v, s in zip(fv, src_col)]
            pd.DataFrame({"node_id": nodes, "node_type": "protein", "source": src_col, "direction": "",
                          "sender_weight": 1.0, "expression": expr_vals}
                         ).to_csv(d / "network_nodes.tsv", sep="\t", index=False)
            pd.DataFrame({"source": e.src, "target": e.dst, "edge_origin": "OmniPath", "edge_property": "",
                          "weight": 1.0, "direction": ""}).to_csv(d / "network_edges.tsv", sep="\t", index=False)

        d_ug = {g: i for i, g in enumerate(ug)}
        for i, c in enumerate(cellsets):
            arm, s8 = c["arm"], c["study"][:8]
            dser = pd.Series(delta.iloc[i].values, index=ug)
            corrected = raw[i].add(dser, fill_value=0.0)           # per-study corrected feature
            write(f"{arm}_{s8}_{tissue}_{ct}_allstates", arm, corrected)
            n_written += 1
            # donor split halves (inter-donor floor)
            ud = np.unique(c["donors"])
            if len(ud) >= 2:
                perm = rng.permutation(ud); halves = {"A": set(perm[: len(ud) // 2]), "B": set(perm[len(ud) // 2:])}
                for h, dset in halves.items():
                    hmask = c["mask"].copy()
                    hmask[hmask] = np.isin(c["donors"], list(dset))
                    if hmask.sum() < MIN_SPLIT_CELLS:
                        continue
                    hfeat = pb(c["base"], hmask).add(dser, fill_value=0.0)   # half raw + study ComBat delta
                    write(f"{arm}_{s8}_{tissue}_{ct}_allstates_split{h}", arm, hfeat)
                    n_written += 1
        print(f"  {tissue}/{ct}: {len(cellsets)} (arm,study) samples, arms={arms}", flush=True)

    print(f"\nwrote {n_written} control nets -> {out_root}")
    print(f"next: infer_controls.py --encoder {RES/out}/encoder.pt --networks {out_root} "
          f"--out {RES/out}/controls/control_embeddings.npz")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crohn_alzheimer_ild_uc_embedding_pinnacle_combat_ct")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raise SystemExit(main(a.out, a.seed))
