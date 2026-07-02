"""ComBat-correct the per-network expression feature across studies, then write a NEW build dir.

The encoder's only study-varying input is the per-node `expression` feature (log1p mean CP10k); edges
(OmniPath) and healthy sender-weights (1.0) are study-independent. So removing the between-study expression
batch is the lever. This script:

  1. Reads the source build's main networks (results/<src>/networks) and per-study control networks
     (results/<src>/controls/networks). Each carries an `expression` column = a per-study pseudobulk.
  2. Per (tissue, celltype) GROUP, assembles a gene x sample matrix from the SINGLE-study control networks
     (batch = study8; biological covariates arm + state are PRESERVED). Running ComBat per group keeps
     tissue/celltype constant, sidestepping the study<->context rank-deficiency. Groups with <2 studies are
     left uncorrected (no between-study batch to estimate).
  3. Applies a ComBat batch adjustment per group (Johnson et al. 2007, WITHOUT empirical-Bayes shrinkage --
     see combat_ls). The non-iterative form is used because scanpy's parametric EB solver fails to converge
     (hangs) on these small, sparse pseudobulk groups; lstsq makes it robust to arm/batch collinearity.
     arm + state are preserved as covariates. DEFAULT is LOCATION-ONLY (remove the additive per-batch mean
     gamma only): at our batch sizes (2-3 samples) the multiplicative scale (delta) estimate is unreliable
     and adds over-shrink risk with no measurable benefit (location-only and location+scale gave near-
     identical ladders -- see CONTROLS.md). Pass --with-scale to also apply delta.
  4. Writes corrected expression into a NEW build dir <dst> (node/edge sets copied unchanged):
       - single-study control nets -> their own ComBat-corrected expression
       - pooled main nets / loopool -> raw expression + mean per-gene ComBat DELTA over their constituent
         studies (preserves the net's own magnitude; applies only the learned batch shift)

After this: retrain the encoder on <dst>, re-infer controls, re-run compare_controls/control_centered.

Run: .venv/bin/python mlp_mods/de_ppi/scripts/build/controls/apply_combat_expression.py \
        --src crohn_alzheimer_ild_uc_embedding_expressed \
        --dst crohn_alzheimer_ild_uc_embedding_expressed_combat
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("mlp_mods/de_ppi/results")


def combat_ls(X: np.ndarray, batch: np.ndarray, cov: pd.DataFrame, scale: bool = False) -> np.ndarray:
    """ComBat location/scale batch adjustment (Johnson et al. 2007), WITHOUT empirical-Bayes shrinkage.

    Deterministic, non-iterative (so it can't hang the way scanpy's parametric EB solver does on small
    sparse pseudobulk groups), and robust to covariate/batch collinearity via least-squares. `cov` holds the
    biological covariates to PRESERVE (arm, state); their effects are kept, only the per-batch location (γ)
    and (if scale=True) scale (δ) are removed. With scale=False this is LOCATION-ONLY: only the additive
    per-batch mean is removed, the within-batch spread is left untouched -- safer at small n/batch where the
    δ variance estimate is unreliable and can over-shrink between-study spread. X is (n_samples, n_genes);
    batch is the per-sample study label.
    """
    n, G = X.shape
    Xb = pd.get_dummies(pd.Series(batch, name="b")).astype(float)        # full batch one-hot (no intercept)
    levels = list(Xb.columns)
    Xc = pd.get_dummies(cov.astype(str), drop_first=True).astype(float) if cov.shape[1] else pd.DataFrame(index=range(n))
    design = np.hstack([Xb.values, Xc.values]) if Xc.shape[1] else Xb.values
    nb = Xb.shape[1]
    B, *_ = np.linalg.lstsq(design, X, rcond=None)                       # (p, G) coefficients per gene
    grand = (Xb.values.sum(0) / n) @ B[:nb]                              # batch-size-weighted grand mean (G,)
    stand_mean = np.tile(grand, (n, 1))
    if Xc.shape[1]:
        stand_mean += Xc.values @ B[nb:]
    var_pooled = ((X - design @ B) ** 2).mean(0)                         # residual variance per gene
    var_pooled[var_pooled <= 0] = 1.0
    sd = np.sqrt(var_pooled)
    Z = (X - stand_mean) / sd
    out = Z.copy()
    for b in levels:
        idx = np.where(batch == b)[0]
        gamma = Z[idx].mean(0)
        if scale:
            delta = Z[idx].var(0)
            delta[delta <= 0] = 1.0
            out[idx] = (Z[idx] - gamma) / np.sqrt(delta)
        else:
            out[idx] = Z[idx] - gamma                            # location-only: remove additive batch mean
    return out * sd + stand_mean


def parse_ctrl(tag: str) -> dict:
    """control tag -> fields. Handles _split{A,B}, allstates, loopool/loosingle prefixes on the study token."""
    split = None
    for h in ("A", "B"):
        if tag.endswith(f"_split{h}"):
            split, tag = h, tag[: -len(f"_split{h}")]
            break
    p = tag.split("_")
    studytok = p[1]
    pooled = studytok.startswith("loopool")                       # multi-study pool -> excluded from fit
    study = studytok
    for pre in ("loopool", "loosingle"):
        if studytok.startswith(pre):
            study = studytok[len(pre):]
            break
    return dict(arm=p[0], study=study, studytok=studytok, tissue=p[2], ct=p[3],
                state="_".join(p[4:]), split=split, pooled=pooled)


def parse_main(tag: str) -> dict:
    p = tag.split("_")
    return dict(arm=p[0], tissue=p[1], ct=p[2], state="_".join(p[3:]))


def read_expr(ndir: Path) -> pd.Series:
    df = pd.read_csv(ndir / "network_nodes.tsv", sep="\t", keep_default_na=False)
    return pd.Series(df["expression"].astype(float).values, index=df["node_id"].values)


def write_expr(ndir: Path, corrected: pd.Series) -> None:
    """Overwrite only the `expression` column; nodes without a corrected value keep their original."""
    df = pd.read_csv(ndir / "network_nodes.tsv", sep="\t", keep_default_na=False)
    new = df["node_id"].map(corrected)
    df["expression"] = new.fillna(df["expression"].astype(float)).astype(float)
    df.to_csv(ndir / "network_nodes.tsv", sep="\t", index=False)


def combat_group(items: list[tuple[str, dict, pd.Series]], scale: bool = False) -> tuple[dict, dict]:
    """Run ComBat on one (tissue,celltype) group. Returns (corrected_by_tag, delta_primary).

    corrected_by_tag: tag -> corrected expression Series (genes).
    delta_primary: (arm,tissue,ct,state) -> list of (study, delta Series) from PRIMARY per-study nets
                   (non-split, non-allstates, plain dataset) -- used to shift the pooled mains/loopool.
    """
    corrected, delta_primary = {}, defaultdict(list)
    genes = sorted(set().union(*[s.index for _, _, s in items]))
    raw = np.vstack([s.reindex(genes).fillna(0.0).values for _, _, s in items])
    batch = [pc["study"] for _, pc, _ in items]
    bc = pd.Series(batch).value_counts()
    fit_mask = np.array([bc[b] >= 2 for b in batch])              # ComBat needs >=2 samples / batch
    n_batches = bc[bc >= 2].shape[0]
    if n_batches < 2:                                             # nothing to harmonize across studies
        for (tag, _, s) in items:
            corrected[tag] = s
        return corrected, delta_primary, "skip(<2 studies)"

    bvec = np.array(batch)[fit_mask].astype(str)
    cov = pd.DataFrame({"arm": np.array([pc["arm"] for _, pc, _ in items])[fit_mask].astype(str),
                        "state": np.array([pc["state"] for _, pc, _ in items])[fit_mask].astype(str)})
    Xfit = raw[fit_mask]
    corr = combat_ls(Xfit, bvec, cov, scale=scale)               # ComBat L/S (no EB) -- deterministic, no hang
    if np.isnan(corr).any():                                     # numerical fallback -> leave raw
        for (tag, _, s) in items:
            corrected[tag] = s
        return corrected, delta_primary, "nan->raw"

    fit_items = [it for it, keep in zip(items, fit_mask) if keep]
    for i, (tag, pc, _) in enumerate(fit_items):
        cs = pd.Series(corr[i], index=genes)
        corrected[tag] = cs
        is_primary = (pc["split"] is None and pc["state"] != "allstates"
                      and not pc["studytok"].startswith("loo"))
        if is_primary:
            delta = cs - pd.Series(Xfit[i], index=genes)
            delta_primary[(pc["arm"], pc["tissue"], pc["ct"], pc["state"])].append((pc["study"], delta))
    for (tag, _, s) in (it for it, keep in zip(items, fit_mask) if not keep):
        corrected[tag] = s                                       # singleton-batch nets: uncorrected
    return corrected, delta_primary, f"ok studies={n_batches} fit={len(fit_items)}/{len(items)}"


def mean_delta(deltas: list[tuple[str, pd.Series]], drop_study: str | None = None) -> pd.Series:
    sub = [d for st, d in deltas if drop_study is None or st != drop_study]
    if not sub:
        return pd.Series(dtype=float)
    return pd.concat(sub, axis=1).mean(axis=1)


def shifted(raw: pd.Series, d: pd.Series) -> pd.Series:
    """raw expression + per-gene delta (delta 0 for genes absent from the fit), index-aligned to raw."""
    if not len(d):
        return raw
    return raw + pd.Series([float(d.get(g, 0.0)) for g in raw.index], index=raw.index)


def main(src: str, dst: str, scale: bool = False) -> int:
    src_root, dst_root = RES / src, RES / dst
    src_nets, src_ctrl = src_root / "networks", src_root / "controls" / "networks"
    dst_nets, dst_ctrl = dst_root / "networks", dst_root / "controls" / "networks"
    assert src_nets.exists() and src_ctrl.exists(), f"missing source networks under {src_root}"

    print(f"copying networks -> {dst_root}", flush=True)
    for s, d in ((src_nets, dst_nets), (src_ctrl, dst_ctrl)):
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)

    main_tags = sorted(p.name for p in dst_nets.iterdir() if (p / "network_nodes.tsv").exists())
    ctrl_tags = sorted(p.name for p in dst_ctrl.iterdir() if (p / "network_nodes.tsv").exists())

    # ---- fit ComBat per (tissue, celltype) on single-study control nets (exclude loopool) ----
    groups: dict[tuple, list] = defaultdict(list)
    for t in ctrl_tags:
        pc = parse_ctrl(t)
        if pc["pooled"]:
            continue
        groups[(pc["tissue"], pc["ct"])].append((t, pc, read_expr(dst_ctrl / t)))

    corrected_ctrl: dict[str, pd.Series] = {}
    delta_primary: dict[tuple, list] = defaultdict(list)
    print(f"\nComBat per (tissue, celltype)  [{'location+scale' if scale else 'LOCATION-ONLY'}]:", flush=True)
    for key in sorted(groups):
        corr, dprim, msg = combat_group(groups[key], scale=scale)
        corrected_ctrl.update(corr)
        for k, v in dprim.items():
            delta_primary[k].extend(v)
        print(f"  {key[0]+'/'+key[1]:24s} {msg}", flush=True)

    # ---- write corrected expression ----
    # control nets: own corrected (loopool: raw + mean delta over healthy primaries excluding held-out study)
    for t in ctrl_tags:
        pc = parse_ctrl(t)
        if pc["pooled"]:
            d = mean_delta(delta_primary.get(("healthy", pc["tissue"], pc["ct"], pc["state"]), []),
                           drop_study=pc["study"])
            write_expr(dst_ctrl / t, shifted(read_expr(dst_ctrl / t), d))
        else:
            write_expr(dst_ctrl / t, corrected_ctrl[t])

    # main nets (pooled): raw + mean delta over constituent studies of (arm,tissue,ct,state)
    n_corr_main = 0
    for t in main_tags:
        pm = parse_main(t)
        d = mean_delta(delta_primary.get((pm["arm"], pm["tissue"], pm["ct"], pm["state"]), []))
        if len(d):
            write_expr(dst_nets / t, shifted(read_expr(dst_nets / t), d))
            n_corr_main += 1
        # else: single-study context (no between-study delta) -> unchanged
    print(f"\nwrote corrected expression: {len(ctrl_tags)} control nets, "
          f"{n_corr_main}/{len(main_tags)} main nets shifted (rest single-study)", flush=True)
    print(f"new build dir: {dst_root}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crohn_alzheimer_ild_uc_embedding_expressed")
    ap.add_argument("--dst", default="crohn_alzheimer_ild_uc_embedding_expressed_combat")
    ap.add_argument("--with-scale", action="store_true",
                    help="also apply the multiplicative scale (delta) step; default is LOCATION-ONLY "
                         "(remove only the additive per-batch mean gamma -- the scale step adds risk, not "
                         "signal, at our small batch sizes; see CONTROLS.md)")
    a = ap.parse_args()
    raise SystemExit(main(a.src, a.dst, scale=a.with_scale))
