"""Step 3b — supervised drug-target classifier from the cisTarget context embedding, PER CONTEXT.
See SEQ_CONTEXT_EMBED.md.

Per disease context: label = that disease's known drug targets (all phases). Features EMB / BASE(degree+DE) /
EMB+BASE / ESM. Model = L2 logistic regression (balanced), standardized. 5x5 stratified CV AUC. Prints a table.
The bar: EMB+BASE > BASE.

Run: .venv_scvi/bin/python mlp_mods/seq_context/validation/target_classifier.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

SEQ = Path("mlp_mods/seq_context")
NET = SEQ / "scenic/networks"
DEPPI_NET = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed_scvi/networks")
ESM_ALL = torch.load("ESM/protein_embeddings.pt", map_location="cpu")
PROT = sorted(ESM_ALL.keys())
DRUG = {"crohn": "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0000384.tsv",
        "uc":    "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0000729.tsv",
        "ild":   "mlp_mods/03_opentargets_rebuild/known_drugs_EFO_0004244.tsv"}


def expr(ctx):
    f = DEPPI_NET / ctx / "network_nodes.tsv"
    if not f.exists():
        return {}
    df = pd.read_csv(f, sep="\t"); return dict(zip(df.node_id, df.expression))


def build(ctx, d, arm=None):
    m = d["context"] == ctx
    genes = np.array([PROT[i] for i in d["prot_idx"][m]])
    EMB = d["emb"][m].astype(np.float64)
    ESM = np.stack([ESM_ALL[g].numpy() for g in genes]).astype(np.float64)
    ct = pd.read_csv(NET / ctx / "edges_cistarget.tsv", sep="\t")
    outdeg = ct.tf.value_counts(); indeg = ct.target.value_counts()
    a = ctx.split("_"); healthy = f"healthy_{a[1]}_macrophage_{'_'.join(a[3:])}"
    ed, eh = expr(ctx), expr(healthy)
    base = np.array([[np.log1p(outdeg.get(g, 0)), np.log1p(indeg.get(g, 0)),
                      ed.get(g, 0.0), abs(ed.get(g, 0.0) - eh.get(g, 0.0))] for g in genes])
    kd = pd.read_csv(DRUG[arm or a[0]], sep="\t")   # arm overrides -> FIX label to one disease across contexts
    pos = set(kd.gene_symbol.astype(str))
    y = np.array([1 if g in pos else 0 for g in genes])
    return {"BASE": base, "EMB": EMB, "ESM": ESM}, y, genes


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000, C=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="link_v4_cistarget")
    ap.add_argument("--fix-disease", default=None, choices=["crohn", "uc", "ild"],
                    help="fix the target LABEL to this disease across ALL contexts (isolates context relevance)")
    args = ap.parse_args()
    d = np.load(SEQ / "results" / args.run / "embeddings.npz", allow_pickle=True)
    ctxs = (sorted(set(d["context"])) if args.fix_disease
            else [c for c in sorted(set(d["context"])) if c.split("_")[0] in DRUG])
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    if args.fix_disease:
        print(f"FIXED label = {args.fix_disease} drug targets, scored across all contexts", flush=True)

    # OT-association ("implicated in the disease") gene sets, score_indirect > 0.1
    OT = {"crohn": "mlp_mods/opentargets_associations/crohn_target_association_EFO_0000384.tsv",
          "uc": "mlp_mods/opentargets_associations/uc_target_association_EFO_0000729.tsv",
          "ild": "mlp_mods/opentargets_associations/ild_target_association_EFO_0004244.tsv"}
    ot_impl = {a: set(pd.read_csv(f, sep="\t").query("score_indirect > 0.1").gene_symbol.astype(str))
               for a, f in OT.items()}
    rows = []
    for ctx in ctxs:
        X, y, genes = build(ctx, d, arm=args.fix_disease)
        if y.sum() < 5:
            continue
        present = set(genes[y == 1])
        row_disease = ctx.split("_")[0]                       # the CONTEXT's disease (may differ from label)
        impl = len(present & ot_impl[row_disease]) if row_disease in ot_impl else -1
        r = {"context": ctx.replace("_macrophage", ""), "n_pos": int(y.sum()),
             "impl_ctxdis": impl}
        for fs in ["BASE", "EMB", "ESM"]:
            r[fs] = float(cross_val_score(clf(), X[fs], y, cv=rskf, scoring="roc_auc").mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    print("\n=== Step 3b: drug-target classifier per context — 5x5 CV AUC ===", flush=True)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\nMEDIAN AUC: BASE={df.BASE.median():.3f} EMB={df.EMB.median():.3f} ESM={df.ESM.median():.3f} | "
          f"contexts EMB>BASE: {int((df.EMB>df.BASE).sum())}/{len(df)}", flush=True)
    df.to_csv(SEQ / "validation" / f"classifier_percontext_{args.run}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
