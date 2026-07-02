"""ONE-OFF (delete me): in-fold probe for ITGA4/IFNAR1.
Trains one MLP on ALL instances (incl. these genes) with the same settings as the pipeline, then
scores them. Compares in-fold vs out-of-fold (OOF) to tell generalization gap from un-representable."""
import sys, json
sys.path.insert(0, "mlp_mods/de_ppi/scripts/test_mlp")
import numpy as np, pandas as pd, torch
import train_target_mlp as T

build = "crohn_alzheimer_ild_uc_embedding_expressed"
labels_by_dis = {dis: T.load_efo(code) for dis, code in T.DISEASE_EFO.items()}
ibd_pos = json.load(open(T.OT / f"positive_proteins_{T.IBD}_new_phase2.json"))[T.IBD]
d = np.load(T.ROOT / f"de_ppi/results/{build}/embeddings.npz", allow_pickle=True)
tags, Z, present = list(d["tags"]), d["Z"], d["present"]
ids = np.array(d["node_id"], dtype=object); isp = d["node_type"] == "protein"

feats, labels, genes, ctxs = [], [], [], []
for ti, t in enumerate(tags):
    dis = T.disease_of(t); ck = T.celltype_of(t)
    if dis is None or ck is None:
        continue
    pos, neg = labels_by_dis[dis]
    if ck not in pos:
        continue
    P, N = set(pos[ck]), set(neg.get(ck, []))
    if dis in ("crohn", "uc"):
        N -= set(ibd_pos.get(ck, []))
    for pi in np.where(present[ti] & isp)[0]:
        g = ids[pi]; lab = 1 if g in P else (0 if g in N else None)
        if lab is None:
            continue
        feats.append(Z[ti, pi]); labels.append(lab); genes.append(g); ctxs.append(t)
X = np.asarray(feats, np.float32); y = np.asarray(labels)
genes = np.asarray(genes, object); ctxs = np.asarray(ctxs, object)

dev = torch.device("cpu")
# IN-FOLD: train on everything (genes of interest INCLUDED), same hyperparams/seed as pipeline
model = T.train_fold(X, y, dev, epochs=300, lr=1e-3, hidden=128, seed=3, neg_ratio=25)
infold = T.predict(model, X, dev)
from sklearn.metrics import roc_auc_score
print(f"in-fold AUROC on all training data = {roc_auc_score(y, infold):.3f}\n")

oof = pd.read_csv("mlp_mods/de_ppi/results/_tmp_mlp/predictions_per_instance.tsv", sep="\t")
for prot in ["ITGA4", "IFNAR1", "ICAM1"]:   # ICAM1 = recovered-positive control
    m = genes == prot
    inf = pd.DataFrame({"context": ctxs[m], "label": y[m], "infold_prob": np.round(infold[m], 4)})
    merged = inf.merge(oof[oof.protein == prot][["context", "oof_prob"]], on="context")
    print(f"=== {prot} ===")
    print(merged.to_string(index=False))
    print(f"  OOF {merged.oof_prob.min():.4f}-{merged.oof_prob.max():.4f}  |  "
          f"IN-FOLD {merged.infold_prob.min():.4f}-{merged.infold_prob.max():.4f}\n")
