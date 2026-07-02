"""TEMP probe: ILD as OUTGROUP origin. Per protein, for every cross-disease UC<->Crohn macrophage
context pair, mean shift magnitude from the pooled-ILD reference and cosine between the two
shift-from-ILD vectors. (healthy cancels: UC-ILD, Crohn-ILD.)
Output: results/_tmp_shared_disease/context_pair_cosine_ILD.tsv
"""
import sys; sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
import numpy as np, pandas as pd
from itertools import combinations
from pathlib import Path
from _layout import tag_disease, tag_tissue, tag_state
B = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed")
OUT = Path("mlp_mods/de_ppi/results/_tmp_shared_disease")
d = np.load(B/"embeddings.npz", allow_pickle=True)
tags, Z, present = list(d["tags"]), d["Z"], d["present"]
ids = np.array(d["node_id"], dtype=object); isp = d["node_type"] == "protein"
# pooled ILD macrophage reference (outgroup origin)
ild = [t for t in tags if t.startswith("ild_macrophage_")]
acc = np.zeros(Z.shape[1:]); cnt = np.zeros(Z.shape[1])
for t in ild:
    m = present[tags.index(t)] & isp; acc[m] += Z[tags.index(t), m]; cnt += m
presI = cnt > 0; ILD = np.full(Z.shape[1:], np.nan); ILD[presI] = acc[presI] / cnt[presI, None]
# UC + Crohn macrophage disease-state contexts
ctx = [t for t in tags if any(k in t for k in ("crohn_mac_","crohn_colon_mac_","uc_macrophage_")) and "healthy" not in t]
shift = {t: Z[tags.index(t)] - ILD for t in ctx}
pres  = {t: present[tags.index(t)] & isp for t in ctx}   # NOT requiring ILD: ILD-absent -> NaN mag/cos
rows = []
for a, b in combinations(sorted(ctx), 2):
    if tag_disease(a) == tag_disease(b): continue          # cross-disease (UC<->Crohn) only
    m = pres[a] & pres[b]
    sa, sb = shift[a][m], shift[b][m]
    na, nb = np.linalg.norm(sa, axis=1), np.linalg.norm(sb, axis=1)
    cos = (sa*sb).sum(1) / (na*nb + 1e-12)
    rows.append(pd.DataFrame({
        "disease_a": tag_disease(a), "tissue_a": tag_tissue(a), "state_a": tag_state(a),
        "disease_b": tag_disease(b), "tissue_b": tag_tissue(b), "state_b": tag_state(b),
        "cell_type": "macrophage", "protein": ids[m],
        "mean_magnitude_from_ILD": np.round((na+nb)/2, 4), "cosine": np.round(cos, 4)}))
df = pd.concat(rows, ignore_index=True).sort_values("mean_magnitude_from_ILD", ascending=False)
df.to_csv(OUT/"context_pair_cosine_ILD.tsv", sep="\t", index=False)
print(f"ILD ref pooled over {len(ild)} states | UC/Crohn contexts: {len(ctx)} | rows: {len(df)}")
print(df.head(12).to_string(index=False))
print(f"\nwrote {OUT/'context_pair_cosine_ILD.tsv'}")
