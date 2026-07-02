"""TEMP probe (crohn_alzheimer_ild_uc_embedding_expressed): per protein, for EVERY pair of macrophage
disease-state contexts (UC, Crohn ileum, Crohn colon, ILD), the mean shift magnitude from the healthy
reference and the cosine between the two shift-from-healthy vectors.
Row = (context_a, context_b, protein); origin = healthy_pinnacle_macrophage.
Output: results/_tmp_shared_disease/context_pair_cosine.tsv
Run: .venv/bin/python mlp_mods/de_ppi/results/_tmp_shared_disease/context_pair_cosine.py
"""
import sys; sys.path.insert(0, "mlp_mods/de_ppi/scripts/analysis")
import numpy as np, pandas as pd
from itertools import combinations
from pathlib import Path
from _layout import tag_disease, tag_tissue, tag_state
B = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed")
OUT = Path("mlp_mods/de_ppi/results/_tmp_shared_disease")
REF = "healthy_pinnacle_macrophage"
d = np.load(B/"embeddings.npz", allow_pickle=True)
tags, Z, present = list(d["tags"]), d["Z"], d["present"]
ids = np.array(d["node_id"], dtype=object); isp = d["node_type"] == "protein"
ri = tags.index(REF); H = Z[ri]; presH = present[ri] & isp
ctx = [t for t in tags if any(k in t for k in ("crohn_mac_","crohn_colon_mac_","uc_macrophage_","ild_macrophage_"))
       and "healthy" not in t]
shift = {t: Z[tags.index(t)] - H for t in ctx}                 # per-context shift-from-healthy
pres  = {t: present[tags.index(t)] & presH for t in ctx}
rows = []
for a, b in combinations(sorted(ctx), 2):
    if tag_disease(a) == tag_disease(b): continue   # cross-disease pairs only
    m = pres[a] & pres[b]
    sa, sb = shift[a][m], shift[b][m]
    na, nb = np.linalg.norm(sa, axis=1), np.linalg.norm(sb, axis=1)
    cos = (sa*sb).sum(1) / (na*nb + 1e-12)
    sub = pd.DataFrame({
        "disease_a": tag_disease(a), "tissue_a": tag_tissue(a), "state_a": tag_state(a),
        "disease_b": tag_disease(b), "tissue_b": tag_tissue(b), "state_b": tag_state(b),
        "cell_type": "macrophage", "protein": ids[m],
        "mean_magnitude_from_healthy": np.round((na+nb)/2, 4), "cosine": np.round(cos, 4)})
    rows.append(sub)
df = pd.concat(rows, ignore_index=True).sort_values("mean_magnitude_from_healthy", ascending=False)
df.to_csv(OUT/"context_pair_cosine.tsv", sep="\t", index=False)
print(f"contexts: {len(ctx)} | pairs: {len(list(combinations(ctx,2)))} | rows: {len(df)}")
print(df.head(10).to_string(index=False))
print(f"\nwrote {OUT/'context_pair_cosine.tsv'}")
