"""TEMP one-off probe (crohn_alzheimer_ild_uc_embedding_expressed): per UC/Crohn(IBD) macrophage
disease-STATE network, rank proteins by embedding shift away from the healthy macrophage reference
(healthy_pinnacle_macrophage). One table per state, ordered by ||Z_state[p] - Z_healthy[p]|| desc.
Outputs: results/_tmp_shared_disease/shift_from_healthy__<tag>.tsv
Run: .venv/bin/python mlp_mods/de_ppi/results/_tmp_shared_disease/shift_from_healthy.py
"""
import numpy as np, pandas as pd
from pathlib import Path
B = Path("mlp_mods/de_ppi/results/crohn_alzheimer_ild_uc_embedding_expressed")
OUT = Path("mlp_mods/de_ppi/results/_tmp_shared_disease")
REF = "healthy_pinnacle_macrophage"
d = np.load(B/"embeddings.npz", allow_pickle=True)
tags, Z, present = list(d["tags"]), d["Z"], d["present"]
ids = np.array(d["node_id"], dtype=object); isp = d["node_type"] == "protein"
ri = tags.index(REF); H = Z[ri]; presH = present[ri] & isp
# UC + Crohn (IBD) macrophage disease-state networks
states = [t for t in tags if any(k in t for k in ("crohn_mac_", "crohn_colon_mac_", "uc_macrophage_"))
          and "healthy" not in t]
print(f"healthy ref = {REF} | {len(states)} UC/IBD macrophage states\n")
for t in sorted(states):
    ti = tags.index(t); both = present[ti] & presH
    shift = np.linalg.norm(Z[ti, both] - H[both], axis=1)
    df = pd.DataFrame({"gene": ids[both], "shift_from_healthy": np.round(shift, 4)}) \
           .sort_values("shift_from_healthy", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.to_csv(OUT/f"shift_from_healthy__{t}.tsv", sep="\t", index=False)
    print(f"{t:32s} n={int(both.sum()):4d}  top: {', '.join(df.gene.head(8))}")
print(f"\nwrote {len(states)} tables to {OUT}")
