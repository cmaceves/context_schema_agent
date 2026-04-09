"""
Ad-hoc script: PCA of node context vectors.

Builds a binary matrix from output/nodes.json where each row is a node and
each column is a (field, vocab_term) pair.  Runs PCA and plots PC1 vs PC2,
colored by entity type.

Usage:
    python plot_pca.py
"""

import json
import glob
import re
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from color_scheme import ENTITY_TYPE_PALETTE, ENTITY_TYPE_ORDER

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load latest schema ──────────────────────────────────────────────────────
archive = _PROJECT_ROOT / "output" / "archive"
schema_files = glob.glob(str(archive / "schema_final_*.json"))
latest_schema = max(schema_files, key=lambda f: int(re.search(r"(\d+)", f.split("schema_final_")[-1]).group(1)))
with open(latest_schema) as f:
    schema = json.load(f)

# ── Build ordered list of (field_name, vocab_terms) ─────────────────────────
controlled_fields = []
for field in schema["fields"]:
    if field["field_type"] == "controlled":
        vocab_key = field["controlled_vocabulary"]
        terms = schema["controlled_vocabularies"][vocab_key]
        controlled_fields.append((field["name"], terms))

# Build column names: "field_name::term"
columns = []
for field_name, terms in controlled_fields:
    for term in terms:
        columns.append(f"{field_name}::{term}")

# ── Load latest nodes file ──────────────────────────────────────────────────
nodes_files = glob.glob(str(archive / "nodes_*.json"))
latest_nodes = max(nodes_files, key=lambda f: int(re.search(r"(\d+)", f.split("nodes_")[-1]).group(1)))
with open(latest_nodes) as f:
    nodes = json.load(f)
print(f"Loaded nodes from {latest_nodes}")

# ── Build entity-type lookup from db/nodes.csv ──────────────────────────────
id_to_type = {}
with open(_PROJECT_ROOT / "db" / "nodes.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_to_type[row["id"]] = row["label"]

# ── Build binary matrix ─────────────────────────────────────────────────────
matrix = np.zeros((len(nodes), len(columns)), dtype=int)

for i, node in enumerate(nodes):
    col_offset = 0
    for field_name, terms in controlled_fields:
        values = node.get(field_name)
        if values is not None:
            for j, term in enumerate(terms):
                if term in values:
                    matrix[i, col_offset + j] = 1
        col_offset += len(terms)

node_ids = [n["id"] for n in nodes]
node_names = [n["name"] for n in nodes]
entity_types = [n.get("label") or id_to_type.get(n.get("id", ""), "Unknown") for n in nodes]

# ── PCA ─────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2)
coords = pca.fit_transform(matrix)

df = pd.DataFrame({
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "Entity Type": entity_types,
    "Name": node_names,
})

# ── Plot ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    data=df, x="PC1", y="PC2", hue="Entity Type",
    hue_order=ENTITY_TYPE_ORDER, palette=ENTITY_TYPE_PALETTE,
    s=60, ax=ax,
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA of Node Context Vectors (binary vocab features)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
_out_path = _PROJECT_ROOT / "images" / "pca_context.png"
plt.savefig(_out_path, dpi=150)
plt.show()
print(f"Saved to {_out_path}")
