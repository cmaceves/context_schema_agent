"""
Boxplot: cosine similarity between diseases and their FDA-approved drugs
vs. diseases and random entities.

For each disease in the drug_disease_test output:
  1. Build a binary feature vector from its controlled-vocabulary labels
  2. Find its true-positive drugs (indication edges in graph.txt)
  3. Compute mean cosine similarity to those drugs
  4. Compute mean cosine similarity to 10 random nodes from the latest
     archive nodes file
  5. Plot both distributions as a boxplot

Usage:
    python plot_drug_disease_similarity.py
"""

import json
import glob
import re
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from color_scheme import ENTITY_TYPE_PALETTE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load latest schema ─────────────────────────────────────────────────────
archive = _PROJECT_ROOT / "output" / "archive"
schema_files = glob.glob(str(archive / "schema_final_*.json"))
latest_schema_path = max(
    schema_files,
    key=lambda f: int(re.search(r"(\d+)", f.split("schema_final_")[-1]).group(1)),
)
with open(latest_schema_path) as f:
    schema = json.load(f)

# Build ordered (field_name, terms) pairs for the binary matrix
controlled_fields = []
for field in schema["fields"]:
    if field["field_type"] == "controlled":
        vocab_key = field["controlled_vocabulary"]
        terms = schema["controlled_vocabularies"][vocab_key]
        controlled_fields.append((field["name"], terms))

num_features = sum(len(terms) for _, terms in controlled_fields)


def node_to_vector(node: dict) -> np.ndarray:
    """Convert a node dict into a binary feature vector."""
    vec = np.zeros(num_features, dtype=int)
    offset = 0
    for field_name, terms in controlled_fields:
        values = node.get(field_name)
        if values is not None:
            for j, term in enumerate(terms):
                if term in values:
                    vec[offset + j] = 1
        offset += len(terms)
    return vec


# ── Load drug-disease test nodes ───────────────────────────────────────────
dd_dir = _PROJECT_ROOT / "output" / "drug_disease_test"
dd_files = glob.glob(str(dd_dir / "nodes_*.json"))
if not dd_files:
    raise FileNotFoundError("No nodes_*.json found in output/drug_disease_test/")
latest_dd = max(
    dd_files,
    key=lambda f: int(re.search(r"(\d+)", f.split("nodes_")[-1]).group(1)),
)
with open(latest_dd) as f:
    dd_nodes = json.load(f)
print(f"Loaded {len(dd_nodes)} drug-disease test nodes from {latest_dd}")

dd_by_id = {n["id"]: n for n in dd_nodes}

# ── Load graph.txt and extract indication edges ───────────────────────────
indication_edges: list[tuple[str, str]] = []  # (drug_id, disease_id)
graph_path = _PROJECT_ROOT / "db" / "graph.txt"
with open(graph_path) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[1] == "indication":
            indication_edges.append((parts[0], parts[2]))
print(f"Loaded {len(indication_edges)} indication edges from graph.txt")

# Build disease -> [drug_ids] mapping, restricted to nodes in the test set
disease_to_drugs: dict[str, list[str]] = {}
for drug_id, disease_id in indication_edges:
    if disease_id in dd_by_id and drug_id in dd_by_id:
        disease_to_drugs.setdefault(disease_id, []).append(drug_id)

# Keep only diseases that have at least 1 matched drug
disease_to_drugs = {d: drugs for d, drugs in disease_to_drugs.items() if drugs}
print(f"Diseases with matched drugs in test set: {len(disease_to_drugs)}")

# ── Load 10 random nodes from latest archive nodes file ────────────────────
archive_nodes_files = glob.glob(str(archive / "nodes_*.json"))
if not archive_nodes_files:
    raise FileNotFoundError("No nodes_*.json found in output/archive/")
latest_archive_nodes = max(
    archive_nodes_files,
    key=lambda f: int(re.search(r"(\d+)", f.split("nodes_")[-1]).group(1)),
)
with open(latest_archive_nodes) as f:
    archive_nodes = json.load(f)
print(f"Loaded {len(archive_nodes)} archive nodes from {latest_archive_nodes}")

random_sample = random.sample(archive_nodes, min(10, len(archive_nodes)))
random_vectors = np.array([node_to_vector(n) for n in random_sample])

# ── Compute similarities ──────────────────────────────────────────────────
true_positive_sims = []
random_sims = []

for disease_id, drug_ids in disease_to_drugs.items():
    disease_vec = node_to_vector(dd_by_id[disease_id]).reshape(1, -1)

    # True positives: FDA-approved drugs for this disease
    drug_vecs = np.array([node_to_vector(dd_by_id[d]) for d in drug_ids])
    tp_sim = cosine_similarity(disease_vec, drug_vecs).flatten()
    tp_mean = tp_sim.mean()
    true_positive_sims.append(tp_mean)

    # Random nodes
    rand_sim = cosine_similarity(disease_vec, random_vectors).flatten()
    rand_mean = rand_sim.mean()
    random_sims.append(rand_mean)

print(f"Computed similarities for {len(true_positive_sims)} diseases")
print(f"  True positives — mean: {np.mean(true_positive_sims):.3f}, "
      f"median: {np.median(true_positive_sims):.3f}")
print(f"  Random         — mean: {np.mean(random_sims):.3f}, "
      f"median: {np.median(random_sims):.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 6))

plot_data = {
    "Cosine Similarity": true_positive_sims + random_sims,
    "Group": (
        ["FDA-Approved Drugs"] * len(true_positive_sims)
        + ["Random Entities"] * len(random_sims)
    ),
}

sns.boxplot(
    x="Group",
    y="Cosine Similarity",
    data=plot_data,
    palette=[ENTITY_TYPE_PALETTE["ChemicalSubstance"], ENTITY_TYPE_PALETTE["Disease"]],
    width=0.5,
    ax=ax,
)

ax.set_title("Disease–Entity Cosine Similarity\n(Schema Feature Vectors)", fontsize=13)
ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
ax.set_xlabel("")

plt.tight_layout()
out_path = _PROJECT_ROOT / "images" / "drug_disease_similarity.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.close()
