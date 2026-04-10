"""
Drug repurposing prediction via logistic regression on schema feature vectors.

Approach:
  1. Load classified drug-disease nodes from output/drug_disease_test/
  2. Build binary feature vectors from controlled-vocabulary labels
  3. Extract positive pairs (real indication edges from graph.txt)
  4. Sample equal-count negative pairs (random drug-disease combos with no edge)
  5. Feature per pair: drug_vec - disease_vec (difference)
  6. Train logistic regression (80/20 split), report AUC-ROC and AUC-PR
  7. Pick a random disease and rank all drugs by predicted probability

Usage:
    python drug_repurposing_model.py
"""

import json
import glob
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
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
print(f"Schema: {latest_schema_path}")

controlled_fields = []
for field in schema["fields"]:
    if field["field_type"] == "controlled":
        vocab_key = field["controlled_vocabulary"]
        terms = schema["controlled_vocabularies"][vocab_key]
        controlled_fields.append((field["name"], terms))

num_features = sum(len(terms) for _, terms in controlled_fields)
print(f"Feature dimensions: {num_features}")


def node_to_vector(node: dict) -> np.ndarray:
    """Convert a node dict into a binary feature vector."""
    vec = np.zeros(num_features, dtype=float)
    offset = 0
    for field_name, terms in controlled_fields:
        values = node.get(field_name)
        if values is not None:
            for j, term in enumerate(terms):
                if term in values:
                    vec[offset + j] = 1.0
        offset += len(terms)
    return vec


# ── Load drug-disease test nodes ───────────────────────────────────────────
dd_dir = _PROJECT_ROOT / "output" / "drug_disease_test"
dd_files = glob.glob(str(dd_dir / "nodes_*.json"))
if not dd_files:
    raise FileNotFoundError("No nodes_*.json in output/drug_disease_test/")
latest_dd = max(
    dd_files,
    key=lambda f: int(re.search(r"(\d+)", f.split("nodes_")[-1]).group(1)),
)
with open(latest_dd) as f:
    dd_nodes = json.load(f)
print(f"Drug-disease test nodes: {len(dd_nodes)} from {latest_dd}")

dd_by_id = {n["id"]: n for n in dd_nodes}

# ── Load indication edges from graph.txt ───────────────────────────────────
indication_edges: set[tuple[str, str]] = set()  # (drug_id, disease_id)
graph_path = _PROJECT_ROOT / "db" / "graph.txt"
with open(graph_path) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[1] == "indication":
            indication_edges.add((parts[0], parts[2]))
print(f"Total indication edges in graph: {len(indication_edges)}")

# ── Separate drugs and diseases in the test set ────────────────────────────
drug_ids = set()
disease_ids = set()
for drug_id, disease_id in indication_edges:
    if drug_id in dd_by_id and disease_id in dd_by_id:
        drug_ids.add(drug_id)
        disease_ids.add(disease_id)

# Positive pairs: indication edges where both nodes are in the test set
positive_pairs = [
    (drug_id, disease_id)
    for drug_id, disease_id in indication_edges
    if drug_id in dd_by_id and disease_id in dd_by_id
]
print(f"Positive pairs (both in test set): {len(positive_pairs)}")
print(f"  Unique drugs: {len(drug_ids)}, unique diseases: {len(disease_ids)}")

# ── Sample negative pairs ──────────────────────────────────────────────────
positive_set = set(positive_pairs)
drug_list = sorted(drug_ids)
disease_list = sorted(disease_ids)

negative_pairs = []
max_attempts = len(positive_pairs) * 10
attempts = 0
while len(negative_pairs) < len(positive_pairs) and attempts < max_attempts:
    d = random.choice(drug_list)
    dis = random.choice(disease_list)
    if (d, dis) not in positive_set and (d, dis) not in negative_pairs:
        negative_pairs.append((d, dis))
    attempts += 1
print(f"Negative pairs sampled: {len(negative_pairs)}")

# ── Build feature matrix (difference vectors) ─────────────────────────────
# Precompute vectors for all nodes in the test set
vectors = {nid: node_to_vector(n) for nid, n in dd_by_id.items()}

all_pairs = positive_pairs + negative_pairs
labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))

X = np.zeros((len(all_pairs), num_features))
for i, (drug_id, disease_id) in enumerate(all_pairs):
    X[i] = vectors[drug_id] - vectors[disease_id]

print(f"Dataset: {X.shape[0]} pairs × {X.shape[1]} features")

# ── Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
    X, labels, all_pairs, test_size=0.2, random_state=42, stratify=labels,
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train logistic regression ─────────────────────────────────────────────
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr = average_precision_score(y_test, y_prob)

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")

# ── Plot ROC curve ────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color=ENTITY_TYPE_PALETTE["ChemicalSubstance"], lw=2, label=f"Logistic Regression (AUC = {auc_roc:.3f})")
ax.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--", label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Drug Repurposing — ROC Curve\n(Difference Vectors, Logistic Regression)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()

roc_path = _PROJECT_ROOT / "images" / "drug_repurposing_roc.png"
roc_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(roc_path, dpi=150)
print(f"Saved ROC curve → {roc_path}")
plt.close()

# ── Pick a random disease and rank all drugs ───────────────────────────────
target_disease_id = random.choice(disease_list)
target_disease = dd_by_id[target_disease_id]
disease_vec = vectors[target_disease_id]

# True drugs for this disease
true_drugs = {
    drug_id for drug_id, dis_id in indication_edges
    if dis_id == target_disease_id and drug_id in dd_by_id
}

print(f"\n{'='*60}")
print(f"DRUG RANKING FOR: {target_disease.get('name', '?')} ({target_disease_id})")
print(f"Known indications: {len(true_drugs)} drug(s)")
print(f"{'='*60}")

# Score every drug against this disease
drug_scores = []
for drug_id in drug_list:
    diff = vectors[drug_id] - disease_vec
    prob = clf.predict_proba(diff.reshape(1, -1))[0, 1]
    is_true = drug_id in true_drugs
    drug_scores.append((prob, drug_id, dd_by_id[drug_id].get("name", "?"), is_true))

drug_scores.sort(key=lambda x: -x[0])

print(f"\n{'Rank':<6}{'Prob':>8}  {'Drug':<40} {'Known?'}")
print("-" * 70)
for rank, (prob, did, name, is_true) in enumerate(drug_scores, 1):
    marker = " <<<" if is_true else ""
    print(f"{rank:<6}{prob:>8.4f}  {name[:40]:<40}{marker}")
    if rank >= 50:
        remaining_true = sum(1 for p, _, _, t in drug_scores[rank:] if t)
        if remaining_true:
            print(f"  ... ({remaining_true} more known drug(s) below rank 50)")
        break
