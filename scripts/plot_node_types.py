"""
Stacked barplot: node type distribution per iteration.

Reads nodes_N.json files from output/archive/ and cross-references
node IDs with db/nodes.csv to get the original entity type (label).
X-axis = iteration (N), Y-axis = count, colored by label.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from color_scheme import ENTITY_TYPE_PALETTE, ENTITY_TYPE_ORDER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_ARCHIVE_DIR = _ROOT / "output" / "archive"
_NODES_CSV = _ROOT / "db" / "nodes.csv"
_IMAGES_DIR = _ROOT / "images"
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load label lookup from nodes.csv
# ---------------------------------------------------------------------------


def _load_label_lookup() -> dict[str, str]:
    """Return {node_id: label} from db/nodes.csv."""
    lookup: dict[str, str] = {}
    with open(_NODES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["id"]] = row["label"]
    return lookup


# ---------------------------------------------------------------------------
# Scan archive for nodes_N.json files
# ---------------------------------------------------------------------------


def _load_iterations() -> list[tuple[int, list[dict]]]:
    """Return sorted list of (run_number, nodes_list) from output/archive/."""
    iterations: list[tuple[int, list[dict]]] = []
    for p in _ARCHIVE_DIR.glob("nodes_*.json"):
        stem = p.stem  # e.g. nodes_2
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            run_num = int(parts[1])
            nodes = json.loads(p.read_text(encoding="utf-8"))
            iterations.append((run_num, nodes))
    iterations.sort(key=lambda x: x[0])
    return iterations


# ---------------------------------------------------------------------------
# Build counts
# ---------------------------------------------------------------------------


def _count_by_type(
    iterations: list[tuple[int, list[dict]]],
    label_lookup: dict[str, str],
) -> tuple[list[int], dict[str, list[int]]]:
    """Return (run_numbers, {entity_type: [count_per_run]})."""
    run_numbers = [r for r, _ in iterations]
    counts: dict[str, list[int]] = {et: [] for et in ENTITY_TYPE_ORDER}

    for _, nodes in iterations:
        type_counts: dict[str, int] = {et: 0 for et in ENTITY_TYPE_ORDER}
        for node in nodes:
            # Prefer label from the node itself, fall back to CSV lookup
            label = node.get("label") or label_lookup.get(node.get("id", ""), "")
            if label in type_counts:
                type_counts[label] += 1
        for et in ENTITY_TYPE_ORDER:
            counts[et].append(type_counts[et])

    return run_numbers, counts


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def main():
    label_lookup = _load_label_lookup()
    iterations = _load_iterations()

    if not iterations:
        print("No nodes_N.json files found in output/archive/.")
        return

    run_numbers, counts = _count_by_type(iterations, label_lookup)
    x = np.arange(len(run_numbers))

    fig, ax = plt.subplots(figsize=(max(6, len(run_numbers) * 1.2), 6))

    bottom = np.zeros(len(run_numbers))
    for et in ENTITY_TYPE_ORDER:
        values = np.array(counts[et])
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=et,
            color=ENTITY_TYPE_PALETTE[et],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.set_title("Node Type Distribution per Iteration")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    out_path = _IMAGES_DIR / "node_types_by_iteration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
