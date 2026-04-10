"""
Visual schematic of the drug repurposing training pipeline.

Three panels showing:
  1. Where the data vectors come from (schema → binary vectors)
  2. How they get formed into training vectors (drug - disease = diff)
  3. What is being trained and evaluated (logistic regression + drug ranking)

Saves to images/training_schematic.png

Usage:
    python plot_training_schematic.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from color_scheme import ENTITY_TYPE_PALETTE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

C_DRUG = ENTITY_TYPE_PALETTE["ChemicalSubstance"]
C_DISEASE = ENTITY_TYPE_PALETTE["Disease"]
C_POS = "#55A868"
C_NEG = "#C44E52"
C_BG = "#FAFAFA"
C_GRID = "#E0E0E0"
C_TEXT = "#222222"
C_BORDER = "#444444"


def draw_panel_border(ax, label):
    """Draw a visible border around a panel with a step label."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(C_BORDER)
        spine.set_linewidth(1.5)


fig = plt.figure(figsize=(18, 14), facecolor=C_BG)
fig.suptitle("Drug Repurposing Model — Training Pipeline",
             fontsize=24, fontweight="bold", color=C_TEXT, y=0.98)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 1: Where vectors come from (top, full width)
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.04, 0.66, 0.92, 0.28])
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 6)
ax1.set_facecolor("#F5F5F5")
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
draw_panel_border(ax1, "1")

ax1.text(0.3, 5.4, "Step 1: Schema → Binary Feature Vectors", fontsize=18,
         fontweight="bold", color=C_TEXT)

# Schema fields table
fields = ["organism", "tissue_location", "cell_type", "biological_system",
          "mechanism_of_action", "drug_class", "disease_assoc"]
terms_example = [
    ["human", "mouse", "rat"],
    ["blood", "liver", "brain", "lung"],
    ["neuron", "epithelial", "immune"],
    ["nervous", "cardiovasc", "immune"],
    ["enzyme_inh", "receptor_ag", "..."],
    ["analgesic", "antibiotic", "..."],
    ["cancer", "cardiac", "metabolic"],
]

ax1.text(0.5, 4.7, "Schema Fields", fontsize=14, fontweight="bold", color=C_TEXT)
ax1.text(3.0, 4.7, "Vocabulary Terms", fontsize=14, fontweight="bold", color=C_TEXT)
for i, (f, t) in enumerate(zip(fields, terms_example)):
    y = 4.2 - i * 0.55
    ax1.add_patch(mpatches.FancyBboxPatch((0.3, y - 0.2), 2.4, 0.4,
                  boxstyle="round,pad=0.05", facecolor="#E8EDF2",
                  edgecolor="#8899AA", linewidth=0.8))
    ax1.text(1.5, y, f, fontsize=11, ha="center", va="center", color=C_TEXT,
             family="monospace")
    terms_str = ", ".join(t)
    ax1.text(3.0, y, terms_str, fontsize=11, ha="left", va="center", color=C_TEXT,
             family="monospace")

# Arrow
ax1.annotate("", xy=(9.0, 3.0), xytext=(7.5, 3.0),
             arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=2))
ax1.text(8.25, 3.4, "encode", fontsize=14, ha="center", color=C_TEXT,
         fontweight="bold", style="italic")

# Binary vectors
drug_bits = [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
disease_bits = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]

vec_x = 9.5
cell_w, cell_h = 0.38, 0.55

# Drug vector
ax1.text(vec_x + len(drug_bits) * cell_w / 2, 5.0, "Aspirin (drug)",
         fontsize=14, ha="center", fontweight="bold", color=C_DRUG)
for j, b in enumerate(drug_bits):
    color = C_DRUG if b else "#E8E8E8"
    rect = mpatches.Rectangle((vec_x + j * cell_w, 4.2), cell_w, cell_h,
                               facecolor=color, edgecolor="white", linewidth=0.8)
    ax1.add_patch(rect)
    ax1.text(vec_x + j * cell_w + cell_w / 2, 4.45, str(b),
             fontsize=9, ha="center", va="center",
             color="white" if b else "#999999", fontweight="bold")

# Disease vector
ax1.text(vec_x + len(disease_bits) * cell_w / 2, 3.6, "Heart Failure (disease)",
         fontsize=14, ha="center", fontweight="bold", color=C_DISEASE)
for j, b in enumerate(disease_bits):
    color = C_DISEASE if b else "#E8E8E8"
    rect = mpatches.Rectangle((vec_x + j * cell_w, 2.8), cell_w, cell_h,
                               facecolor=color, edgecolor="white", linewidth=0.8)
    ax1.add_patch(rect)
    ax1.text(vec_x + j * cell_w + cell_w / 2, 3.05, str(b),
             fontsize=9, ha="center", va="center",
             color="white" if b else "#999999", fontweight="bold")

# Dimension labels
dim_labels = ["org", "", "", "tis", "", "", "", "cel", "", "", "bio", "", "",
              "moa", "", "", "drg", "", "", "dis", "", ""]
for j, lab in enumerate(dim_labels):
    if lab:
        ax1.text(vec_x + j * cell_w + cell_w / 2, 2.45, lab, fontsize=9,
                 ha="center", va="top", color=C_TEXT, rotation=45,
                 family="monospace")

ax1.text(vec_x + len(drug_bits) * cell_w / 2, 1.6,
         "Each node becomes a binary vector: 1 = vocabulary term present, 0 = absent",
         fontsize=13, ha="center", va="top", color=C_TEXT, style="italic")

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 2: How training vectors are formed (middle left)
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.04, 0.34, 0.45, 0.28])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.set_facecolor("#F5F5F5")
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
draw_panel_border(ax2, "2")

ax2.text(0.3, 5.4, "Step 2: Pair Construction → Difference Vectors",
         fontsize=18, fontweight="bold", color=C_TEXT)

# Positive pair
vx = 0.5
cy = 4.5
ax2.text(vx, cy + 0.3, "Positive pair (label = 1)", fontsize=14,
         fontweight="bold", color=C_POS)
ax2.text(vx, cy - 0.1, "Known indication edge from graph.txt", fontsize=12, color=C_TEXT)

# Drug vec
cw, ch = 0.36, 0.4
for j, b in enumerate(drug_bits[:12]):
    color = C_DRUG if b else "#E8E8E8"
    rect = mpatches.Rectangle((vx + j * cw, cy - 0.75), cw, ch,
                               facecolor=color, edgecolor="white", linewidth=0.5)
    ax2.add_patch(rect)
ax2.text(vx + 12 * cw + 0.2, cy - 0.55, "...", fontsize=14, color=C_TEXT)
ax2.text(vx - 0.1, cy - 0.55, "drug", fontsize=13, ha="right", color=C_DRUG,
         fontweight="bold")

# Minus
ax2.text(vx + 5 * cw, cy - 1.15, "−", fontsize=22, ha="center", color=C_TEXT,
         fontweight="bold")

# Disease vec
for j, b in enumerate(disease_bits[:12]):
    color = C_DISEASE if b else "#E8E8E8"
    rect = mpatches.Rectangle((vx + j * cw, cy - 1.65), cw, ch,
                               facecolor=color, edgecolor="white", linewidth=0.5)
    ax2.add_patch(rect)
ax2.text(vx + 12 * cw + 0.2, cy - 1.45, "...", fontsize=14, color=C_TEXT)
ax2.text(vx - 0.1, cy - 1.45, "dis", fontsize=13, ha="right", color=C_DISEASE,
         fontweight="bold")

# Equals
ax2.text(vx + 5 * cw, cy - 2.05, "=", fontsize=20, ha="center", color=C_TEXT,
         fontweight="bold")

# Diff vector
diff_bits = [d - s for d, s in zip(drug_bits[:12], disease_bits[:12])]
for j, b in enumerate(diff_bits):
    if b > 0:
        color = C_POS
    elif b < 0:
        color = C_NEG
    else:
        color = "#E8E8E8"
    rect = mpatches.Rectangle((vx + j * cw, cy - 2.55), cw, ch,
                               facecolor=color, edgecolor="white", linewidth=0.5)
    ax2.add_patch(rect)
ax2.text(vx + 12 * cw + 0.2, cy - 2.35, "...", fontsize=14, color=C_TEXT)
ax2.text(vx - 0.1, cy - 2.35, "diff", fontsize=13, ha="right", color=C_TEXT,
         fontweight="bold")

# Legend
lx = 5.8
ax2.add_patch(mpatches.Rectangle((lx, cy - 0.7), 0.3, 0.3, facecolor=C_POS, edgecolor="none"))
ax2.text(lx + 0.45, cy - 0.55, "+1  drug has, disease doesn't",
         fontsize=12, va="center", color=C_TEXT)
ax2.add_patch(mpatches.Rectangle((lx, cy - 1.15), 0.3, 0.3, facecolor=C_NEG, edgecolor="none"))
ax2.text(lx + 0.45, cy - 1.0, "−1  disease has, drug doesn't",
         fontsize=12, va="center", color=C_TEXT)
ax2.add_patch(mpatches.Rectangle((lx, cy - 1.6), 0.3, 0.3, facecolor="#E8E8E8", edgecolor="#BBBBBB"))
ax2.text(lx + 0.45, cy - 1.45, " 0   both same",
         fontsize=12, va="center", color=C_TEXT)

# Negative pair callout
ax2.add_patch(mpatches.FancyBboxPatch((5.5, 0.3), 4.0, 1.2,
              boxstyle="round,pad=0.15", facecolor="#FFF3E0", edgecolor="#FFAB40", linewidth=1.2))
ax2.text(7.5, 1.15, "Negative pairs (label = 0)", fontsize=13, ha="center",
         fontweight="bold", color=C_NEG)
ax2.text(7.5, 0.7, "Random drug + random disease\nwith no known edge",
         fontsize=12, ha="center", color=C_TEXT)

# Dataset note
ax2.text(2.5, 0.5, "50% positive + 50% negative\n80/20 train-test split",
         fontsize=13, ha="center", color=C_TEXT, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 3: What is being trained + evaluation (middle right)
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.53, 0.34, 0.43, 0.28])
ax3.set_xlim(-4, 4)
ax3.set_ylim(-3.5, 3.5)
ax3.set_facecolor("#F5F5F5")
ax3.set_title("Step 3: Logistic Regression — Decision Boundary", fontsize=18,
              fontweight="bold", color=C_TEXT, pad=12)
ax3.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
ax3.set_xlabel("Difference Vector Dim 1 (illustrative)", fontsize=12, color=C_TEXT)
ax3.set_ylabel("Difference Vector Dim 2 (illustrative)", fontsize=12, color=C_TEXT)
ax3.tick_params(labelsize=11, colors=C_TEXT)
for spine in ax3.spines.values():
    spine.set_edgecolor(C_BORDER)
    spine.set_linewidth(1.5)

# Scatter
np.random.seed(42)
n_pts = 40
pos_x = np.random.normal(1.2, 0.8, n_pts)
pos_y = np.random.normal(1.0, 0.9, n_pts)
neg_x = np.random.normal(-1.0, 0.8, n_pts)
neg_y = np.random.normal(-0.8, 0.9, n_pts)

ax3.scatter(pos_x, pos_y, c=C_POS, s=50, alpha=0.8, edgecolors="white",
            linewidth=0.5, label="Real indication (y=1)", zorder=3)
ax3.scatter(neg_x, neg_y, c=C_NEG, s=50, alpha=0.8, edgecolors="white",
            linewidth=0.5, label="Random pair (y=0)", zorder=3)

# Decision boundary
bx = np.linspace(-4, 4, 100)
by = -bx * 0.8 + 0.2
ax3.plot(bx, by, color=C_TEXT, lw=2, linestyle="--", alpha=0.7, zorder=2)
ax3.text(2.8, -2.8, "Decision\nboundary", fontsize=13, ha="center", color=C_TEXT,
         fontweight="bold", style="italic")

# Shade regions
ax3.fill_between(bx, by, 5, alpha=0.06, color=C_POS)
ax3.fill_between(bx, by, -5, alpha=0.06, color=C_NEG)
ax3.text(-2.5, 2.5, "P(indication) > 0.5", fontsize=13, color=C_POS,
         fontweight="bold")
ax3.text(2.5, -1.6, "P(indication) < 0.5", fontsize=13, color=C_NEG,
         fontweight="bold")

ax3.legend(loc="upper left", fontsize=12, framealpha=0.9)

# Equation
ax3.text(0, -3.1,
         r"$P(\mathrm{indication}) = \sigma(\mathbf{w} \cdot (\vec{drug} - \vec{disease}) + b)$",
         fontsize=14, ha="center", color=C_TEXT,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_BORDER))

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 4: Output — Drug ranking table (bottom, full width)
# ═══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_axes([0.04, 0.03, 0.92, 0.27])
ax4.set_xlim(0, 20)
ax4.set_ylim(0, 5)
ax4.set_facecolor("#F5F5F5")
ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
draw_panel_border(ax4, "4")

ax4.text(0.3, 4.5, "Output: Drug Ranking for a Target Disease", fontsize=18,
         fontweight="bold", color=C_TEXT)
ax4.text(0.3, 4.0,
         "For each drug, compute  diff = drug_vec - disease_vec,  "
         "then predict  P(indication).  Rank by probability.",
         fontsize=14, color=C_TEXT)

# Table headers
hx = [2.0, 5.5, 9.5, 13.0, 16.5]
headers = ["Rank", "Drug", "P(indication)", "Known FDA?", "Interpretation"]
for x, h in zip(hx, headers):
    ax4.text(x, 3.3, h, fontsize=14, fontweight="bold", color=C_TEXT, ha="center")

ax4.plot([0.5, 19.5], [3.05, 3.05], color=C_BORDER, lw=1)

rankings = [
    ("1", "Drug A", "0.94", "Yes", "Correctly identified known treatment"),
    ("2", "Drug B", "0.87", "Yes", "Correctly identified known treatment"),
    ("3", "Drug C", "0.82", "No", "Potential repurposing candidate"),
    ("4", "Drug D", "0.71", "No", "Potential repurposing candidate"),
    ("5", "Drug E", "0.63", "Yes", "Correctly identified known treatment"),
]
for i, (rank, name, prob, known, interp) in enumerate(rankings):
    y = 2.6 - i * 0.5
    bg = "#E8F5E9" if known == "Yes" else "#FFFFFF"
    ax4.add_patch(mpatches.Rectangle((0.5, y - 0.2), 19.0, 0.45,
                  facecolor=bg, edgecolor=C_GRID, linewidth=0.5))
    vals = [rank, name, prob, known, interp]
    colors = [C_TEXT, C_TEXT, C_TEXT, C_POS if known == "Yes" else C_NEG, C_TEXT]
    weights = ["normal", "normal", "bold", "bold", "normal"]
    for x, v, c, w in zip(hx, vals, colors, weights):
        ax4.text(x, y, v, fontsize=13, ha="center", va="center", color=c, fontweight=w)

# Evaluation note
ax4.text(10, 0.15,
         "Evaluated via AUC-ROC on held-out 20% test set  |  "
         "High-ranked unknowns = repurposing candidates",
         fontsize=14, ha="center", color=C_TEXT, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_BORDER))


plt.savefig(_PROJECT_ROOT / "images" / "training_schematic.png",
            dpi=180, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {_PROJECT_ROOT / 'images' / 'training_schematic.png'}")
plt.close()
