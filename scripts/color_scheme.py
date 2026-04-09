"""
Colorblind-friendly palette for 9 entity types.

Uses a purple-to-orange diverging scheme based on the
Wong (2011) colorblind-safe palette, extended with
perceptually distinct intermediates.
"""

ENTITY_TYPE_PALETTE = {
    "AnatomicalEntity":           "#7B2D8E",  # deep purple
    "BiologicalProcessOrActivity":"#A05DB5",  # medium purple
    "ChemicalSubstance":          "#C490D1",  # light purple
    "Disease":                    "#8C6D31",  # dark gold
    "GeneFamily":                 "#B5861C",  # amber
    "MacromolecularMachine":      "#D4A017",  # golden orange
    "OrganismTaxon":              "#E8751A",  # burnt orange
    "Pathway":                    "#F5A623",  # bright orange
    "PhenotypicFeature":          "#F7C97E",  # pale orange
}

# Ordered list for consistent legend ordering
ENTITY_TYPE_ORDER = [
    "AnatomicalEntity",
    "BiologicalProcessOrActivity",
    "ChemicalSubstance",
    "Disease",
    "GeneFamily",
    "MacromolecularMachine",
    "OrganismTaxon",
    "Pathway",
    "PhenotypicFeature",
]
