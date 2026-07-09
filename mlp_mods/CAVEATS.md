# CAVEATS.md — how to read the state-shift / reach results

Load-bearing limitations of the current pipeline, with the evidence that
established each. Read these before trusting any rank-shift, influence, or
reach-based target claim. Ordered most → least fundamental (input signal first,
then graph, then interpretation).

---

## 1. Sequencing-depth / dataset batch confound — ADDRESSED in the current DE code (INPUT SIGNAL)

**Status: fixed.** The dysregulation signal is now produced by the per-build DE
scripts in `rank_shifts/de_scripts/<build>.py` (→ `pseudobulk_de.tsv`, the
`de_table` consumed by `de_ppi`), which were rebuilt specifically to remove this
confound. The original cross-dataset marker-Wilcoxon rank-shift that this section
documented is retired. What the current `macrophage_crohn.py` does:

1. **Same-dataset design.** Both the normal and Crohn macrophage arms are pulled
   from the *single* IBD atlas (`dataset_id a37f857c`) that contains both states,
   so depth / protocol / batch / site are matched by construction — removing the
   three stacked confounds (different dataset, ≈4× depth gap, colon-vs-ileum) that
   the old design carried.
2. **Per-donor pseudobulk.** Raw counts are summed per donor, which removes the
   per-cell dropout that drove the housekeeping-gene artifact below.
3. **DESeq2 negative-binomial DE** (`pydeseq2`, donors as replicates). DESeq2's
   median-of-ratios size factors normalize library-size differences across the
   pseudobulk samples — the standard depth-robust contrast.
4. **Housekeeping sanity check** (ACTB / PPIA / FKBP1A asserted ~flat, padj≈1) is
   printed every run, so a re-emergence of the artifact is caught.

**Original problem (historical, motivated the rebuild).** The old arms came from
*different datasets* at very different depth, and the Wilcoxon marker rank is
depth-sensitive, so `rank_shift` partly measured library depth, not biology:

| | healthy arm | Crohn arm | ratio |
|---|---|---|---|
| file | `healthy_macrophage_large_intestine.h5ad` (one source) | `macrophage_ibd.h5ad` (different `dataset_id`) | — |
| tissue | colorectum / rectum / appendix | **ileum / ileum lamina propria / sigmoid** | different site |
| n cells | 1,549 | 8,199 | — |
| median counts/cell | **2,890** | **711** | **0.25 (≈4× shallower)** |
| median genes/cell | 1,083 | 373 | 0.34 |

Detection collapsed for *every* gene in the shallower Crohn arm, including
housekeeping genes that cannot be biologically "dysregulated" (CD14 62.8%→31.4%,
ACTB 98.2%→89.0%, PPIA 81.5%→47.4%, FKBP1A 61.3%→31.0%). CD14 then looked like a
big mover (rank 146→3,825) while actually being *relatively preserved* once
depth-corrected — consistent with CD14⁺ inflammatory macrophages *expanding* in
Crohn's. Same-dataset pseudobulk + DESeq2 size factors remove this class of
artifact.

**Residual caveat (separate axis, NOT depth — see §4).** The `de_ppi` sender-weight
gate still derives from a within-arm **CPM expression rank-shift**
(`rank_shift = crohn_rank − healthy_rank`), not the DESeq2 statistic. It is now
computed on depth-normalized, dropout-free pseudobulk so it is far more robust
than the old per-cell ranks — but it remains an *expression* rank, so the
expression≠activity caveat still applies to the weights.

---

## 2. The reach metric is blind to non-protein-protein mechanisms (GRAPH / EDGE COVERAGE)

`influence_on_dysregulated` propagates only along OmniPath's directed
protein→protein arcs (post-translational signaling + transcriptional). A target
whose mechanism is **not** a protein→protein signaling/transcription edge is
structurally a sink (out-degree ≈ 0) and scores ~0 — regardless of disease
relevance. This is a **coverage false-negative**, not evidence the target is
irrelevant.

**Worked contrast (both bottom-ranked IBD targets, opposite reasons):**

| target | what it is | why low influence rank | low rank correct? |
|---|---|---|---|
| **CUL4A** (#2189) | scaffold of the CRL4(CRBN) E3 ligase; drugged only as a cereblon **molecular-glue chassis** (IMiDs); every ChEMBL "CUL4A" entry is a CRBN-CUL4 complex/PPI; phase 3 in IBD, never phase 4 | genuinely not an upstream control node | **Yes** — label is a glue-MoA annotation artifact |
| **PIKFYVE** (#1077) | bona-fide druggable lipid kinase (apilimod, phase 2 Crohn's); substrate is a **lipid** (PI3P→PI(3,5)P₂); controls endolysosomal trafficking / MHC-II antigen presentation; out-degree 3, in-degree 8 | its lipid/membrane mechanism **has no protein→protein arc** to broadcast on | **No** — false negative from edge coverage |

**Fix.** Don't change the scoring metric; extend edge coverage (lipid/membrane,
vesicular-trafficking, or pathway-level relations) if such targets must be
recoverable.

---

## 3. Influence ranking is, mechanically, an out-degree ranking (INTERPRETATION)

Directionality lifts upstream regulators above the dysregulated module
(undirected PINNACLE: 35/50 of the top influencers were themselves dysregulated;
directed OmniPath: 1/50). But the deciding quantity is **out-degree ≫ in-degree**,
which is exactly the transcription-factor signature (MYC out-deg 629, NFKB1 294,
MAPK14 234, NR3C1 144; sinks like CUL4A out-deg 1, PIKFYVE 3).

**Consequence.** Top-of-list ordering ≈ "has many outgoing edges." Any claim that
disease-relevant control is enriched **must** be tested against an
**out-degree-matched (or label-permuted) null**, not against a uniform random
baseline. Reported IBD-target ranks (best MAPK14 #16 / top 0.6%; median ≈ #219 /
top 8% over the 10 targets present) are not meaningful until that null is run.

---

## 4. "Labeled target" ≠ "should rank high on reach"

OpenTargets positives = targets of drugs tested in IBD by clinical phase
(`knownDrugsAggregated`). That label conflates (a) genuine upstream disease
drivers, (b) molecular-glue chassis (CUL4A), and (c) mechanism-of-action targets
invisible to the graph (PIKFYVE). Reach measures *topological control over the
dysregulated set* — a different axis from druggability or MoA. Only ~10 of 32
labeled macrophage targets survive into the OmniPath node set, so target-set
enrichment is computed over a small, non-random surviving subset.

---

## Standing expression-vs-activity caveat (from PLAN2.md, still applies)

Expression rank ≠ activity: good targets (kinases/receptors) are often
expression-stable, so the encoder cannot see their disease relevance from rank
alone. Reach scores measure controllability (unsigned magnitude), not
druggability or correction direction — combine with ESM/druggability +
essentiality for target candidates.
