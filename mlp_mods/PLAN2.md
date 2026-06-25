# PLAN2.md — Disease state-shift influence on a PINNACLE/OmniPath PPI

## Goal

For a (cell type, disease), build a directed PPI whose **outgoing** edge weights are
set by each protein's **rank shift** between the healthy and disease states
(sender-gated), and score how much each protein **influences the dysregulated set** —
to ask which nodes can *reach* the dysregulated proteins/metabolites (control-point /
target reasoning). Influence is computed **analytically** (`P^k` propagation on the
weighted PPI; see `de_ppi/`), not via a learned embedding.

This is run as a controlled single-cell-type case (macrophage / Crohn's), but the
build stage is **parameterized by (cell type, disease)** so other builds can be
added.

> **Status (current).** The shared **R-GCN embedding encoder** that originally produced
> the embedding shift + Jacobian influence/reach has been **retired** — its code (former
> stage 4, `04_state_shift_encoder/`) and its inputs (the marker-Wilcoxon per-arm ranks
> `rank_shifts/<build>/<arm>.tsv` and `02_build_ppi/builds/<build>/node_vocab.tsv`) were
> removed. Influence/target analysis now runs **analytically** through the **`de_ppi/`
> pipeline** — a `P^k` propagation on the weighted PPI, no learned embeddings (see the
> *de_ppi influence pipeline* section below). The encoder sections here are kept for
> historical context and marked accordingly.

---

## Course we settled on (the model) — *historical (retired encoder)*

> This section describes the original **R-GCN encoder** approach, now **retired** (code +
> inputs removed). The **current** model is the analytic `de_ppi/` influence pipeline
> documented below. Kept for context.

- **Nodes (per build):** top-Wilcoxon-marker pool per arm (resample recipe:
  100 cells/cluster × 10 trials, keep markers in ≥90%) induced on the **PINNACLE
  global PPI**; union of arm pools = the node vocabulary. OmniPath is NOT used.
- **Edges:** the PINNACLE PPI is undirected → made **bidirectional** (each pair
  `{a,b}` → arcs `a→b` and `b→a`, unsigned). Topology is shared across arms; one
  shared scaffold, no isolated nodes.
- **Edge weights = rank-shift, sender-gated** (the key idea). Each message is
  gated by the **sender's** weight, so a node's own rank controls its *outgoing*
  signal (not what it receives):
  - **healthy arm:** binary (1 if rank < cutoff=4000, else 0) — reference, no gradation.
  - **disease arm:** `exp(-drop/tau)` on the uncapped rank drop `drop = rank_disease -
    rank_healthy` (tau=4000). Demotions damp toward 0; promotions amplify above 1,
    capped at `max_weight=5`. So topology is symmetric but **influence is
    rank-asymmetric**, and an absent/demoted node still *receives* but barely
    *broadcasts*.
- **Encoder:** featureless (fixed random per-protein identity + a presence
  scalar), 2-layer weighted directed R-GCN (`weighted_encoder.py`), shared weights
  across arms -> comparable embeddings, no Procrustes.
- **Objective:** unsupervised directed link prediction (asymmetric MLP decoder),
  loss summed over both arms. **Edge split is pair-grouped** (both arcs of a pair
  in the same fold) to avoid reverse-edge leakage. Held-out AUC ~= 0.88.
- **Outputs:** two embeddings per protein, the per-protein **embedding shift**,
  pairwise **influence** matrices (Jacobian, per arm), and **reach scores**
  (does perturbing a node propagate to the rank-dysregulated set, hops 1-3).

### Caveats baked into how we read results
> See **[CAVEATS.md](CAVEATS.md)** for the full, evidence-backed list (depth/batch
> confound in rank-shift, edge-coverage false negatives, out-degree null). Short
> form below.
- Expression rank != activity: good targets (kinases/receptors) are expression-
  stable, so the encoder can't see their disease relevance directly.
- Reach scores measure **controllability**, not druggability or correction
  direction (unsigned magnitude). Combine with ESM/druggability + essentiality
  to get target candidates.
- Reach is intentionally **not** hub-corrected (hubs matter); filter out
  self-dysregulated nodes to find control points that aren't themselves moving.

---

## Layout

```
mlp_mods/
├── PLAN2.md
├── 01_expression/            # single-cell expression (CellxGene pulls), reorganized as
│   ├── <disease>/<tissue_general>/<cell_type>.h5ad   # provenance-explicit slices
│   ├── pull_ibd_family.py, pull_healthy_macrophage.py, pull_fibroblast_small_intestine.py
│   ├── reorg_to_disease_tissue.py, slice_manifest.tsv
├── rank_shifts/
│   ├── de_scripts/<build>.py        # per-build one-off paired-donor DE (params kept verbatim)
│   ├── de_scripts/<build>_states.py # cell-STATE split -> per-state DE (thin wrappers, see state_split.py)
│   ├── de_scripts/state_split.py    # SHARED Leiden state-definition module (disease-blind; replaces marker-argmax)
│   ├── <build>_paired/              # pseudobulk_de.tsv (+ cached cells) -> de_ppi `de_table`
│   ├── <build>_states/              # states/<state>/pseudobulk_de.tsv (Leiden states; cell_states.tsv, state_counts.tsv)
│   └── _stale/                      # superseded outputs (old marker-argmax states, orphan clusters)
├── 03_opentargets_rebuild/   # disease labels + known drugs (OpenTargets positive/negative)
├── de_ppi/                   # influence models (analytic P^k AND learned embedding; see sections below)
│   ├── config.py, build_literature_weighted_influence.py
│   ├── precompute_expressed_genes.py  # --expressed backbone (detect>=floor) for cell types PINNACLE lacks
│   ├── embed_influence.py            # single-network GNN encoder + Jacobian influence
│   ├── joint_embed_influence.py      # SHARED-encoder joint embedding across networks (cross-disease shared space)
│   ├── influence_analysis/           # plot_phase_*, plot_embedding_clusters (PCA/t-SNE), compare_joint_vs_single
│   ├── results/<build>/              # per-build analytic network + influence
│   └── results/<embedding>/          # joint embedding: networks/<tag>/, joint_influence.tsv, embedding_shift.tsv, embeddings.npz
├── lit_search/               # stage-L LLM-proposed, literature-verified searches
├── 02_build_ppi/             # (retired-encoder support) builds_manifest.json is SHARED + still
│   │                         #   read by de_ppi/config.py + lit_search; build_disease_ppi.py +
│   │                         #   reference_h5ad/ are now orphaned (only fed the removed encoder)
│   └── _provenance/                # superseded one-off build scripts + intermediates (kept)
├── go_cache/                 # GO annotations for validation (re-fetch; see README)
└── images/

# REMOVED: 04_state_shift_encoder/ (the R-GCN encoder), 02_build_ppi/builds/ (node_vocab),
#          rank_shifts/<build>/<arm>.tsv (marker-Wilcoxon ranks) — the embedding pipeline.
```

## Pipeline

```
# 1. paired-donor DE for a (cell type, disease)  ->  rank_shifts/<build>_paired/pseudobulk_de.tsv
.venv/bin/python mlp_mods/rank_shifts/de_scripts/macrophage_crohn.py
#    cell-STATE split (Leiden, disease-blind)    ->  rank_shifts/<build>_states/states/<state>/pseudobulk_de.tsv
.venv/bin/python mlp_mods/rank_shifts/de_scripts/macrophage_crohn_states.py

# 2. per-build network + analytic influence (de_ppi)  ->  de_ppi/results/<build>/
#    --no-lit for builds without a literature panel; --expressed when PINNACLE lacks the cell type
.venv/bin/python mlp_mods/de_ppi/build_literature_weighted_influence.py    --build macrophage_crohn

# 3. learned embedding influence (single network)  ->  de_ppi/results/<build>/embedding_influence.tsv
.venv/bin/python mlp_mods/de_ppi/embed_influence.py --build macrophage_crohn

# 4. JOINT embedding across many networks (cross-disease shared space)
#    stage each network into results/<embedding>/networks/<tag>/, then:
.venv/bin/python mlp_mods/de_ppi/joint_embed_influence.py --out-name crohn_alzheimer_ild_embedding
.venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_embedding_clusters.py --out-name crohn_alzheimer_ild_embedding --method pca   # or tsne
.venv/bin/python mlp_mods/de_ppi/influence_analysis/plot_phase_joint_boxplot.py --out-name crohn_alzheimer_ild_embedding [--norm percentile|mean] [--layout tag]
```

Convention: each build's reference arm is named **`healthy`**; the other is the disease
arm. Outputs are keyed by build name, so multiple diseases/cell types coexist. (The
former stage-2 `build_disease_ppi.py` + stage-4 R-GCN encoder are retired; the learned
embedding in step 3–4 is a NEW, analytic-free encoder built on the de_ppi networks, not the
old R-GCN.)

## Open validation thread
External agreement (does it recover known IBD biology?): GO/pathway enrichment of
high-influence genes (go_cache), and correlation against an independent IBD intestinal
proteomics signature (e.g. PXD001608), with a degree-matched / label-permuted null.
Targeting = influence + ESM (druggability).

---

## Literature-evidence layer (stage L) — LLM-proposed, literature-verified searches

**Two searches**, same shape: an LLM **proposes a claim** from its own knowledge (the
ungrounded baseline — what does the model think?); we then **search Europe PMC full
text** for that claim and **verify** it against the retrieved papers with a **verbatim
guard** (a real supporting sentence, or empty — never a recalled/paraphrased
citation). An off-topic search hit fails the guard and yields no provenance. (We tried
having the LLM *cite* the PMIDs itself — both gpt-4o-mini and gpt-5 fabricated ~100%
of citations, so the LLM proposes and the literature search sources the evidence.)
They produce machine-readable evidence tables the rest of the pipeline leans on: both
the literature-weighted edge directions and the metabolite/lipid sink nodes (which fix
the edge-coverage false-negatives in **[CAVEATS.md](CAVEATS.md) §2**) feed
`de_ppi/build_literature_weighted_influence.py`.

1. **`dysregulation_genes`** — the differentially-expressed GENES in the build's
   `(cell type, disease)`, each with an `elevated`/`suppressed` direction.
2. **`dysregulation_metabolites`** — the dysregulated small molecules / lipids in the
   same `(cell type, disease)`, each with a direction and a ChEBI id.

Genes and metabolites are proposed by **separate searches** (one LLM call each, and
each prompt explicitly excludes the other kind) so the gene list is never polluted by
metabolites and the metabolite list is never polluted by proteins/cytokines/genes —
the failure mode of a single combined list.

### Generalizable across (disease, cell type)
This stage is **parameterized by a build** (same `(cell type, disease)` keying as the
rest of the pipeline; see `02_build_ppi/builds_manifest.json`). Nothing is hard-coded
to Crohn's / macrophage.

- **Slugs.** Each build defines a `disease_slug` and `celltype_slug`. For the
  `macrophage_crohn` build these are `crohns` and `macrophages`.
- **Output dir.** `mlp_mods/literature_<disease_slug>/<celltype_slug>/`
  → for this build, `literature_crohns/macrophages/`.
- **Inputs.** Neither search needs an external input list — the entities are
  LLM-proposed from the build's `(cell type, disease)`.

### Method (shared by both searches)
Each search runs in **three code phases — `--prepare` → `--fetch` → `--extract`**.
The OpenAI calls (`gpt-4o`, via `lit_search/extract.py`, reading `OPENAI_API_KEY`
from the environment or `.env`) are the propose and verify steps; `--fetch` is plain
HTTP (Europe PMC REST) — **no MCP**.

1. **Propose (`--prepare`, LLM).** One LLM call lists the dysregulated entities of the
   search's kind — genes (`dysregulation_genes`) or small molecules/lipids
   (`dysregulation_metabolites`) — in the build's `(cell_type, disease)`, each with an
   `elevated`/`suppressed` direction. This is the **ungrounded baseline**. The
   metabolite prompt explicitly **excludes proteins/cytokines/enzymes/genes**; the
   gene prompt asks for **official gene symbols only** — so the two lists can't
   cross-contaminate. One candidate per entity, each carrying **several broad query
   combinations** (disease-anchored + cell-type/expression-anchored; Europe PMC
   syntax, no PubMed `[field]` tags) — the direction/cell-type match is the verifier's
   job, not the query's.
2. **Search Europe PMC full text (`--fetch`, code — `fetch_refs.py`).** For each
   candidate, run every query combination via **Europe PMC `search`** (relevance-
   sorted, top **`PER_QUERY = 10`** each, abstract returned inline), pool them
   **round-robin** (one hit per query per pass) up to **`MAX_PMIDS = 30`** unique
   papers → `retrieved/<id>.json` (`location = abstract`). **Europe PMC indexes the
   article body**, not just title/abstract/MeSH like PubMed `esearch` — so it finds
   papers that mention an entity only in their figures/methods (empirically ~2 → ~91
   hits for `CD3E AND Crohn`). **Recall still leans on relevance ranking** — only the
   top-10 per query are pulled, so a paper ranked low is missed (raise `PER_QUERY`).
3. **Verify (`--extract`, LLM).** Per entity, the model gathers the verbatim
   supporting evidence (**1–3 sentences, possibly from different papers**) for the
   `(entity, direction)` claim and reports `inferred_cell_type` / `inferred_disease` —
   the named term or empty. It reads the first **`MAX_ARTICLES = 15`** abstracts.
4. **Full-text escalation.** For any claim still unsupported by the abstracts, the
   open-access papers among the candidate's hits (those with a PMCID) have their
   **full text pulled from Europe PMC**, sentence-filtered to the claim's terms
   (`location = full_text`, up to **`FT_MAX_DOCS = 5`** papers), and re-verified;
   the full text is appended to the cache. `--no-fulltext` disables this.
5. **Verbatim guard (hard rule).** Assert `evidence_sentence` is a substring of the
   retrieved text (whitespace-normalized), and that each `inferred_*` value it
   reports actually appears in that text. If not recoverable, write
   `evidence_sentence = ""` and `evidence_location = none` — **never** emit an
   LLM-recalled or paraphrased sentence. The proposed claim is kept either way; only
   its provenance depends on the retrieved papers.

`--extract` **streams rows to the TSV and flushes per candidate**, so a run
interrupted partway keeps the rows produced so far; `--fetch` writes each
`retrieved/<id>.json` as it goes, so retrieval progress also persists incrementally.

### Shared provenance columns (appended to every output table)
| column | meaning |
|---|---|
| `pmid` | PubMed ID (or `biorxiv:<doi>` for preprints) |
| `doi` | article DOI |
| `evidence_sentence` | verbatim sentence(s) from the retrieved text, or `""` if unrecoverable |
| `evidence_location` | `full_text` \| `abstract` \| `none` |
| `source` | `pubmed` \| `pmc` \| `biorxiv` \| `medrxiv` |
| `query` | the Europe PMC full-text query that surfaced the article |

Each search emits **one TSV** into the build's literature dir. A row with
`evidence_location = none` is kept (the claim was made) but no retrieved paper supported it.

### The two searches

**1. dysregulation_genes — DE genes** → `<disease>_<celltype>_dysregulation_genes.tsv`
- **`--prepare` (propose).** One LLM call lists the human genes differentially
  expressed in `{cell_type}`s in `{disease}`, each `elevated`/`suppressed`. The prompt
  asks for **official human gene symbols only** (no metabolites/lipids/drugs). One
  candidate per gene, with broad query combinations (`<gene> AND <disease-clause>`, a
  cell-type/expression-anchored one, `<gene> AND <cell_type>`).
- **`--fetch` + `--extract`.** `--fetch` round-robin-pools each gene's Europe PMC
  full-text hits; `--extract` gathers the verbatim evidence (**1–3 sentences, possibly
  from different papers**) supporting the `(gene, direction)` claim and reports
  `inferred_cell_type` / `inferred_disease`. Unsupported claims escalate to OA full
  text and are re-verified.

| column | meaning |
|---|---|
| `entity` | gene symbol (model-proposed) |
| `entity_type` | `gene` |
| `direction` | `elevated` \| `suppressed` (in disease vs healthy) — model-proposed |
| `cell_type` | the build cell type the search was anchored on (e.g. `macrophage`) |
| `inferred_cell_type` | cell type the **supporting evidence** names (guarded to appear in it), or empty |
| `disease` | **asserted** build disease the search was anchored on (e.g. `Crohn disease`) |
| `inferred_disease` | disease the supporting evidence is actually about, or empty |
| `model` | the LLM that proposed the entity (`gpt-4o`) |
| + shared provenance columns | **verified** against Europe PMC hits (real pmid/sentence, or `evidence_location = none`) |

Because the gene list is ungrounded, this surfaces canonical genes a local DE call
would miss (e.g. **IL6**, which fell just under our `padj` cutoff) — and every row
that lands carries a verified, real PubMed citation.

**2. dysregulation_metabolites — dysregulated small molecules / lipids** → `<disease>_<celltype>_dysregulation_metabolites.tsv`
- **`--prepare` (propose).** One LLM call lists the endogenous small molecules / lipids
  dysregulated in `{cell_type}`s in `{disease}`, each `elevated`/`suppressed`. The
  prompt **explicitly excludes proteins, cytokines, chemokines, interleukins, enzymes,
  receptors, and genes** — the contamination that arises when metabolites and genes
  share one prompt. One candidate per metabolite, same broad query combinations.
- **`--fetch` + `--extract`.** As for genes; additionally each metabolite name is
  grounded to a `chebi_id` (see below).

| column | meaning |
|---|---|
| `entity` | metabolite / lipid name (model-proposed) |
| `entity_type` | `metabolite` |
| `chebi_id` | `CHEBI:NNNN` grounded from the name via EBI OLS (empty if unresolved) |
| `direction` | `elevated` \| `suppressed` (in disease vs healthy) — model-proposed |
| `cell_type` | the build cell type the search was anchored on (e.g. `macrophage`) |
| `inferred_cell_type` | cell type the **supporting evidence** names (guarded to appear in it), or empty |
| `disease` | **asserted** build disease the search was anchored on (e.g. `Crohn disease`) |
| `inferred_disease` | disease the supporting evidence is actually about, or empty |
| `model` | the LLM that proposed the entity (`gpt-4o`) |
| + shared provenance columns | **verified** against Europe PMC hits (real pmid/sentence, or `evidence_location = none`) |

> **Metabolite → ChEBI grounding.** Each metabolite row's name is resolved to a
> ChEBI id (`resolve_ids.resolve_chebi`, via the EBI Ontology Lookup Service). It
> accepts **only an exact label/synonym match** — no fuzzy top-hit fallback, which
> would mis-ground a non-metabolite the model slips in (a cytokine maps to a wrong
> CHEBI id otherwise). To lift recall it tries a few **normalized variants** of the
> name (strip parenthetical, expand a known abbreviation e.g. `PGE2 → prostaglandin
> E2`, singularise a plural, swap hyphen/space) and takes the first variant with an
> exact hit. Best-effort and on-disk cached (`cache/chebi_name_cache.json`); needs
> network at `--extract` time; `--no-resolve` skips it. **Unresolved rows are
> kept** with an empty `chebi_id` (never dropped).

> **Asserted vs. inferred disease.** Both searches put the build's `{disease}` into
> the prompt as the **asserted** context, and separately record the disease the
> supporting sentence actually discusses as `inferred_disease`. Comparing the two
> flags off-disease evidence (a sentence that verifies the direction claim but in a
> different disease).

> **Reading the `inferred_*` columns.** The proposed `entity`/`direction`/`cell_type`/
> `disease` are what the model *claimed*; the `inferred_cell_type`/`inferred_disease`
> columns are what the *verified evidence actually names* (guarded to appear in the
> sentence; empty = the evidence didn't confirm that dimension). They flag rows whose
> support is generic (Crohn's/IBD in general) rather than cell-type-specific.

### Code (generalizable, one parameterized entry point)
```
mlp_mods/lit_search/
  search_literature.py   # CLI: --build <build> --search <S> --prepare|--fetch|--extract [--limit N]
  config.py              # resolve build from builds_manifest.json -> paths/slugs/output
  candidates.py          # turn an LLM-proposed entity into a per-entity candidate + Europe PMC queries
  fetch_refs.py          # --fetch: Europe PMC full-text search (per query) + OA full-text fetch -> retrieved/<id>.json (HTTP, no MCP)
  schema.py              # per-search columns, the verbatim guard, TSV writer
  extract.py             # OpenAI propose (--prepare) + verify (--extract) over the retrieved text; verbatim guard
  resolve_ids.py         # metabolite name -> ChEBI id via EBI OLS (exact match over normalized variants; cached)
  README.md              # the propose -> search -> verify run protocol + cache schema
```
All three phases are code (the search is plain Europe PMC REST — no MCP). Run from
the repo root:
```
PYTHONPATH=mlp_mods .venv/bin/python -m lit_search.search_literature \
    --build macrophage_crohn --search dysregulation_genes --prepare --limit 10
PYTHONPATH=mlp_mods .venv/bin/python -m lit_search.search_literature \
    --build macrophage_crohn --search dysregulation_genes --fetch --limit 10
PYTHONPATH=mlp_mods .venv/bin/python -m lit_search.search_literature \
    --build macrophage_crohn --search dysregulation_genes --extract --limit 10
# search keys: dysregulation_genes (1) | dysregulation_metabolites (2)
# NOTE: --prepare and --extract make OpenAI (gpt-4o) calls; --fetch is HTTP only.
```
Reads `cell_type`, `disease`, `disease_slug`, `celltype_slug` from
`builds_manifest.json`; writes the TSV into `literature_<disease_slug>/<celltype_slug>/`.

### How these feed the model (wired into `de_ppi/build_literature_weighted_influence.py`)
- **`dysregulation_genes`** → literature edge **directions** (elevated→2.0,
  suppressed→0.5) for genes that are *not* in the CellxGene DE set; read from
  `<disease>_<celltype>_dysregulation_genes.tsv`.
- **`dysregulation_metabolites`** → the metabolite/lipid **sink nodes** that close the
  edge-coverage gap in CAVEATS §2, unioned with the HMDB disease metabolites. Each
  carries a `chebi_id` (grounded from the name via EBI OLS, exact match) for joining to
  the graph; rows that don't resolve have an empty `chebi_id`.
- For both searches, the **model-proposed** columns (entity/direction) are the
  ungrounded baseline — what the model believes — while the verified
  `pmid`/`evidence_sentence`/`inferred_disease` are the trustworthy, paper-backed
  provenance. Prefer verified rows (`evidence_location != none`) when feeding the
  model; use `inferred_disease` to drop off-disease support.

**Wiring (done).** `de_ppi/build_literature_weighted_influence.py` now consumes these
searches directly — gene directions from `dysregulation_genes`, metabolite sink nodes
(+ `chebi_id`) from `dysregulation_metabolites`, unioned with the HMDB disease
metabolites (`hmdb_<disease_slug>/<disease_slug>_metabolite_chebi.tsv`, regenerated
from the raw HMDB XML by disease-association filter). The hand-staged
`crohn_macrophage_DE_literature.tsv` / `crohns_only_metabolites.tsv` dependency is
gone, and the standalone `build_metabolite_augmented_influence.py` /
`build_de_ppi_influence.py` builders were removed (the metabolite-sink and baseline
logic folded into the one literature-weighted builder). Metabolite **ChEBI grounding**
is wired (`resolve_ids.resolve_chebi`, EBI OLS, **exact label/synonym match only** over
a few normalized name variants — no fuzzy fallback, so non-metabolites the model
mis-lists stay unresolved; unresolved metabolites keep an empty `chebi_id` rather than
being dropped).

### de_ppi influence pipeline (generalized by `--build`)
`de_ppi/` builds a directed disease/cell-type PPI and scores how much each protein
**influences the dysregulated set**. It is parameterized by `--build <name>` via
`de_ppi/config.py` (which resolves every path from `builds_manifest.json` + the build's
slugs); per-build outputs land in `de_ppi/results/<build>/`.

```
de_ppi/
├── config.py                                  # load_build(name) -> resolved paths (read-only on the manifest)
├── build_literature_weighted_influence.py     # --build: node/edge/weight construction -> P^k influence
├── influence_analysis/
│   ├── drug_influence_table.py                # --build: OpenTargets drugs x influence rank/percentile
│   └── plot_phase_vs_percentile.py            # --build: clinical-phase vs influence-percentile boxplot
└── results/<build>/
    ├── P3_influence.tsv                        # per-protein influence + rank
    ├── networks/{network_nodes,network_edges}.tsv
    └── influence_analysis/{<disease_slug>_drug_influence.tsv, phase_vs_percentile.png}
```

- **Nodes** = PINNACLE cell-type proteins ∪ DE genes (padj<0.05) ∪ literature genes.
  OmniPath-orphans are dropped *unless* they reach a metabolite via a MIND edge.
- **Edges** = OmniPath protein→protein + MIND protein→metabolite (sink) edges.
- **Sender weights**, two tracks: DE genes get the paired rank-shift gate
  `w = min(exp(-(disease_rank − ref_rank)/tau), wmax)` (tau=4000, wmax=5); literature
  genes (not DE) get elevated→2.0 / suppressed→0.5; everything else 1.0.
- **Influence** `= sum_{k=0..3} (P^k @ m)`, `P[i,j] = w(i)/Z(j)`, `m` = the dysregulated
  set (DE ∪ literature ∪ metabolites). The k=0 **self-loop** credits a node for being
  itself in the target set; reach-only influence = `influence − is_target`.
- **influence_analysis/** joins the influence ranking to OpenTargets known drugs for the
  build's disease family (per-drug table) and plots influence-percentile by clinical
  phase, with comparison boxes for other macrophage diseases' drug targets.

### Cell-state definition (Leiden, disease-blind) — `rank_shifts/de_scripts/state_split.py`
All `<build>_states.py` are thin wrappers over one shared module. States are defined by
**unsupervised Leiden clustering** (normalize → HVG → PCA → kNN → Leiden) on the pooled
normal+disease cells, **disease-blind**; marker signatures only *name* clusters post hoc
(cluster → annotate → merge to the annotation label = the state), then per-state pseudobulk
DESeq2. This replaced the earlier hand-picked-marker **argmax** (which (1) could encode the
disease signal — gating conditions on the outcome, and (2) forced every cell into a state).
Disease-skewed clusters with no normal arm are reported **SKIPPED** — their signal is
**compositional** (a proportion shift), not within-state DE (e.g. the ILD profibrotic-SPP1+
state, Sjogren immunofibroblast). Needs `leidenalg`+`igraph`.

### Build roster (cross-disease) and backbones
Builds span four diseases on shared PINNACLE backbones, so the same cell type is comparable
across diseases:
- **Crohn** (gut): `macrophage_crohn`, `stem_crohn` (intestinal crypt stem), `fibroblast_crohn` — each + Leiden states.
- **Alzheimer** (brain): `microglia_alzheimers` (+states), `fibroblast_alzheimers`, `glutamatergic_neuron_alzheimers`.
- **ILD** (lung): `macrophage_ild` (+states) — the clean, well-powered, paired cross-disease macrophage comparator.
- Backbone rules: cell-type PINNACLE edgelist when it exists; **`--expressed`** expression-floor backbone
  (`precompute_expressed_genes.py`, detect≥0.065) when PINNACLE lacks the cell type (Tabula Sapiens has no
  cortical neuron → glutamatergic neuron uses this); `--no-lit`/no-OT for builds without a literature/drug panel.
- **PAIRED requirement:** DE needs the normal + disease arms in the *same* dataset(s) with ≥3 donors and ≥20
  cells/donor *each*. Cross-dataset (unpaired) builds are confounded — `macrophage_atherosclerosis_unpaired`
  failed this (housekeeping genes significantly "DE", disease ≡ dataset); not usable. Sjogren fibroblast was
  clean-paired but 0 DE (compositional disease, removed).

### Learned-embedding influence — `embed_influence.py` / `joint_embed_influence.py`
A learned complement to the analytic P^k reach (a NEW analytic-free encoder on the de_ppi networks —
**not** the retired R-GCN). A 2-layer weighted-directed message-passing encoder is trained by unsupervised
link prediction; influence(i) = `‖ d(Σ_{j∈m} z_j) / dx_i ‖` (Jacobian of the dysregulated-set readout).
The self-loop is gated by the node's sender weight `w(i)`.
- **`embed_influence.py --build`** → single-network `embedding_influence.tsv`.
- **`joint_embed_influence.py --out-name`** → ONE shared encoder + input table across many networks
  (a forward pass per network, loss summed), giving a comparable shared space. Each network is a **tag** =
  a subdir `results/<out_name>/networks/<tag>/`; tags mix cell types/states/diseases freely. Outputs:
  `joint_influence.tsv` (`influence_<tag>` per tag), `embedding_shift.tsv` (per-node ‖Z_a−Z_b‖, all pairs),
  `embeddings.npz`. Absent-in-a-network proteins are masked (NaN influence; excluded from shifts/plots).
- **`crohn_alzheimer_ild_embedding`** is the current cross-disease shared space: 15 Leiden-state tags across
  Crohn/Alzheimer/ILD. Plots: `plot_embedding_clusters.py` (PCA/t-SNE, colored by disease / cell type /
  cell-type-state), `plot_phase_joint_boxplot.py` (drug-target influence by phase + degree-matched-hub null;
  `--layout tag` for per-disease/cell/state target-vs-hub).
- **Flattening check** (`compare_joint_vs_single.py`): the joint embedding does NOT flatten per-network
  structure — vs single-network it has *higher* effective dimensionality and *better* edge reconstruction.
  Weak disease separation in 2D is a projection effect (cell type dominates the top variance), not flattening.

### Cross-network comparison caveat (raw influence is set-size-confounded)
Raw influence is a **sum over the dysregulated set**, so its magnitude tracks |set| (empirically
Spearman ≈ 0.99 across tags). DE-set size in turn scales with donor count (power), so ILD (49 disease
donors, ~2,000 targets) gives larger raw influence than Crohn macrophage (~6–11 donors, ~8–187 targets) —
an artifact, not "Crohn drugs work better in ILD." Normalizations each fail one way: **mean** (÷|set|)
favors small sets; **percentile** is inflated by out-degree (hubs rank high everywhere). The fair
cross-disease metric is a **degree-matched, size-matched permutation z-score** (the analytic
`specificity` column): observed influence on the real set vs random degree-matched same-size sets —
controls both set size (same-size null) and degree (matched). Use that (or per-tag target-vs-degree-matched-hub)
for cross-disease claims, never raw/mean/percentile alone.

## External UC ingest (Smillie 2019) — for a disease-similarity positive control

**Why:** to show the pipeline can capture *disease similarity* we need a gut-IBD comparator close to
Crohn (e.g. UC). The only UC in CellxGene census is dataset 19053a82, whose macrophage matrix is
**defective** — canonical markers absent across ALL arms incl. normal (CD68 ~15%, S100A8/9 0%, TNF 0%,
C1QA ~1%; verified per-arm), so it cannot represent inflammatory UC macrophages. So we go outside census.

**Source:** Smillie et al. 2019 Cell, *Broad Single Cell Portal SCP259* (the field-standard UC colon
atlas; healthy + UC donors within one study => within-study paired DE, proper myeloid capture). NOTE:
GSE116222 is a DIFFERENT paper (Parikh 2019, epithelial) — not Smillie. SCP259 download needs a
Broad/Google login, so it is NOT auto-pulled.

**Plan / files:**
1. `de_scripts/pull_smillie_uc.py` (written) — converts SCP259 immune-compartment files
   (`gene_sorted-Imm.matrix.mtx[.gz]`, `Imm.genes.tsv`, `Imm.barcodes2.tsv`, `all.meta2.txt`) placed in
   `rank_shifts/macrophage_uc_smillie_paired/_raw/` into `pulled_macrophages.h5ad` (raw counts, symbol
   var_names, obs `disease`∈{normal,ulcerative colitis}, `donor_id`=Subject, `cell_type`=macrophage),
   subsetting `Cluster`∈{Macrophages, Inflammatory Monocytes} and mapping `Health` Healthy→normal /
   Inflamed→UC (Non-inflamed dropped). Prints marker-capture sanity (CD68/S100A8/9/TNF) to confirm it
   is NOT defective like 19053a82.
2. Then clone `macrophage_crohn.py` DE (pseudobulk DESeq2, UC vs healthy) -> de_table; add manifest
   entry `macrophage_uc`; build network (expressed-backbone, rank-weight-all) + per-disease healthy.
3. Add to an embedding and run the Crohn-vs-UC displacement/heatmap as the disease-similarity test.

**Decisions (resolved):** (a) user downloads SCP259 immune files into
`rank_shifts/macrophage_uc_smillie_paired/_raw/`; (b) disease arm = **Inflamed vs Healthy** (Non-inflamed
dropped); (c) macrophage set = **Macrophages + Inflammatory Monocytes**. These match the script defaults
(`HEALTH_TO_DISEASE`, `MAC_CLUSTERS`). STATUS: blocked on the manual SCP259 download; ingest runs as soon
as the 4 files are in `_raw/`.
