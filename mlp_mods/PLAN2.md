# PLAN2.md — Disease state-shift embedding on a PINNACLE PPI

## Goal

Embed the proteins of a disease and its healthy reference in a **single shared
encoder** over a **bidirectional PINNACLE PPI**, where each protein's **outgoing**
edge weights are set by its **rank shift** between states. The healthy and disease
embeddings live in one comparable space; the **embedding shift** (and pairwise
**influence / reach**) is the object of interest — used to ask which proteins'
network context reorganizes in disease and which nodes can *reach* the
dysregulated set (control-point / target reasoning).

This is run as a controlled single-cell-type case (macrophage / Crohn's), but the
build stage is **parameterized by (cell type, disease)** so other builds can be
added.

---

## Course we settled on (the model)

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

## 4-stage layout

```
mlp_mods/
├── PLAN2.md
├── 01_expression/            # Stage 1 - fetch single-cell expression (CellxGene pulls)
│   ├── pull_healthy_macrophage.py, pull_ibd_family.py
│   └── *.h5ad
├── 02_build_ppi/             # Stage 2 - build a disease PPI + save Wilcoxon ranks
│   ├── build_disease_ppi.py        # --build <name>: markers-on-PINNACLE -> vocab; full ranks -> rank_shifts/
│   ├── builds_manifest.json        # registry of (cell type, disease) builds + knobs
│   ├── builds/<build>/             # node_vocab.tsv, build_config.json
│   ├── reference_h5ad/             # body-wide reference for the marker Wilcoxon
│   └── _provenance/                # superseded one-off build scripts + intermediates (kept)
├── 03_opentargets_rebuild/   # Stage 3 - disease labels (OpenTargets positive/negative)
├── 04_state_shift_encoder/   # Stage 4 - the shared encoder + analyses
│   ├── gnn_utils.py                # load_vocab(build), make_features, build_relational_edges, EdgeDecoder
│   ├── weighted_encoder.py         # rank-weighted directed R-GCN
│   ├── train.py                    # --build <name>: train, write embeddings + embedding_shift
│   ├── influence.py                # per-arm pairwise influence (Jacobian) + delta
│   ├── reach_scoring.py            # structural P^k reach to dysregulated set (hops 1-3), targets flagged
│   ├── influence_reach.py          # encoder-Jacobian reach: sum_g ||dz_g/dx_i||_1 over dysregulated g
│   ├── plot_shift_scatter.py
│   ├── plot_rankshift_influence.py # scatter: rank_shift (x) vs influence_on_dysregulated (y)
│   └── results/<build>/            # per-build outputs
├── rank_shifts/<build>/      # per-build full Wilcoxon ranks: healthy.tsv, <disease>.tsv
├── go_cache/                 # GO annotations for validation (re-fetch; see README)
└── images/
```

## Pipeline

```
# stage 2: build PPI + ranks for a (cell type, disease)
.venv/bin/python mlp_mods/02_build_ppi/build_disease_ppi.py --build macrophage_crohn

# stage 4: encode, influence, reach (su-devenv has torch_geometric)
su-devenv/bin/python mlp_mods/04_state_shift_encoder/train.py         --build macrophage_crohn
su-devenv/bin/python mlp_mods/04_state_shift_encoder/influence.py     --build macrophage_crohn
su-devenv/bin/python mlp_mods/04_state_shift_encoder/reach_scoring.py --build macrophage_crohn
```

Convention: each build's reference arm is named **`healthy`** (binary sender
weight); the other arm is the disease arm (rank-drop-gated). Build outputs and
encoder results are keyed by build name, so multiple diseases/cell types coexist.

## Open validation thread
External agreement (does the shift recover known IBD biology?): GO/pathway
enrichment of high-shift / high-reach genes (go_cache), and correlation against
an independent IBD intestinal proteomics signature (e.g. PXD001608), with a
degree-matched / label-permuted null. Targeting = reach + ESM (druggability).

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
