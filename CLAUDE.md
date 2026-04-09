# Knowledge Graph Schema Discovery Agent

## Project goal
Build an agentic pipeline that takes a set of biological and chemical entities and produces a rich, structured context object for every node.

---

## Background and design decisions

### Unit of work
The unit of work is a **node**. Nodes represent biological or chemical entities including but not limited to diseases, phenotypes, proteins, small molecules, pathways, biological processes, and anatomical features. Each node gets a context object.

Relationships between nodes are captured as part of that context (ranked by
importance). Consider things such as entity location, entity prevalence, and entity interactors, at different biological scale for context.

### Why agentic
The starting schema is loaded from the archive (`output/archive/schema_final_N.json`,
highest N). The agent's job is to **validate and refine the controlled vocabularies**
by attempting to populate the schema for real nodes, discovering gaps, and
adjusting vocabulary terms. The schema fields themselves (names, descriptions,
types, applies_to_types) are **fixed** and must not be added, removed, or
renamed. Only the controlled vocabulary term lists may be modified.

### Entity summarization (LLM-based)
Prior to extracting data to the schema, each entity is first queried via an LLM
(OpenAI) to produce a summary of important biological/chemical information about
that entity. The LLM draws on all of its training knowledge — not a single
external source. The summary response is capped at **1,000 tokens** (`max_tokens=1000`).
This summary then serves as the source material for populating schema fields.

### External data sources
- **LLM knowledge** (via OpenAI API) is the primary source for entity context.
  No external search APIs are used.

### OpenAI Batch API
All LLM calls in Phase 1 (summarization) and Phase 2 (schema-field mapping)
use the **OpenAI Batch API**. This applies to every run — both the 100-node
test set and production runs on the full 250k node set.

Batch API mechanics:
1. Build a JSONL file where each line is a chat completion request:
   ```json
   {"custom_id": "node-DOID:123", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [...], "max_tokens": 1000}}
   ```
2. Upload the JSONL file via `client.files.create()`
3. Create a batch via `client.batches.create(input_file_id=..., endpoint="/v1/chat/completions", completion_window="24h")`
4. Poll `client.batches.retrieve(batch_id)` until status is `completed`
5. Download results via `client.files.content(output_file_id)`

Batch size: **5,000 nodes per batch**. For 100 nodes this means 1 batch per
phase; for 250,000 nodes this means 50 batches per phase.

Batch API pricing is **50% off** standard token pricing:
- gpt-4o-mini batch: $0.075/M input, $0.30/M output

Each batch submission produces three artifacts that must be persisted:

| Artifact | Location | Format |
|---|---|---|
| Batch identifiers | `output/batches/batch_ids/` | One JSON file per submission: `{batch_id, phase, batch_number, node_count, submitted_at, status}` |
| Batch inputs | `output/batches/inputs/` | JSONL files as submitted: `phase1_batch_001.jsonl`, `phase2_batch_001.jsonl` |
| Batch outputs | `output/batches/outputs/` | JSONL files returned: `phase1_batch_001_output.jsonl`, `phase2_batch_001_output.jsonl` |

Naming convention: `{phase}_{batch_number:03d}` (e.g., `phase1_batch_001`,
`phase2_batch_042`).

### Nodes
The nodes are found in `./db/nodes.csv`. The starting schema is loaded from
`./output/archive/schema_final_N.json` (the file with the highest N).

### Cost
Calculate the cost of the experiment up front and ask the user to approve a
certain dollar value to spend. Do not exceed this amount in API queries. Use
Batch API pricing (50% off) for the estimate.

---

## Starting schema

The starting schema (`output/archive/schema_final_N.json`, highest N) contains:
- **2 identity fields**: `id`, `name` (carried from CSV)
- **21 novel biological context fields**, each with a controlled vocabulary:
  organism, tissue_location, cell_type, cellular_compartment, biological_system,
  biological_scale, biological_process, molecular_function, pathway_category,
  mechanism_of_action, disease_association, clinical_relevance,
  phenotype_category, chemical_classification, drug_class, regulatory_role,
  interaction_type, inheritance_pattern, developmental_stage, taxonomic_domain,
  expression_context
- **No CSV columns are repeated** except id and name
- All context fields use `field_type: "controlled"` with vocabularies of 8–20 terms (max 20)

---

## Schema versioning

All schema and summary artifacts live in `output/archive/` with run-number
suffixes. There is no separate `output/schema/` folder.

At the **start of each run**:

1. Scan `output/archive/` for `schema_final_N.json` files.
2. Load the file with the **highest N** as the starting schema.
3. The new run's outputs use run number **N+1**:
   - `output/archive/schema_final_(N+1).json`
   - `output/archive/refinement_summary_(N+1).md`

No files are moved or deleted — each run appends new numbered files.

---

## Current task

Run **multiple iterations** of schema refinement in a single invocation.
Each iteration processes **100 diverse nodes** (different random sample each
time) and refines the controlled vocabularies. The schema output of iteration
N feeds as input to iteration N+1, creating a chained refinement process.

### CLI interface
```bash
python schema_agent.py --mode async --iterations 10
```
- `--iterations N` (default 10): number of refinement iterations to run
- `--mode batch|async` (default batch): API mode for Phase 1 & 2

### Per-iteration workflow
For each iteration (i = 1 … N):

1. **Load the latest schema** from `output/archive/schema_final_K.json`
   (highest K). On iteration 1 this is the pre-existing schema; on subsequent
   iterations it is the schema finalized by the previous iteration.
2. **Select 100 new diverse nodes** from `nodes.csv` (covering all 9 entity
   types, mixing high-degree and low-degree nodes). Each iteration draws a
   fresh random sample — nodes may repeat across iterations but each sample
   is independently randomized.
3. **Phase 1 — Summarize**: query the LLM to summarize each entity (capped at
   1,000 tokens).
4. **Phase 2 — Populate**: map each summary to schema fields. Each
   controlled-vocabulary field returns a **list of labels**.
5. **Phase 3 — Refine**: synchronous agent loop reviews population results and
   modifies controlled vocabularies only.
6. **Output three files** for this iteration:
   - `output/archive/schema_final_(K+1).json` — the refined schema
   - `output/archive/refinement_summary_(K+1).md` — structured per-field summary
   - `output/archive/nodes_(K+1).json` — the 100 populated nodes

### After the final iteration
All plots are generated automatically:
- `images/pca_context.png` — PCA of node context vectors (from latest nodes file)
- `images/node_types_by_iteration.png` — stacked barplot of entity types per iteration
- `images/term_changes_by_iteration.png` — grouped barplot of terms added/removed per iteration

### Budget
The user is prompted **once** for a per-iteration budget cap (in USD). This
same cap applies independently to each iteration. Total spend =
per-iteration budget × number of iterations (worst case).

#### Refinement summary format
For each controlled-vocabulary field, ranked by coverage % (highest first):
- **field_name** — coverage: XX% | applicable coverage: YY%
  - Terms added: term1, term2, ...
  - Count of terms added: N
  - Terms removed: term1, term2, ...
  - Count of terms removed: N

Where "coverage" = nodes with values / total, "applicable coverage" = nodes
with values / nodes where the LLM responded (excludes genuinely inapplicable).
All 21 fields must be listed. If no changes, show "Terms added: none" /
"Terms removed: none". Nothing else in the summary.

### Rules for this task
- Each iteration selects 100 new diverse nodes: cover all 9 entity types, mix
  high-degree and low-degree nodes. Samples are independently randomized.
- Use the LLM for entity context via Batch API or async API
- For each node, record which fields you could fill and which you could not
- When you encounter a value that fits a field but is not in the controlled
  vocabulary, add, rename, or merge vocabulary terms as appropriate
- **Schema fields are fixed** — do NOT add, remove, or rename fields. Only
  modify the controlled vocabulary term lists.
- **Maximum 20 unique terms** per controlled vocabulary. This is a hard limit
  enforced programmatically — vocabularies over 20 terms are truncated on save.
  Remove or merge terms before adding new ones if at the cap.
- **No null-like placeholder values** in any controlled vocabulary. Terms like
  "not_applicable", "unknown", "none", "not_specified", "none_known",
  "unclassified", "other", "not_a_drug", "not_organism_specific" are
  automatically stripped from vocabularies on save and from populated node
  values after Phase 2. If no vocabulary term fits a node's field, the value
  should be `null` (JSON null), not a placeholder term.
- Controlled-vocabulary fields return **lists** of labels, not single values
- Save at least 2 versioned checkpoints before finalizing
- When calling save_schema or finalize_schema, always pass the full schema object

---

## Architecture

### Pipeline overview
The pipeline runs an outer loop of I iterations (default 10). Each iteration
chains from the previous one's finalized schema.

**Outer loop** (for iteration i = 1 … I):
1. Load the latest schema from `output/archive/schema_final_K.json` (highest K)
   and set the output run number to K+1
2. Select 100 new diverse node IDs (fresh random sample)
3. **Phase 1 — Summarize** (Batch API or async):
   a. Build requests — one summarization request per node
   b. Submit and collect results
   c. Parse summaries
4. **Phase 2 — Populate** (Batch API or async):
   a. Build requests — one schema-mapping request per node
   b. Submit and collect results
   c. Parse populated node objects
   d. Write populated nodes to `output/archive/nodes_(K+1).json`
5. **Phase 3 — Refine vocabularies** (synchronous agent loop):
   a. Analyze Phase 2 results (coverage stats, term frequencies, suggested
      vocabulary additions)
   b. Agent loop receives aggregate analysis + current schema (NOT all 100 nodes)
   c. Agent modifies only the controlled vocabularies (not the fields), saves
      checkpoints, writes refinement summary, and finalizes
6. Clean up schema checkpoint intermediates from `output/archive/`
7. Output: `schema_final_(K+1).json`, `refinement_summary_(K+1).md`, and
   `nodes_(K+1).json` in `output/archive/`

**After final iteration**: generate all plots (PCA, node types, term changes).

### Tools the agent has access to (Phase 3 only)
- `save_schema(schema, version)` — checkpoint to `output/archive/`
- `finalize_schema(schema)` — saves `schema_final_N.json` in archive, ends session
- `write_summary(content)` — writes `refinement_summary_N.md` in archive

### Utility functions (used programmatically, not as agent tools)
- `get_type_distribution()` — understand graph composition
- `get_predicate_distribution()` — understand relationship types
- `sample_nodes(node_type, count, strategy)` — strategy: random | high_degree | low_degree
- `get_node_by_id(node_id)` — fetch a single node
- `build_summarize_request(node, custom_id)` — build a Phase 1 JSONL line
- `build_populate_request(node, summary, schema, custom_id)` — build a Phase 2 JSONL line
- `submit_batch(jsonl_path)` — upload file and create batch
- `poll_batch(batch_id)` — poll until completed, return output file ID
- `download_batch_results(output_file_id, dest_path)` — download output JSONL

### Schema output format
The final schema_final.json should include:
- `fields`: array of field definitions, each with:
  - `name` (snake_case)
  - `description`
  - `field_type`: string (id/name only) | controlled (everything else)
  - `controlled_vocabulary`: vocabulary name (key in controlled_vocabularies)
  - `applies_to_types`: list of node types, or empty for universal
  - `required`: boolean
- `controlled_vocabularies`: dict of vocab name → value list (8–20 terms each, max 20)
- `type_specific_fields`: dict of node type → field names
- `notes`: agent's observations about the domain

### Populated nodes output format
The output/archive/nodes_N.json should be an array of objects, one per test node.
Each controlled-vocabulary field is a **list** of matching labels (an entity may
belong to multiple categories). Fields that could not be determined are `null`.
```json
[
  {
    "id": "DOID:0001816",
    "name": "angiosarcoma",
    "organism": ["Homo sapiens"],
    "tissue_location": ["blood", "soft_tissue"],
    "cell_type": ["endothelial"],
    "cellular_compartment": null,
    "biological_system": ["cardiovascular"],
    ...
  }
]
```
Each value in a list must come from the corresponding controlled vocabulary.

---

## Stack and conventions

- **Language**: Python 3.11+
- **LLM**: OpenAI API via `openai` Python SDK (synchronous client)
- **Model**: `gpt-4o-mini` for all LLM calls
- **Batch API**: OpenAI Batch API for Phase 1 and Phase 2 (JSONL upload, poll, download)
- **Validation**: `pydantic` for all structured output
- **Checkpointing**: write to `output/archive/` after every meaningful step
- **Environment**: `OPENAI_API_KEY` set in `.env`

### Code conventions
- Use snake_case everywhere
- All tool functions return a plain dict (serialized to JSON for the API)
- Keep tool functions pure — no side effects except `save_schema`, `finalize_schema`, `write_summary`, `write_nodes`
- Hard ceiling of 40 agent turns in Phase 3 to prevent runaway loops
- Print progress and tool calls to stdout so the run is observable
- Phase 1 & 2 support two modes: synchronous Batch API or async direct API calls (`--mode batch` or `--mode async`)
- Multi-iteration support via `--iterations N` flag (default 10)
- Plots are generated automatically after the final iteration

---

## File structure
```
scripts/
  schema_agent.py     # main pipeline: Phase 1/2 (batch or async) + agent loop (Phase 3)
  plot_pca.py         # ad-hoc PCA plotting script
  plot_node_types.py  # stacked barplot of node type distribution per iteration
  color_scheme.py     # colorblind-friendly palette for entity types
  tools/
    graph_tools.py    # sample_nodes, get_type_distribution, get_predicate_distribution
    batch_tools.py    # build_summarize_request, build_populate_request, submit_batch, poll_batch, download_batch_results
    async_tools.py    # async direct-API alternatives to batch_tools (Phase 1 & 2)
    schema_tools.py   # load_latest_schema, save_schema, finalize_schema, write_summary, write_nodes, cleanup_checkpoints
output/
  archive/            # all versioned outputs (schema_final_N.json, refinement_summary_N.md, nodes_N.json)
  batches/
    batch_ids/        # one JSON file per batch submission (batch_id, phase, metadata)
    inputs/           # JSONL files submitted to the Batch API (phase1_batch_001.jsonl, etc.)
    outputs/          # JSONL files returned by the Batch API (phase1_batch_001_output.jsonl, etc.)
images/               # ad-hoc plots (PCA, etc.)
db/
  nodes.csv           # source node data (250k nodes, 9 entity types)
```
