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

Take the starting schema and **100 diverse nodes** from `nodes.csv` (covering
all 9 entity types, mixing high-degree and low-degree nodes). For each node:

1. **Query the LLM** via the Batch API to summarize important
   biological/chemical information about the entity (capped at 1,000 tokens).
2. **Map the summary to schema fields** via a second Batch API call. Each
   controlled-vocabulary field should return a **list of labels** (an entity
   may belong to multiple categories).
3. **Track what works and what doesn't**: which fields are easy to populate,
   which need new or modified vocabulary terms, which are ambiguous or
   inapplicable.

After processing all 100 nodes:

4. **Refine the controlled vocabularies**: add, rename, or merge vocabulary
   terms as needed so they best fit the data without being too granular. Do
   NOT add, remove, or rename schema fields — only modify the term lists
   within `controlled_vocabularies`.
5. **Output three files**:
   - `output/archive/schema_final_N.json` — the updated/improved schema (N = previous run + 1)
   - `output/archive/refinement_summary_N.md` — structured per-field refinement
     summary (see format below)
   - `output/nodes.json` — the 100 test nodes with every schema field filled in
     using the values determined during the run. Each entry is a JSON object
     keyed by field name, with values drawn from controlled vocabularies.
     Each controlled-vocabulary field is a **list** (possibly empty) of matching
     labels. Fields that could not be determined should be set to `null`.

#### Refinement summary format
For each controlled-vocabulary field, ranked by coverage % (highest first):
- **field_name** — coverage: XX%
  - Terms added: term1, term2, ...
  - Count of terms added: N
  - Terms removed: term1, term2, ...
  - Count of terms removed: N

All 21 fields must be listed. If no changes, show "Terms added: none" /
"Terms removed: none". Nothing else in the summary.

### Rules for this task
- Select 100 nodes that are diverse: cover all 9 entity types, mix high-degree
  and low-degree nodes
- Use the LLM for entity context via Batch API
- For each node, record which fields you could fill and which you could not
- When you encounter a value that fits a field but is not in the controlled
  vocabulary, add, rename, or merge vocabulary terms as appropriate
- **Schema fields are fixed** — do NOT add, remove, or rename fields. Only
  modify the controlled vocabulary term lists.
- **Maximum 20 unique terms** per controlled vocabulary. If a vocabulary is at
  the cap, remove a term before adding a new one.
- **No null-like placeholder values** in any controlled vocabulary. Do not use
  terms like "not_applicable", "unknown", "none", "not_specified",
  "none_known", "unclassified", or "other". If no vocabulary term fits a
  node's field, the value should be `null` (JSON null), not a placeholder term.
- Controlled-vocabulary fields return **lists** of labels, not single values
- Save at least 2 versioned checkpoints before finalizing
- When calling save_schema or finalize_schema, always pass the full schema object

---

## Architecture

### Pipeline overview
1. Load the starting schema from `output/archive/schema_final_N.json` (highest N)
   and set the output run number to N+1
2. Select 100 diverse node IDs
3. **Phase 1 — Summarize** (Batch API):
   a. Build a JSONL file with one summarization request per node
   b. Upload and submit the batch
   c. Poll until complete, download results
   d. Parse summaries from the output JSONL
4. **Phase 2 — Populate** (Batch API):
   a. Build a JSONL file with one schema-mapping request per node (summary +
      schema → populated fields as lists of labels)
   b. Upload and submit the batch
   c. Poll until complete, download results
   d. Parse populated node objects from the output JSONL
   e. Write populated nodes directly to `output/nodes.json`
5. **Phase 3 — Refine vocabularies** (synchronous agent loop):
   a. Analyze Phase 2 results (coverage stats, term frequencies, suggested
      vocabulary additions)
   b. Agent loop receives aggregate analysis + current schema (NOT all 100 nodes)
   c. Agent modifies only the controlled vocabularies (not the fields), saves
      checkpoints, writes refinement summary, and finalizes
6. Output: `schema_final_(N+1).json` and `refinement_summary_(N+1).md` in
   `output/archive/`, and `nodes.json` in `output/`

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
The output/nodes.json should be an array of objects, one per test node.
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
- No asyncio or aiohttp — all I/O is synchronous or via Batch API

---

## File structure
```
schema_agent.py       # main pipeline: batch submission (Phase 1/2) + agent loop (Phase 3)
tools/
  graph_tools.py      # sample_nodes, get_type_distribution, get_predicate_distribution
  batch_tools.py      # build_summarize_request, build_populate_request, submit_batch, poll_batch, download_batch_results
  schema_tools.py     # load_latest_schema, save_schema, finalize_schema, write_summary, write_nodes
output/
  nodes.json          # populated context objects for test nodes (100) or full run (250k)
  archive/            # all schema versions and summaries (schema_final_N.json, refinement_summary_N.md)
  batches/
    batch_ids/        # one JSON file per batch submission (batch_id, phase, metadata)
    inputs/           # JSONL files submitted to the Batch API (phase1_batch_001.jsonl, etc.)
    outputs/          # JSONL files returned by the Batch API (phase1_batch_001_output.jsonl, etc.)
db/
  nodes.csv           # source node data (250k nodes, 9 entity types)
```
