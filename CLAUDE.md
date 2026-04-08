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
The starting schema is already defined (see `output/schema/schema_final.json`).
The agent's job is now to **validate and refine** that schema by attempting to
populate it for real nodes using web search, discovering gaps, adding controlled
vocabulary terms, and producing a final improved schema.

### External data sources
- **Web search** (Wikipedia API) is the primary source for looking up biological context.
- **Wikidata is excluded** — do not use the `lookup_wikidata` tool.
- The agent should build its own local cache of retrieved context during the run
  so repeated lookups hit local storage first.

### Nodes
The nodes are found in `./db/nodes.csv`. The starting schema is in
`./output/schema/schema_final.json`.

### Cost
Calculate the cost of the experiment up front and ask the user to approve a certain dollar value to spend. Do not exceed this amount in API queries.

---

## Starting schema

The starting schema (`output/schema/schema_final.json`) contains:
- **2 identity fields**: `id`, `name` (carried from CSV)
- **21 novel biological context fields**, each with a controlled vocabulary:
  organism, tissue_location, cell_type, cellular_compartment, biological_system,
  biological_scale, biological_process, molecular_function, pathway_category,
  mechanism_of_action, disease_association, clinical_relevance,
  phenotype_category, chemical_classification, drug_class, regulatory_role,
  interaction_type, inheritance_pattern, developmental_stage, taxonomic_domain,
  expression_context
- **No CSV columns are repeated** except id and name
- All context fields use `field_type: "controlled"` with vocabularies of 8–35 terms

---

## Archiving previous runs

At the **start of each run**, before any processing:

1. Create `./output/archive/` if it does not exist.
2. If `output/schema/schema_final.json` exists, move it to
   `output/archive/schema_final_N.json` where N is the next available run
   number (if `_1` exists, use `_2`; if `_4` exists, use `_5`).
3. If `output/schema/refinement_summary.md` exists, move it to
   `output/archive/refinement_summary_N.md` using the same run number.
4. If `output/nodes.json` exists, move it to
   `output/archive/nodes_N.json` using the same run number.

This ensures every run's outputs are preserved and numbered sequentially.

---

## Current task

Take the starting schema and **10 diverse nodes** from `nodes.csv` (at least
one per major entity type). For each node:

1. **Search the web** (Wikipedia API only — no Wikidata) to gather biological
   context about the entity.
2. **Attempt to fill in every schema field** using the retrieved information.
3. **Track what works and what doesn't**: which fields are easy to populate,
   which need new vocabulary terms, which are ambiguous or inapplicable.

After processing all 10 nodes:

4. **Update the schema**: add any new controlled vocabulary terms discovered,
   adjust field descriptions if needed, add or remove fields based on what
   was learned.
5. **Output three files**:
   - `output/schema/schema_final.json` — the updated/improved schema
   - `output/schema/refinement_summary.md` — a summary of what changed and why,
     including per-node notes on what was easy/hard to populate
   - `output/nodes.json` — the 10 test nodes with every schema field filled in
     using the values determined during the run. Each entry is a JSON object
     keyed by field name, with values drawn from controlled vocabularies.
     Fields that could not be determined should be set to `null`.

### Rules for this task
- Select 10 nodes that are diverse: cover all 9 entity types, mix high-degree
  and low-degree nodes
- Use `web_search` (Wikipedia API) only — do NOT use `lookup_wikidata`
- For each node, record which fields you could fill and which you could not
- When you encounter a value that fits a field but is not in the controlled
  vocabulary, ADD the new term to the vocabulary
- Do not remove existing vocabulary terms — only add
- Do not remove existing fields — only add or refine descriptions
- Save at least 2 versioned checkpoints before finalizing
- When calling save_schema or finalize_schema, always pass the full schema object
- Archive existing outputs at the start of each run (see Archiving section above)

---

## Architecture

### Agent loop
1. Archive any existing output files from previous runs
2. Agent receives the starting schema and a list of 10 diverse node IDs
3. Agent calls tools in whatever order it chooses:
   - Fetch node details (`get_node_by_id`, `sample_nodes`)
   - Search the web for biological context (`web_search`)
   - Attempt to map retrieved context to schema fields
   - Track coverage gaps and missing vocabulary terms
   - Update the schema with new terms and refinements
   - Save checkpoints after meaningful revisions
   - Finalize when all 10 nodes have been processed and the schema is updated
4. Output: updated `schema_final.json`, `refinement_summary.md` in `output/schema/`,
   and `nodes.json` in `output/`

### Tools the agent has access to
- `get_type_distribution` — understand graph composition
- `get_predicate_distribution` — understand relationship types
- `sample_nodes(node_type, count, strategy)` — strategy: random | high_degree | low_degree
- `get_node_by_id(node_id)` — fetch a single node
- `web_search(query)` — search Wikipedia for biological context
- `test_schema_against_nodes(schema, node_ids)` — self-evaluation
- `save_schema(schema, version)` — checkpoint to disk
- `finalize_schema(schema)` — saves final version, ends session
- `write_summary(content)` — writes refinement_summary.md
- `write_nodes(content)` — writes the populated nodes.json

**Do NOT use**: `lookup_wikidata`

### Schema output format
The final schema_final.json should include:
- `fields`: array of field definitions, each with:
  - `name` (snake_case)
  - `description`
  - `field_type`: string (id/name only) | controlled (everything else)
  - `controlled_vocabulary`: vocabulary name (key in controlled_vocabularies)
  - `applies_to_types`: list of node types, or empty for universal
  - `required`: boolean
- `controlled_vocabularies`: dict of vocab name → value list (8+ terms each)
- `type_specific_fields`: dict of node type → field names
- `notes`: agent's observations about the domain

### Populated nodes output format
The output/nodes.json should be an array of objects, one per test node:
```json
[
  {
    "id": "DOID:0001816",
    "name": "angiosarcoma",
    "organism": "Homo sapiens",
    "tissue_location": "blood",
    "cell_type": "endothelial",
    "cellular_compartment": null,
    "biological_system": "cardiovascular",
    ...
  }
]
```
Each value must come from the corresponding controlled vocabulary, or be `null`
if the field is not applicable or could not be determined.

---

## Stack and conventions

- **Language**: Python 3.11+
- **LLM**: OpenAI API via `openai` Python SDK (async)
- **Model**: `gpt-4o-mini` for the agent loop
- **Validation**: `pydantic` for all structured output
- **Async**: `asyncio` throughout, `aiohttp` for external requests
- **Checkpointing**: write to `output/schema/` after every meaningful step
- **Environment**: `OPENAI_API_KEY` set in `.env`

### Code conventions
- Use snake_case everywhere
- All tool functions return a plain dict (serialized to JSON for the API)
- Keep tool functions pure — no side effects except `save_schema`, `finalize_schema`, `write_summary`, `write_nodes`
- Hard ceiling of 40 agent turns to prevent runaway loops
- Print agent reasoning and tool calls to stdout so the run is observable

---

## File structure
```
schema_agent.py       # main agent loop and tool dispatcher
tools/
  graph_tools.py      # sample_nodes, get_type_distribution, get_predicate_distribution
  external_tools.py   # web_search (Wikipedia API), lookup_wikidata (DO NOT USE)
  schema_tools.py     # test_schema_against_nodes, save_schema, finalize_schema, write_summary, write_nodes
output/
  schema/             # schema drafts, schema_final.json, refinement_summary.md
  nodes.json          # populated context objects for the 10 test nodes
  archive/            # previous runs' outputs (schema_final_N.json, refinement_summary_N.md, nodes_N.json)
  cache/              # local cache of web search results
db/
  nodes.csv           # source node data (250k nodes, 9 entity types)
```
