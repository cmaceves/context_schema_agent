# Knowledge Graph Schema Discovery Agent

## Project goal
Build an agentic pipeline that takes a set of biological and chemical entities and produces a rich, structured context object for every node.

---

## Background and design decisions

### Unit of work
The unit of work is a **node**. Nodes represent biological or chemical entities including but not limited to diseases, phenotypes, proteins, small molecules, pathways, biological processes, and anatomical features. Each node gets a context object.

Relationships between nodes are captured as part of that context (ranked by
importance). Consider things such as entity location, entity prevelance, and entity iteractors, at different biological scale for context.

### Why agentic
The schema cannot be hand-designed upfront — it needs to emerge from the data.
The agent explores the graph, proposes a schema, tests it against
diverse nodes, reflects on what fits poorly, and iterates until it is satisfied.
The model decides what tools to call and when. It decides when it is done.

### No existing document corpus
There is no pre-existing document corpus. External context comes from Wikidata
(preferred, structured, free) and web search (fallback, used sparingly).
The agent should build its own local cache of retrieved context during the run
so repeated lookups hit local storage first.

### Nodes
The nodes are found in "./db/nodes.csv". Run this pipeline on a subsample (say 100) nodes to start with and return the results.

### Cost
Calculate the cost of the experiment up front and ask the user to approve a certain dollar value to spend. Do not exceed this amount in API queries.

---

## Architecture

### Agent loop
1. Agent receives the goal and graph statistics
2. Agent calls tools in whatever order it chooses:
   - Inspect graph structure (type distribution, predicate distribution)
   - Sample nodes (by type, degree, or randomly)
   - Look up external context for specific nodes (Wikidata, web search)
   - Propose a schema draft
   - Test the schema against a diverse set of nodes
   - Reflect, revise, and re-test
   - Save intermediate drafts as checkpoints
   - Finalize when satisfied
3. Output: schema_final.json in output/schema/

### Tools the agent has access to
- `get_type_distribution` — understand graph composition
- `get_predicate_distribution` — understand relationship types
- `sample_nodes(node_type, count, strategy)` — strategy: random | high_degree | low_degree
- `lookup_wikidata(entity_label)` — fetch structured external context
- `web_search(query)` — fallback for entities not in Wikidata
- `test_schema_against_nodes(schema, node_ids)` — self-evaluation
- `save_schema(schema, version)` — checkpoint to disk
- `finalize_schema(schema)` — saves, ends session

### Schema output format
The final schema_final.json should include:
- `fields`: array of field definitions, each with:
  - `name` (snake_case)
  - `description`
  - `field_type`: string | list | float | controlled
  - `controlled_vocabulary`: list of values (if controlled)
  - `applies_to_types`: list of node types, or empty for universal
  - `required`: boolean
- `controlled_vocabularies`: dict of vocab name → value list
- `type_specific_fields`: dict of node type → field names
- `notes`: agent's observations about the domain

---

## Stack and conventions

- **Language**: Python 3.11+
- **LLM**: Anthropic API via `anthropic` Python SDK (async)
- **Model**: `claude-sonnet-4-6` for the agent loop
- **Validation**: `pydantic` for all structured output
- **Async**: `asyncio` throughout, `aiohttp` for external requests
- **Checkpointing**: write to `output/schema/` after every meaningful step
- **Environment**: `ANTHROPIC_API_KEY` set in environment

### Code conventions
- Use snake_case everywhere
- All tool functions return a plain dict (serialized to JSON for the API)
- Keep tool functions pure — no side effects except `save_schema` and `finalize_schema`
- Hard ceiling of 40 agent turns to prevent runaway loops
- Print agent reasoning and tool calls to stdout so the run is observable

---

## File structure to generate
```
schema_agent.py       # main agent loop and tool dispatcher
tools/
  graph_tools.py      # sample_nodes, get_type_distribution, get_predicate_distribution
  external_tools.py   # lookup_wikidata, web_search
  schema_tools.py     # test_schema_against_nodes, save_schema, finalize_schema
output/
  schema/             # schema drafts and final output land here
```

---

## Current task
Generate the full project from scratch based on this spec. Start with:
1. `tools/graph_tools.py`
2. `tools/external_tools.py` — stub Wikidata and web search with a clear
   comment marking where real API calls go
3. `tools/schema_tools.py`
4. `schema_agent.py` — the agent loop that imports from the above

Do not generate placeholder code. Every function should be fully implemented.
```