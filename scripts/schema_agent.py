"""
Knowledge Graph Schema Refinement Agent

Three-phase pipeline:
  Phase 1 — Summarize: one summarization request per node.
  Phase 2 — Populate: one schema-mapping request per node.
  Phase 3 — Refine: Synchronous agent loop reviews results and modifies the schema.

Modes:
  batch — OpenAI Batch API (cheap, slow).  Default.
  async — Direct async API calls (full price, fast).

Usage:
    source .venv/bin/activate
    python schema_agent.py              # batch mode (default)
    python schema_agent.py --mode async  # async mode
"""

import argparse
import os
import sys
import json
import asyncio
import random
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

from tools.graph_tools import (
    get_type_distribution,
    _ensure_loaded,
    _nodes,
    _nodes_by_type,
)
from tools.batch_tools import (
    build_summarize_request,
    build_populate_request,
    write_jsonl,
    submit_batch,
    poll_batch,
    download_batch_results,
    parse_phase1_results,
    parse_phase2_results,
    estimate_batch_cost,
    _BATCH_INPUTS_DIR,
    _BATCH_OUTPUTS_DIR,
    MODEL,
)
from tools.async_tools import (
    async_phase1_summarize,
    async_phase2_populate,
    estimate_async_cost,
)
from tools.schema_tools import (
    save_schema,
    finalize_schema,
    write_summary,
    write_nodes,
    load_latest_schema,
    set_run_number,
    cleanup_checkpoints,
    is_null_like,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TURNS = 40
NUM_NODES = 100

# Standard pricing for Phase 3 agent loop (non-batch)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

# Schema is loaded from output/archive/schema_final_N.json (highest N)
# via load_latest_schema() at pipeline start.


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class CostTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Batch tokens tracked separately (50% off pricing)
        self.batch_input_tokens = 0
        self.batch_output_tokens = 0

    @property
    def cost(self) -> float:
        # Standard-rate tokens (Phase 3 agent loop)
        standard = (
            self.total_input_tokens * INPUT_COST_PER_M / 1_000_000
            + self.total_output_tokens * OUTPUT_COST_PER_M / 1_000_000
        )
        # Batch-rate tokens (Phase 1 & 2) — 50% off
        batch = (
            self.batch_input_tokens * (INPUT_COST_PER_M / 2) / 1_000_000
            + self.batch_output_tokens * (OUTPUT_COST_PER_M / 2) / 1_000_000
        )
        return standard + batch

    @property
    def remaining(self) -> float:
        return self.budget - self.cost

    def record(self, usage):
        """Record usage from a standard (non-batch) API call."""
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens

    def record_batch(self, results: list[dict]):
        """Record usage from batch API results."""
        for item in results:
            usage = item.get("response", {}).get("body", {}).get("usage", {})
            self.batch_input_tokens += usage.get("prompt_tokens", 0)
            self.batch_output_tokens += usage.get("completion_tokens", 0)

    def check(self) -> bool:
        return self.cost < self.budget

    def summary(self) -> str:
        return (
            f"Batch tokens — in: {self.batch_input_tokens:,}  out: {self.batch_output_tokens:,} | "
            f"Agent tokens — in: {self.total_input_tokens:,}  out: {self.total_output_tokens:,} | "
            f"Cost: ${self.cost:.4f} / ${self.budget:.2f}"
        )


def estimate_total_cost(mode: str = "batch") -> float:
    """Estimate cost for 100 nodes: Phase 1 + Phase 2 + Phase 3 (standard)."""
    if mode == "async":
        p12_cost = estimate_async_cost(NUM_NODES)
    else:
        p12_cost = estimate_batch_cost(NUM_NODES)
    # Phase 3: agent loop — ~200k in, ~60k out at standard rates
    p3_cost = 200_000 * INPUT_COST_PER_M / 1_000_000 + 60_000 * OUTPUT_COST_PER_M / 1_000_000
    return round(p12_cost + p3_cost, 2)


# ---------------------------------------------------------------------------
# Node selection — pick 100 diverse nodes
# ---------------------------------------------------------------------------


def select_diverse_nodes() -> list[dict]:
    """Pick 100 diverse nodes across all entity types, mixing degree levels."""
    _ensure_loaded()
    entity_types = [
        "Disease",
        "MacromolecularMachine",
        "ChemicalSubstance",
        "BiologicalProcessOrActivity",
        "OrganismTaxon",
        "GeneFamily",
        "PhenotypicFeature",
        "Pathway",
        "AnatomicalEntity",
    ]

    def _degree(n: dict) -> int:
        xrefs = n.get("xrefs", "")
        return len(xrefs.split("|")) if xrefs else 0

    selected = []
    per_type = max(NUM_NODES // len(entity_types), 1)  # ~11 per type

    for etype in entity_types:
        pool = _nodes_by_type.get(etype, [])
        candidates = [n for n in pool if len(n.get("name", "")) > 3]
        if not candidates:
            continue

        high = sorted(candidates, key=_degree, reverse=True)[:max(per_type // 3, 1)]
        low = sorted(candidates, key=_degree)[:max(per_type // 3, 1)]
        remaining_pool = [n for n in candidates if n not in high and n not in low]
        rand_count = per_type - len(high) - len(low)
        rand_sample = random.sample(remaining_pool, min(rand_count, len(remaining_pool)))
        selected.extend(high + low + rand_sample)

    # Deduplicate by node ID
    seen_ids = set()
    deduped = []
    for n in selected:
        if n["id"] not in seen_ids:
            seen_ids.add(n["id"])
            deduped.append(n)
    selected = deduped

    if len(selected) > NUM_NODES:
        selected = random.sample(selected, NUM_NODES)
    elif len(selected) < NUM_NODES:
        all_candidates = [n for n in _nodes if len(n.get("name", "")) > 3 and n["id"] not in seen_ids]
        extra = random.sample(all_candidates, min(NUM_NODES - len(selected), len(all_candidates)))
        selected.extend(extra)

    return selected[:NUM_NODES]


# ---------------------------------------------------------------------------
# Phase 1: Summarize via Batch API
# ---------------------------------------------------------------------------


def phase1_summarize(nodes: list[dict], client: OpenAI) -> tuple[list[dict], dict[str, str]]:
    """Submit a summarization batch and return (raw_results, summaries_by_id)."""
    print("Building Phase 1 JSONL...")
    requests = []
    for i, node in enumerate(nodes):
        req = build_summarize_request(node, custom_id=f"{i:04d}_{node['id']}")
        requests.append(req)

    jsonl_path = _BATCH_INPUTS_DIR / "phase1_batch_001.jsonl"
    write_jsonl(requests, jsonl_path)

    batch_id = submit_batch(client, jsonl_path, phase="phase1", batch_number=1)
    result = poll_batch(client, batch_id)

    if result["status"] != "completed":
        print(f"ERROR: Phase 1 batch failed with status: {result['status']}")
        sys.exit(1)

    output_path = _BATCH_OUTPUTS_DIR / "phase1_batch_001_output.jsonl"
    raw_results = download_batch_results(client, result["output_file_id"], output_path)
    summaries = parse_phase1_results(raw_results)

    success = sum(1 for s in summaries.values() if s)
    print(f"Phase 1 complete: {success}/{len(nodes)} summaries obtained.")
    return raw_results, summaries


# ---------------------------------------------------------------------------
# Phase 2: Populate via Batch API
# ---------------------------------------------------------------------------


def phase2_populate(
    nodes: list[dict],
    summaries: dict[str, str],
    schema: dict,
    client: OpenAI,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Submit a population batch and return (raw_results, populated_nodes, suggestions)."""
    print("Building Phase 2 JSONL...")
    requests = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        summary_text = summaries.get(cid, "")
        if not summary_text:
            continue
        req = build_populate_request(node, summary_text, schema, custom_id=cid)
        requests.append(req)

    jsonl_path = _BATCH_INPUTS_DIR / "phase2_batch_001.jsonl"
    write_jsonl(requests, jsonl_path)

    batch_id = submit_batch(client, jsonl_path, phase="phase2", batch_number=1)
    result = poll_batch(client, batch_id)

    if result["status"] != "completed":
        print(f"ERROR: Phase 2 batch failed with status: {result['status']}")
        sys.exit(1)

    output_path = _BATCH_OUTPUTS_DIR / "phase2_batch_001_output.jsonl"
    raw_results = download_batch_results(client, result["output_file_id"], output_path)
    populated, suggestions = parse_phase2_results(raw_results)

    # Restore identity fields from nodes using _custom_id set by parse_phase2_results
    node_by_cid = {f"{i:04d}_{n['id']}": n for i, n in enumerate(nodes)}
    for p in populated:
        cid = p.pop("_custom_id", "")
        if cid in node_by_cid:
            p["id"] = node_by_cid[cid]["id"]
            p["name"] = node_by_cid[cid]["name"]
            p["label"] = node_by_cid[cid]["label"]

    print(f"Phase 2 complete: {len(populated)} nodes populated.")
    return raw_results, populated, suggestions


# ---------------------------------------------------------------------------
# Phase 2 analysis — build aggregate stats for the agent
# ---------------------------------------------------------------------------


def count_responded(populated: list[dict], schema: dict) -> dict[str, int]:
    """Count how many nodes have non-null values per field (before cleaning)."""
    fields = [f for f in schema.get("fields", []) if f.get("field_type") == "controlled"]
    responded: dict[str, int] = {}
    for f in fields:
        fn = f["name"]
        responded[fn] = sum(1 for n in populated if n.get(fn) is not None)
    return responded


def clean_populated_nodes(populated: list[dict], schema: dict) -> None:
    """Remove null-like placeholder values from populated node fields in-place."""
    controlled_fields = {
        f["name"]
        for f in schema.get("fields", [])
        if f.get("field_type") == "controlled"
    }
    for node in populated:
        for fn in controlled_fields:
            val = node.get(fn)
            if isinstance(val, list):
                cleaned = [t for t in val if not is_null_like(t)]
                node[fn] = cleaned if cleaned else None
            elif isinstance(val, str) and is_null_like(val):
                node[fn] = None


def analyze_population_results(
    populated: list[dict],
    schema: dict,
    suggestions: dict[str, list[str]],
    responded: dict[str, int],
) -> str:
    """Build a text summary of Phase 2 results for the agent."""
    fields = [f for f in schema.get("fields", []) if f.get("field_type") == "controlled"]
    field_names = [f["name"] for f in fields]

    coverage: dict[str, int] = {fn: 0 for fn in field_names}
    term_freq: dict[str, dict[str, int]] = {fn: {} for fn in field_names}

    for node in populated:
        for fn in field_names:
            val = node.get(fn)
            if val is not None and val != []:
                coverage[fn] += 1
                if isinstance(val, list):
                    for term in val:
                        term_freq[fn][term] = term_freq[fn].get(term, 0) + 1

    total = len(populated)
    lines = [
        f"## Phase 2 Population Results ({total} nodes)\n",
        "### Per-field coverage:",
    ]
    for fn in field_names:
        pct = coverage[fn] / total * 100 if total else 0
        resp = responded.get(fn, 0)
        app_pct = coverage[fn] / resp * 100 if resp else 0
        lines.append(
            f"  - {fn}: {coverage[fn]}/{total} ({pct:.0f}%) | "
            f"applicable: {coverage[fn]}/{resp} ({app_pct:.0f}%)"
        )

    lines.append("\n### Most frequent terms per field (top 5):")
    for fn in field_names:
        top = sorted(term_freq[fn].items(), key=lambda x: -x[1])[:5]
        if top:
            terms_str = ", ".join(f"{t} ({c})" for t, c in top)
            lines.append(f"  - {fn}: {terms_str}")
        else:
            lines.append(f"  - {fn}: (no values)")

    if suggestions:
        lines.append("\n### Suggested vocabulary additions from Phase 2:")
        for fn, terms in sorted(suggestions.items()):
            lines.append(f"  - {fn}: {terms}")

    # Show 5 example nodes
    lines.append("\n### Example populated nodes (5 of 100):")
    examples = populated[:5]
    for ex in examples:
        lines.append(f"\n**{ex.get('name', '?')}** ({ex.get('id', '?')}):")
        for fn in field_names:
            val = ex.get(fn)
            if val is not None:
                lines.append(f"  {fn}: {json.dumps(val)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3: Synchronous agent loop for schema refinement
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_schema",
            "description": (
                "Save a schema draft as a versioned checkpoint to disk. "
                "You MUST pass the full schema object as the 'schema' argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "object",
                        "description": "The complete schema object to save.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Version label (e.g. '2.0', '2.1').",
                    },
                },
                "required": ["schema", "version"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_schema",
            "description": (
                "Save the final refined schema as schema_final_N.json in the archive. "
                "Call this AFTER you have written the summary. "
                "You MUST pass the full schema object as the 'schema' argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "object",
                        "description": "The complete final schema object.",
                    },
                },
                "required": ["schema"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": (
                "Write the refinement summary to output/archive/refinement_summary_N.md. "
                "Call this BEFORE finalize_schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The full markdown content for refinement_summary.md.",
                    },
                },
                "required": ["content"],
            },
        },
    },
]


def dispatch_tool(name: str, input_args: dict) -> str:
    """Call the appropriate tool function and return its JSON result."""
    try:
        if name == "save_schema":
            schema = input_args.get("schema")
            version = input_args.get("version", "unknown")
            if not schema:
                result = {"error": "Missing required 'schema' argument."}
            else:
                result = save_schema(schema=schema, version=version)
        elif name == "finalize_schema":
            schema = input_args.get("schema")
            if not schema:
                result = {"error": "Missing required 'schema' argument."}
            else:
                result = finalize_schema(schema=schema)
        elif name == "write_summary":
            content = input_args.get("content", "")
            if not content:
                result = {"error": "Missing required 'content' argument."}
            else:
                result = write_summary(content=content)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": f"Tool '{name}' raised: {type(e).__name__}: {e}"}

    return json.dumps(result, ensure_ascii=False)


def build_refinement_prompt(
    type_summary: str,
    starting_schema: dict,
    analysis: str,
) -> str:
    """Build the system prompt for Phase 3 (schema refinement)."""
    schema_json = json.dumps(starting_schema, indent=2)

    return f"""You are a schema refinement agent for a biomedical knowledge graph.

## Context
You have a starting schema with 21 controlled-vocabulary fields. We have already:
1. Summarized 100 diverse nodes using LLM knowledge (Phase 1, Batch API)
2. Populated every schema field for all 100 nodes (Phase 2, Batch API)
3. Written the populated nodes to output/nodes.json (already done)

Your job is to REVIEW the population results and REFINE the controlled
vocabularies so they best fit the data without being too granular. Each
controlled-vocabulary field returns a LIST of labels (an entity may match
multiple).

CRITICAL CONSTRAINT: The schema fields are FIXED. Do NOT add, remove, or
rename any fields. Do NOT change field descriptions, field_type, applies_to_types,
or required flags. You may ONLY modify the term lists inside
`controlled_vocabularies`.

## Graph composition
{type_summary}

## Current schema
```json
{schema_json}
```

## Population analysis (from 100 nodes)
{analysis}

## Your tasks
1. Review the coverage stats, term frequencies, and suggested additions above.
2. REFINE the controlled vocabularies ONLY:
   - ADD new vocabulary terms where the LLM suggested additions or where coverage is low
   - MERGE or RENAME terms that are redundant or too granular
   - REMOVE vocabulary terms that are never used and don't add value
   - Do NOT add, remove, or rename fields
   - Do NOT change field descriptions, applies_to_types, or type_specific_fields
   - Each controlled vocabulary MUST NOT exceed 20 unique terms. This is a HARD
     LIMIT enforced programmatically — any vocabulary over 20 terms will be
     truncated on save. If a vocabulary is at 20 and you need to add a term,
     you MUST remove or merge an existing term first. Prefer making terms more
     general over exceeding the cap.
   - Do NOT include any placeholder or null-like values in vocabularies (e.g.
     "not_applicable", "unknown", "none", "not_specified", "none_known",
     "unclassified", "other", "not_a_drug", "not_organism_specific"). These
     will be automatically stripped on save. If no vocabulary term fits a node,
     the field value should be null — not a vocabulary term representing absence.
3. Save at least 2 versioned checkpoints (save_schema) with the FULL schema.
4. Write a refinement summary (write_summary) with the following structure for
   EACH controlled-vocabulary field, ranked by coverage percentage (highest first):
     - **field_name** — coverage: XX% | applicable coverage: YY%
       - Terms added: term1, term2, ...
       - Count of terms added: N
       - Terms removed: term1, term2, ...
       - Count of terms removed: N
   Where "coverage" = nodes with values / total nodes, and "applicable coverage"
   = nodes with values / nodes where the LLM responded (excludes nodes where
   the field is not applicable). Use the coverage stats from the analysis above.
   Include ALL 21 fields in the summary, even if no changes were made (show
   "Terms added: none" / "Terms removed: none" in that case). Do not include
   anything else in the summary — no overview, no examples, no commentary.
5. Finalize the schema (finalize_schema) with the FULL schema object.

IMPORTANT: When calling save_schema or finalize_schema, you MUST include the
complete schema JSON as the "schema" argument. Do not omit it. The schema must
retain all original fields unchanged — only `controlled_vocabularies` values
should differ from the input.

## Rules
- Be opinionated about vocabularies: remove unused terms, merge redundant ones
- Don't be too granular — prefer broader, well-populated vocabulary terms
- Maximum 20 terms per controlled vocabulary
- No null-like placeholder values in any vocabulary
- NEVER change the fields array, type_specific_fields, or field metadata
- Process everything and finalize in this session"""


def phase3_refine(
    starting_schema: dict,
    type_summary: str,
    analysis: str,
    client: OpenAI,
    tracker: CostTracker,
) -> None:
    """Run the synchronous agent loop for schema refinement."""
    system = build_refinement_prompt(type_summary, starting_schema, analysis)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Review the population results and modify the schema. "
                "Save checkpoints, write the summary, then finalize."
            ),
        },
    ]

    finalized = False

    for turn in range(1, MAX_TURNS + 1):
        if not tracker.check():
            print(f"\n{'='*60}")
            print(f"BUDGET EXHAUSTED after turn {turn - 1}.")
            print(tracker.summary())
            break

        print(f"\n{'='*60}")
        print(f"Phase 3 — Turn {turn}/{MAX_TURNS}  |  {tracker.summary()}")
        print(f"{'='*60}")

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=16384,
            tools=TOOLS,
            messages=messages,
        )
        tracker.record(response.usage)

        choice = response.choices[0]
        message = choice.message

        if message.content:
            preview = message.content[:500]
            if len(message.content) > 500:
                preview += "..."
            print(f"\n[Agent] {preview}")

        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls or []
        tool_messages = []

        for tc in tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            print(f"\n[Tool call] {func_name}({json.dumps(func_args, ensure_ascii=False)[:200]}...)")
            result_str = dispatch_tool(func_name, func_args)
            preview = result_str[:300] + "..." if len(result_str) > 300 else result_str
            print(f"[Tool result] {preview}")

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

            if func_name == "finalize_schema":
                result_data = json.loads(result_str)
                if result_data.get("finalized"):
                    finalized = True

        if finalized:
            messages.extend(tool_messages)
            print(f"\n{'='*60}")
            print("Schema finalized! Agent session complete.")
            print(tracker.summary())
            break

        if tool_messages:
            messages.extend(tool_messages)
        elif choice.finish_reason == "stop":
            messages.append({
                "role": "user",
                "content": (
                    "Don't just describe your plan — execute it now using your "
                    "tools. Save a checkpoint, write the summary, then "
                    "finalize the schema."
                ),
            })

    else:
        print(f"\n{'='*60}")
        print(f"Reached max turns ({MAX_TURNS}).")
        print(tracker.summary())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(budget: float, mode: str = "batch") -> None:
    """Run the full 3-phase pipeline.

    Parameters
    ----------
    budget : float
        Maximum USD to spend.
    mode : str
        "batch" for OpenAI Batch API (cheap, slow) or
        "async" for direct async API calls (full price, fast).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    tracker = CostTracker(budget)

    # Load the latest schema from output/archive/schema_final_N.json
    try:
        starting_schema, latest_n = load_latest_schema()
    except FileNotFoundError:
        print("ERROR: No schema found in output/archive/. Place a schema_final_N.json there first.")
        sys.exit(1)

    next_run = latest_n + 1
    set_run_number(next_run)
    print(f"Starting run {next_run} (based on schema from run {latest_n})")
    print(f"Mode: {mode}")

    # Pre-load graph data
    _ensure_loaded()
    type_dist = get_type_distribution()
    type_lines = "\n".join(
        f"  - {t}: {c:,} nodes" for t, c in type_dist["type_counts"].items()
    )
    type_summary = (
        f"Total nodes: {type_dist['total_nodes']:,}\n"
        f"Entity types ({type_dist['num_types']}):\n{type_lines}"
    )

    # Select 100 diverse nodes
    diverse_nodes = select_diverse_nodes()
    print(f"\nSelected {len(diverse_nodes)} diverse nodes:")
    type_counts = {}
    for n in diverse_nodes:
        type_counts[n["label"]] = type_counts.get(n["label"], 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print()

    if mode == "async":
        p1_results, summaries, p2_results, populated_nodes, suggestions = asyncio.run(
            _run_async_phases(diverse_nodes, starting_schema, api_key)
        )
    else:
        # ---- Phase 1: Summarize via Batch API ----
        print("=" * 60)
        print("PHASE 1: Summarizing entities via Batch API")
        print("=" * 60)
        p1_results, summaries = phase1_summarize(diverse_nodes, client)
        print(tracker.summary())

        # ---- Phase 2: Populate via Batch API ----
        print("\n" + "=" * 60)
        print("PHASE 2: Populating schema fields via Batch API")
        print("=" * 60)
        p2_results, populated_nodes, suggestions = phase2_populate(
            diverse_nodes, summaries, starting_schema, client
        )

    tracker.record_batch(p1_results)
    tracker.record_batch(p2_results)
    print(tracker.summary())

    # ---- Count responses before cleaning (for applicable coverage) ----
    responded = count_responded(populated_nodes, starting_schema)

    # ---- Clean null-like placeholder values from populated nodes ----
    clean_populated_nodes(populated_nodes, starting_schema)
    print("Cleaned null-like placeholder values from populated nodes.")

    # ---- Write populated nodes to archive ----
    print(f"\nWriting {len(populated_nodes)} populated nodes to output/archive/nodes_{next_run}.json...")
    write_nodes(populated_nodes)

    # ---- Phase 2 analysis ----
    analysis = analyze_population_results(populated_nodes, starting_schema, suggestions, responded)
    print("\n" + analysis)

    if not tracker.check():
        print("Budget exhausted after Phase 2.")
        return

    # ---- Phase 3: Agent refinement (synchronous) ----
    print("\n" + "=" * 60)
    print("PHASE 3: Agent-driven schema refinement")
    print("=" * 60)
    phase3_refine(starting_schema, type_summary, analysis, client, tracker)

    # ---- Cleanup checkpoint intermediates ----
    cleanup_result = cleanup_checkpoints()
    if cleanup_result["count"]:
        print(f"\nCleaned up {cleanup_result['count']} checkpoint file(s).")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(tracker.summary())
    print("=" * 60)


async def _run_async_phases(
    nodes: list[dict],
    schema: dict,
    api_key: str,
) -> tuple[list[dict], dict[str, str], list[dict], list[dict], dict[str, list[str]]]:
    """Run Phase 1 and Phase 2 using direct async API calls."""
    async_client = AsyncOpenAI(api_key=api_key)

    print("=" * 60)
    print("PHASE 1: Summarizing entities via async API")
    print("=" * 60)
    p1_results, summaries = await async_phase1_summarize(nodes, async_client)

    print("\n" + "=" * 60)
    print("PHASE 2: Populating schema fields via async API")
    print("=" * 60)
    p2_results, populated, suggestions = await async_phase2_populate(
        nodes, summaries, schema, async_client
    )

    return p1_results, summaries, p2_results, populated, suggestions


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Schema Refinement Agent")
    parser.add_argument(
        "--mode",
        choices=["batch", "async"],
        default="batch",
        help="batch = OpenAI Batch API (cheap, slow); async = direct async calls (full price, fast)",
    )
    args = parser.parse_args()
    mode = args.mode

    est = estimate_total_cost(mode)
    print("=" * 60)
    print("Knowledge Graph Schema Refinement Agent")
    print("=" * 60)
    mode_label = "Batch API (50% off)" if mode == "batch" else "Async direct API (standard pricing)"
    print(f"\nMode: {mode_label}")
    print(f"Estimated cost for {NUM_NODES}-node run: ${est:.2f}")
    print(f"  Phase 1 (summarize): {NUM_NODES} requests")
    print(f"  Phase 2 (populate):  {NUM_NODES} requests")
    print(f"  Phase 3 (refine):    synchronous agent loop (~{MAX_TURNS} turns max)")
    print(f"  Model: {MODEL}")
    print()

    budget_input = input(f"Enter your budget cap in USD (default ${est:.2f}): ").strip()
    if budget_input:
        try:
            budget = float(budget_input)
        except ValueError:
            print("Invalid number. Using estimate as budget.")
            budget = est
    else:
        budget = est

    print(f"\nBudget set to: ${budget:.2f}")
    print("Starting pipeline...\n")

    run_pipeline(budget, mode=mode)


if __name__ == "__main__":
    main()
