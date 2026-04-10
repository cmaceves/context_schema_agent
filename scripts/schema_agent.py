"""
Knowledge Graph Schema Refinement Agent

Three-phase pipeline run over multiple iterations:
  Phase 1 — Summarize: one summarization request per node.
  Phase 2 — Populate: one schema-mapping request per node.
  Phase 3 — Refine: Synchronous agent loop reviews results and modifies the schema.

Each iteration processes 100 new diverse nodes and refines the schema.
The finalized schema from iteration N feeds as input to iteration N+1.

Modes:
  batch — OpenAI Batch API (cheap, slow).  Default.
  async — Direct async API calls (full price, fast).

Usage:
    source .venv/bin/activate
    python schema_agent.py --mode async --iterations 10
"""

import argparse
import copy
import os
import sys
import json
import asyncio
import random
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI, AsyncOpenAI

from tools.graph_tools import (
    get_type_distribution,
    _ensure_loaded,
    _nodes,
    _nodes_by_id,
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
    normalize_term,
    MAX_VOCAB_SIZE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TURNS = 40
NUM_NODES = 500
STABLE_LOCK_THRESHOLD = 3  # lock a field after N consecutive stable iterations

# Phase 3 uses gpt-4o (more reliable for large structured tool outputs)
PHASE3_MODEL = "gpt-4o"

# gpt-4o-mini pricing for Phase 1 & 2 (batch or async)
MINI_INPUT_COST_PER_M = 0.15
MINI_OUTPUT_COST_PER_M = 0.60

# gpt-4o pricing for Phase 3 agent loop
P3_INPUT_COST_PER_M = 2.50
P3_OUTPUT_COST_PER_M = 10.00

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
        # Phase 3 tokens (gpt-4o, standard rate)
        standard = (
            self.total_input_tokens * P3_INPUT_COST_PER_M / 1_000_000
            + self.total_output_tokens * P3_OUTPUT_COST_PER_M / 1_000_000
        )
        # Phase 1 & 2 tokens (gpt-4o-mini, batch = 50% off, async = full)
        batch = (
            self.batch_input_tokens * MINI_INPUT_COST_PER_M / 1_000_000
            + self.batch_output_tokens * MINI_OUTPUT_COST_PER_M / 1_000_000
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


def estimate_per_iteration_cost(mode: str = "batch") -> float:
    """Estimate cost for a single 100-node iteration: Phase 1 + Phase 2 + Phase 3."""
    if mode == "async":
        p12_cost = estimate_async_cost(NUM_NODES)
    else:
        p12_cost = estimate_batch_cost(NUM_NODES)
    # Phase 3: gpt-4o agent loop — ~80k in, ~15k out (fewer turns than gpt-4o-mini)
    p3_cost = 80_000 * P3_INPUT_COST_PER_M / 1_000_000 + 15_000 * P3_OUTPUT_COST_PER_M / 1_000_000
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
    """Normalize terms and remove null-like placeholders from populated nodes."""
    controlled_fields = {
        f["name"]
        for f in schema.get("fields", [])
        if f.get("field_type") == "controlled"
    }
    for node in populated:
        for fn in controlled_fields:
            val = node.get(fn)
            if isinstance(val, list):
                cleaned = [normalize_term(t) for t in val if not is_null_like(t)]
                node[fn] = cleaned if cleaned else None
            elif isinstance(val, str):
                if is_null_like(val):
                    node[fn] = None
                else:
                    node[fn] = normalize_term(val)


def update_cumulative_freq(
    populated: list[dict],
    schema: dict,
    cumulative_freq: dict[str, dict[str, int]],
) -> None:
    """Update cumulative term frequency counts from populated nodes (in-place)."""
    fields = [f for f in schema.get("fields", []) if f.get("field_type") == "controlled"]
    for f in fields:
        fn = f["name"]
        if fn not in cumulative_freq:
            cumulative_freq[fn] = {}
        for node in populated:
            val = node.get(fn)
            if isinstance(val, list):
                for term in val:
                    cumulative_freq[fn][term] = cumulative_freq[fn].get(term, 0) + 1


def reconstruct_cross_iteration_state(
    archive_dir: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, int], set[str]]:
    """Reconstruct cumulative_freq, stability_counts, and locked_fields
    from existing nodes_N.json and schema_final_N.json files on disk.

    Returns (cumulative_freq, stability_counts, locked_fields).
    """
    # Discover all run numbers that have both a schema and a nodes file
    schema_files = sorted(archive_dir.glob("schema_final_*.json"))
    nodes_files = sorted(archive_dir.glob("nodes_*.json"))

    schema_by_n: dict[int, Path] = {}
    for p in schema_files:
        try:
            n = int(p.stem.split("_")[-1])
            schema_by_n[n] = p
        except ValueError:
            continue

    nodes_by_n: dict[int, Path] = {}
    for p in nodes_files:
        try:
            n = int(p.stem.split("_")[-1])
            nodes_by_n[n] = p
        except ValueError:
            continue

    # Run numbers that produced nodes (i.e. completed iterations)
    completed_runs = sorted(n for n in nodes_by_n if n in schema_by_n)

    if not completed_runs:
        return {}, {}, set()

    # --- 1. Cumulative term frequencies from all nodes files ---
    # Use the latest schema for field definitions
    latest_n = max(schema_by_n)
    latest_schema = json.loads(schema_by_n[latest_n].read_text(encoding="utf-8"))
    controlled_fields = [
        f["name"] for f in latest_schema.get("fields", [])
        if f.get("field_type") == "controlled"
    ]

    cumulative_freq: dict[str, dict[str, int]] = {fn: {} for fn in controlled_fields}
    for n in completed_runs:
        nodes = json.loads(nodes_by_n[n].read_text(encoding="utf-8"))
        for node in nodes:
            for fn in controlled_fields:
                val = node.get(fn)
                if isinstance(val, list):
                    for term in val:
                        cumulative_freq[fn][term] = cumulative_freq[fn].get(term, 0) + 1

    print(f"  Reconstructed cumulative frequencies from {len(completed_runs)} iteration(s)")

    # --- 2. Stability counts from consecutive schema comparisons ---
    # Walk completed runs in order; for each consecutive pair, check which
    # fields changed. Then count backwards from the latest run to find
    # how many consecutive iterations each field has been stable.
    changed_per_run: dict[int, set[str]] = {}
    for i, n in enumerate(completed_runs):
        # The input schema for run N is schema_final_(N-1)
        prev_n = n - 1
        if prev_n not in schema_by_n:
            continue
        prev_schema = json.loads(schema_by_n[prev_n].read_text(encoding="utf-8"))
        curr_schema = json.loads(schema_by_n[n].read_text(encoding="utf-8"))
        prev_vocabs = prev_schema.get("controlled_vocabularies", {})
        curr_vocabs = curr_schema.get("controlled_vocabularies", {})
        changed = set()
        for fn in controlled_fields:
            if sorted(prev_vocabs.get(fn, [])) != sorted(curr_vocabs.get(fn, [])):
                changed.add(fn)
        changed_per_run[n] = changed

    # Count consecutive stable iterations backwards from the latest
    stability_counts: dict[str, int] = {fn: 0 for fn in controlled_fields}
    for n in reversed(completed_runs):
        changed = changed_per_run.get(n, set())
        for fn in controlled_fields:
            if fn in changed:
                # This field changed in run n — stop counting for it
                break  # wrong: need per-field tracking
        # Actually, need to iterate field by field
    # Redo: walk backwards, for each field independently
    stability_counts = {}
    for fn in controlled_fields:
        count = 0
        for n in reversed(completed_runs):
            changed = changed_per_run.get(n, set())
            if fn in changed:
                break
            count += 1
        stability_counts[fn] = count

    # --- 3. Locked fields ---
    locked_fields = {
        fn for fn, count in stability_counts.items()
        if count >= STABLE_LOCK_THRESHOLD
    }

    print(f"  Stability counts: { {fn: stability_counts[fn] for fn in sorted(controlled_fields)} }")
    if locked_fields:
        print(f"  Locked fields: {', '.join(sorted(locked_fields))}")

    return cumulative_freq, stability_counts, locked_fields


def analyze_population_results(
    populated: list[dict],
    schema: dict,
    suggestions: dict[str, list[str]],
    responded: dict[str, int],
    cumulative_freq: dict[str, dict[str, int]] | None = None,
    locked_fields: set[str] | None = None,
) -> str:
    """Build a text summary of Phase 2 results for the agent."""
    fields = [f for f in schema.get("fields", []) if f.get("field_type") == "controlled"]
    field_names = [f["name"] for f in fields]
    locked = locked_fields or set()

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
        lock_tag = " [LOCKED]" if fn in locked else ""
        lines.append(
            f"  - {fn}: {coverage[fn]}/{total} ({pct:.0f}%) | "
            f"applicable: {coverage[fn]}/{resp} ({app_pct:.0f}%){lock_tag}"
        )

    lines.append("\n### This-iteration term frequencies (top 10):")
    for fn in field_names:
        if fn in locked:
            lines.append(f"  - {fn}: [LOCKED — skipped]")
            continue
        top = sorted(term_freq[fn].items(), key=lambda x: -x[1])[:10]
        if top:
            terms_str = ", ".join(f"{t} ({c})" for t, c in top)
            lines.append(f"  - {fn}: {terms_str}")
        else:
            lines.append(f"  - {fn}: (no values)")

    if cumulative_freq:
        lines.append("\n### Cumulative term frequencies across ALL iterations (top 10):")
        for fn in field_names:
            if fn in locked:
                lines.append(f"  - {fn}: [LOCKED — skipped]")
                continue
            cf = cumulative_freq.get(fn, {})
            top = sorted(cf.items(), key=lambda x: -x[1])[:10]
            if top:
                terms_str = ", ".join(f"{t} ({c})" for t, c in top)
                lines.append(f"  - {fn}: {terms_str}")
            else:
                lines.append(f"  - {fn}: (no cumulative data)")

    if suggestions:
        lines.append("\n### Suggested vocabulary additions from Phase 2:")
        for fn, terms in sorted(suggestions.items()):
            if fn not in locked:
                lines.append(f"  - {fn}: {terms}")

    # Show 5 example nodes
    lines.append(f"\n### Example populated nodes (5 of {total}):")
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
                "Save a schema draft as a versioned checkpoint. "
                "Pass ONLY the controlled_vocabularies dict — the rest of the "
                "schema (fields, type_specific_fields, notes) is merged automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "controlled_vocabularies": {
                        "type": "object",
                        "description": "The controlled_vocabularies dict mapping vocab name to list of terms.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Version label (e.g. '2.0', '2.1').",
                    },
                },
                "required": ["controlled_vocabularies", "version"],
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
                "Pass ONLY the controlled_vocabularies dict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "controlled_vocabularies": {
                        "type": "object",
                        "description": "The final controlled_vocabularies dict.",
                    },
                },
                "required": ["controlled_vocabularies"],
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


def dispatch_tool(name: str, input_args: dict, base_schema: dict) -> str:
    """Call the appropriate tool function and return its JSON result.

    For save_schema and finalize_schema, the agent passes only the
    controlled_vocabularies dict. We merge it into a deep copy of
    base_schema before saving, so the agent never needs to pass the
    full ~17KB schema object.
    """
    try:
        if name == "save_schema":
            vocabs = input_args.get("controlled_vocabularies")
            version = input_args.get("version", "unknown")
            if not vocabs:
                result = {"error": "Missing required 'controlled_vocabularies' argument."}
            else:
                merged = copy.deepcopy(base_schema)
                merged["controlled_vocabularies"] = vocabs
                result = save_schema(schema=merged, version=version)
        elif name == "finalize_schema":
            vocabs = input_args.get("controlled_vocabularies")
            if not vocabs:
                result = {"error": "Missing required 'controlled_vocabularies' argument."}
            else:
                merged = copy.deepcopy(base_schema)
                merged["controlled_vocabularies"] = vocabs
                result = finalize_schema(schema=merged)
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
    locked_fields: set[str] | None = None,
) -> str:
    """Build the system prompt for Phase 3 (schema refinement)."""
    schema_json = json.dumps(starting_schema, indent=2)
    locked = locked_fields or set()

    if locked:
        locked_list = "\n".join(f"  - {fn}" for fn in sorted(locked))
        locked_section = (
            f"\n## Locked fields\n"
            f"The following fields have been stable for {STABLE_LOCK_THRESHOLD}+ consecutive\n"
            f"iterations and are LOCKED — do NOT modify their vocabularies:\n"
            f"{locked_list}\n"
        )
    else:
        locked_section = ""

    return f"""You are a schema refinement agent for a biomedical knowledge graph.

## Context
You have a starting schema with 21 controlled-vocabulary fields. We have already:
1. Summarized {NUM_NODES} diverse nodes using LLM knowledge (Phase 1)
2. Populated every schema field for all {NUM_NODES} nodes (Phase 2)
3. Written the populated nodes to the archive (already done)

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

## Population analysis (from {NUM_NODES} nodes)
{analysis}
{locked_section}
## Your tasks
1. Review the coverage stats, term frequencies, and suggested additions above.
2. REFINE the controlled vocabularies ONLY:
   - ADD new vocabulary terms where the LLM suggested additions or where coverage is low
   - MERGE or RENAME terms that are redundant or too granular
   - REMOVE vocabulary terms that are never used and don't add value
   - Do NOT add, remove, or rename fields
   - Do NOT change field descriptions, applies_to_types, or type_specific_fields
   - Each controlled vocabulary MUST NOT exceed {MAX_VOCAB_SIZE} unique terms. This is a
     HARD LIMIT enforced programmatically — any vocabulary over {MAX_VOCAB_SIZE} terms
     will be truncated on save. If a vocabulary is at {MAX_VOCAB_SIZE} and you need to
     add a term, you MUST remove or merge an existing term first. Prefer making
     terms more general over exceeding the cap.
   - Do NOT include any placeholder or null-like values in vocabularies (e.g.
     "not_applicable", "unknown", "none", "not_specified", "none_known",
     "unclassified", "other", "not_a_drug", "not_organism_specific"). These
     will be automatically stripped on save. If no vocabulary term fits a node,
     the field value should be null — not a vocabulary term representing absence.
   - Fields marked [LOCKED] in the analysis are stable — do NOT modify their
     vocabularies. Skip them entirely.
3. Save at least 2 versioned checkpoints (save_schema) with the controlled_vocabularies dict.
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
5. Finalize the schema (finalize_schema) with the controlled_vocabularies dict.

IMPORTANT: When calling save_schema or finalize_schema, pass ONLY the
`controlled_vocabularies` dict (mapping vocab name to list of terms). The rest
of the schema is merged automatically. Do NOT pass the full schema object.

## Rules
- Be opinionated about vocabularies: remove unused terms, merge redundant ones
- Don't be too granular — prefer broader, well-populated vocabulary terms
- Maximum {MAX_VOCAB_SIZE} terms per controlled vocabulary
- No null-like placeholder values in any vocabulary
- NEVER change the fields array, type_specific_fields, or field metadata
- Fields marked [LOCKED] must not be modified
- Process everything and finalize in this session"""


def phase3_refine(
    starting_schema: dict,
    type_summary: str,
    analysis: str,
    client: OpenAI,
    tracker: CostTracker,
    locked_fields: set[str] | None = None,
) -> None:
    """Run the synchronous agent loop for schema refinement."""
    system = build_refinement_prompt(type_summary, starting_schema, analysis, locked_fields)

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
            model=PHASE3_MODEL,
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
            result_str = dispatch_tool(func_name, func_args, starting_schema)
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


def run_pipeline(
    budget: float,
    mode: str = "batch",
    cumulative_freq: dict[str, dict[str, int]] | None = None,
    locked_fields: set[str] | None = None,
) -> set[str] | None:
    """Run the full 3-phase pipeline.

    Parameters
    ----------
    budget : float
        Maximum USD to spend.
    mode : str
        "batch" for OpenAI Batch API (cheap, slow) or
        "async" for direct async API calls (full price, fast).
    cumulative_freq : dict, optional
        Mutable dict tracking term frequencies across iterations (updated in-place).
    locked_fields : set, optional
        Field names whose vocabularies should not be modified.

    Returns
    -------
    set[str] | None
        Set of field names whose vocabularies changed this iteration,
        or None if the iteration did not produce a finalized schema.
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

    # ---- Update cumulative term frequencies ----
    if cumulative_freq is not None:
        update_cumulative_freq(populated_nodes, starting_schema, cumulative_freq)
        print("Updated cumulative term frequencies.")

    # ---- Write populated nodes to archive ----
    print(f"\nWriting {len(populated_nodes)} populated nodes to output/archive/nodes_{next_run}.json...")
    write_nodes(populated_nodes)

    # ---- Phase 2 analysis ----
    analysis = analyze_population_results(
        populated_nodes, starting_schema, suggestions, responded,
        cumulative_freq=cumulative_freq,
        locked_fields=locked_fields,
    )
    print("\n" + analysis)

    if not tracker.check():
        print("Budget exhausted after Phase 2.")
        return None

    # ---- Phase 3: Agent refinement (synchronous) ----
    print("\n" + "=" * 60)
    print("PHASE 3: Agent-driven schema refinement")
    print("=" * 60)
    phase3_refine(starting_schema, type_summary, analysis, client, tracker, locked_fields)

    # ---- Cleanup checkpoint intermediates ----
    cleanup_result = cleanup_checkpoints()
    if cleanup_result["count"]:
        print(f"\nCleaned up {cleanup_result['count']} checkpoint file(s).")

    # ---- Determine which fields changed ----
    archive_dir = Path(__file__).resolve().parent.parent / "output" / "archive"
    finalized_path = archive_dir / f"schema_final_{next_run}.json"
    if finalized_path.exists():
        finalized_schema = json.loads(finalized_path.read_text(encoding="utf-8"))
        changed = set()
        old_vocabs = starting_schema.get("controlled_vocabularies", {})
        new_vocabs = finalized_schema.get("controlled_vocabularies", {})
        for fn in old_vocabs:
            if sorted(old_vocabs.get(fn, [])) != sorted(new_vocabs.get(fn, [])):
                changed.add(fn)
        print(f"\nFields changed this iteration: {sorted(changed) if changed else '(none)'}")
    else:
        print(f"\nWARNING: schema_final_{next_run}.json not found — iteration may have failed.")
        changed = None

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(tracker.summary())
    print("=" * 60)

    return changed


async def _run_async_phases(
    nodes: list[dict],
    schema: dict,
    api_key: str,
) -> tuple[list[dict], dict[str, str], list[dict], list[dict], dict[str, list[str]]]:
    """Run Phase 1 and Phase 2 using direct async API calls."""
    async_client = AsyncOpenAI(api_key=api_key)
    try:
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
    finally:
        await async_client.close()


def generate_plots() -> None:
    """Run all plotting scripts after the final iteration."""
    scripts_dir = Path(__file__).resolve().parent
    plot_scripts = [
        scripts_dir / "plot_pca.py",
        scripts_dir / "plot_node_types.py",
        scripts_dir / "plot_term_changes.py",
    ]
    for script in plot_scripts:
        if script.exists():
            print(f"\nRunning {script.name}...")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(scripts_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"  WARNING: {script.name} failed:\n{result.stderr.strip()}")
        else:
            print(f"  WARNING: {script.name} not found, skipping.")


def run_drug_disease_test(mode: str) -> None:
    """Run the drug-disease indication test.

    Loads graph.txt, filters to 'indication' edges, subsets to 500 rows,
    collects unique node IDs, matches them to nodes.csv, then runs Phase 1
    (summarize) and Phase 2 (populate) using the latest schema. Saves
    classified nodes to output/drug_disease_test/nodes_N.json without
    modifying the schema.
    """
    import csv as _csv

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # ---- Load graph.txt and filter to indication edges ----
    graph_path = Path(__file__).resolve().parent.parent / "db" / "graph.txt"
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found.")
        sys.exit(1)

    indication_rows = []
    with open(graph_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f, fieldnames=["head", "relation", "tail"], delimiter="\t")
        for row in reader:
            if row["relation"] == "indication":
                indication_rows.append(row)

    if not indication_rows:
        print("ERROR: No 'indication' rows found in graph.txt.")
        sys.exit(1)

    print(f"Found {len(indication_rows)} indication edges in graph.txt.")

    # Subset to 500 rows
    if len(indication_rows) > 500:
        indication_rows = random.sample(indication_rows, 500)
    print(f"Sampled {len(indication_rows)} indication edges.")

    # Collect unique node IDs from head and tail
    unique_ids = set()
    for row in indication_rows:
        unique_ids.add(row["head"])
        unique_ids.add(row["tail"])
    print(f"Unique node IDs from sampled edges: {len(unique_ids)}")

    # Match against nodes.csv
    _ensure_loaded()
    matched_nodes = [_nodes_by_id[nid] for nid in unique_ids if nid in _nodes_by_id]
    unmatched = unique_ids - set(n["id"] for n in matched_nodes)
    if unmatched:
        print(f"WARNING: {len(unmatched)} IDs not found in nodes.csv (skipped).")
    print(f"Matched {len(matched_nodes)} nodes in nodes.csv.")

    if not matched_nodes:
        print("ERROR: No matching nodes found. Nothing to classify.")
        sys.exit(1)

    # ---- Load latest schema ----
    try:
        schema, latest_n = load_latest_schema()
    except FileNotFoundError:
        print("ERROR: No schema found in output/archive/. Place a schema_final_N.json there first.")
        sys.exit(1)

    # ---- Cost estimate and budget prompt ----
    if mode == "async":
        est = estimate_async_cost(len(matched_nodes))
    else:
        est = estimate_batch_cost(len(matched_nodes))
    mode_label = "Batch API (50% off)" if mode == "batch" else "Async direct API (standard pricing)"
    print(f"\nMode: {mode_label}")
    print(f"Nodes to classify: {len(matched_nodes)}")
    print(f"Estimated cost (Phase 1 + Phase 2): ${est:.4f}")

    budget_input = input(f"Enter budget cap in USD (default ${est:.4f}): ").strip()
    if budget_input:
        try:
            budget = float(budget_input)
        except ValueError:
            print("Invalid number. Using estimate as budget.")
            budget = est
    else:
        budget = est

    tracker = CostTracker(budget)
    client = OpenAI(api_key=api_key)

    # ---- Phase 1: Summarize ----
    if mode == "async":
        async_client = AsyncOpenAI(api_key=api_key)
        p1_results, summaries = asyncio.run(
            async_phase1_summarize(matched_nodes, async_client)
        )
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: Summarizing entities via Batch API")
        print("=" * 60)
        p1_results, summaries = phase1_summarize(matched_nodes, client)

    tracker.record_batch(p1_results)
    print(tracker.summary())

    if not tracker.check():
        print("Budget exhausted after Phase 1.")
        sys.exit(1)

    # ---- Phase 2: Populate ----
    if mode == "async":
        p2_results, populated_nodes, _ = asyncio.run(
            async_phase2_populate(matched_nodes, summaries, schema, async_client)
        )
    else:
        print("\n" + "=" * 60)
        print("PHASE 2: Populating schema fields via Batch API")
        print("=" * 60)
        p2_results, populated_nodes, _ = phase2_populate(
            matched_nodes, summaries, schema, client
        )

    tracker.record_batch(p2_results)
    print(tracker.summary())

    # ---- Clean null-like placeholders ----
    clean_populated_nodes(populated_nodes, schema)
    print("Cleaned null-like placeholder values from populated nodes.")

    # ---- Save to output/drug_disease_test/ ----
    output_dir = Path(__file__).resolve().parent.parent / "output" / "drug_disease_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nodes_{latest_n}.json"
    output_path.write_text(
        json.dumps(populated_nodes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(populated_nodes)} classified nodes → {output_path}")
    print(tracker.summary())
    print("Drug-disease indication test complete.")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Schema Refinement Agent")
    parser.add_argument(
        "--mode",
        choices=["batch", "async"],
        default="batch",
        help="batch = OpenAI Batch API (cheap, slow); async = direct async calls (full price, fast)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of refinement iterations to run (default 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest schema_final_N.json in output/archive/",
    )
    parser.add_argument(
        "--drug_disease_test",
        action="store_true",
        help="Run drug-disease indication test: classify nodes from indication pairs and save to output/drug_disease_test/",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Per-iteration budget cap in USD (skips interactive prompt)",
    )
    args = parser.parse_args()
    mode = args.mode
    iterations = args.iterations

    # Drug-disease indication test — run and exit
    if args.drug_disease_test:
        print("=" * 60)
        print("Drug-Disease Indication Test")
        print("=" * 60)
        run_drug_disease_test(mode)
        return

    # When resuming, show which schema we're starting from
    if args.resume:
        try:
            _, latest_n = load_latest_schema()
            print("=" * 60)
            print("Knowledge Graph Schema Refinement Agent (RESUME)")
            print("=" * 60)
            print(f"\nResuming from schema_final_{latest_n}.json")
            print(f"Will produce schema_final_{latest_n + 1}.json through schema_final_{latest_n + iterations}.json")
        except FileNotFoundError:
            print("ERROR: --resume specified but no schema_final_N.json found in output/archive/.")
            sys.exit(1)
    else:
        print("=" * 60)
        print("Knowledge Graph Schema Refinement Agent")
        print("=" * 60)

    est_per = estimate_per_iteration_cost(mode)
    est_total = round(est_per * iterations, 2)
    mode_label = "Batch API (50% off)" if mode == "batch" else "Async direct API (standard pricing)"
    print(f"\nMode: {mode_label}")
    print(f"Iterations: {iterations}")
    print(f"Estimated cost per iteration ({NUM_NODES} nodes): ${est_per:.2f}")
    print(f"Estimated total cost ({iterations} iterations): ${est_total:.2f}")
    print(f"  Phase 1+2 (summarize+populate): {NUM_NODES} requests per iteration (gpt-4o-mini)")
    print(f"  Phase 3 (refine):    synchronous agent loop (~{MAX_TURNS} turns max) ({PHASE3_MODEL})")
    print()

    if args.budget is not None:
        budget = args.budget
    else:
        budget_input = input(f"Enter per-iteration budget cap in USD (default ${est_per:.2f}): ").strip()
        if budget_input:
            try:
                budget = float(budget_input)
            except ValueError:
                print("Invalid number. Using estimate as budget.")
                budget = est_per
        else:
            budget = est_per

    print(f"\nPer-iteration budget: ${budget:.2f}")
    print(f"Max total spend: ${budget * iterations:.2f}")
    print(f"Starting {iterations} iterations...\n")

    # Cross-iteration state
    if args.resume:
        archive_dir = Path(__file__).resolve().parent.parent / "output" / "archive"
        print("\nReconstructing cross-iteration state from previous runs...")
        cumulative_freq, stability_counts, locked_fields = (
            reconstruct_cross_iteration_state(archive_dir)
        )
    else:
        cumulative_freq: dict[str, dict[str, int]] = {}
        locked_fields: set[str] = set()
        stability_counts: dict[str, int] = {}

    for i in range(1, iterations + 1):
        print("\n" + "#" * 60)
        print(f"# ITERATION {i} / {iterations}")
        if locked_fields:
            print(f"# Locked fields ({len(locked_fields)}): {', '.join(sorted(locked_fields))}")
        print("#" * 60 + "\n")

        changed_fields = run_pipeline(
            budget,
            mode=mode,
            cumulative_freq=cumulative_freq,
            locked_fields=locked_fields,
        )

        if changed_fields is None:
            print(f"\nWARNING: Iteration {i} did not produce a finalized schema. "
                  "Skipping stability update.")
            continue

        # Update stability tracking — use cumulative_freq keys as the full field list
        all_fields = set(cumulative_freq.keys()) | changed_fields
        for fn in all_fields:
            if fn in changed_fields:
                stability_counts[fn] = 0
            else:
                stability_counts[fn] = stability_counts.get(fn, 0) + 1

        # Lock fields that have been stable for STABLE_LOCK_THRESHOLD iterations
        newly_locked = {
            fn for fn, count in stability_counts.items()
            if count >= STABLE_LOCK_THRESHOLD and fn not in locked_fields
        }
        if newly_locked:
            locked_fields |= newly_locked
            print(f"\nNewly locked fields (stable for {STABLE_LOCK_THRESHOLD}+ iterations): "
                  f"{', '.join(sorted(newly_locked))}")

        stable_summary = {fn: stability_counts.get(fn, 0) for fn in sorted(all_fields)}
        print(f"Stability counts: {stable_summary}")

    # Generate plots after all iterations
    print("\n" + "#" * 60)
    print("# GENERATING PLOTS")
    print("#" * 60)
    generate_plots()

    print("\n" + "#" * 60)
    print(f"# ALL {iterations} ITERATIONS COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()
