"""
Knowledge Graph Schema Refinement Agent

Takes a starting schema and 10 diverse nodes, searches the web for biological
context, attempts to populate every field, and refines the schema based on
what it learns. Uses the OpenAI API with tool use.

Usage:
    source .venv/bin/activate
    python schema_agent.py
"""

import os
import sys
import json
import asyncio
import random
from pathlib import Path

from openai import AsyncOpenAI

from tools.graph_tools import (
    get_type_distribution,
    get_predicate_distribution,
    sample_nodes,
    get_node_by_id,
    _ensure_loaded,
    _nodes,
    _nodes_by_type,
)
from tools.external_tools import web_search
from tools.schema_tools import (
    test_schema_against_nodes,
    save_schema,
    finalize_schema,
    write_summary,
    write_nodes,
    archive_previous_outputs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
MAX_TURNS = 40

# Pricing for gpt-4o-mini (per million tokens)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

STARTING_SCHEMA_PATH = Path(__file__).resolve().parent / "output" / "schema" / "schema_final.json"

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_node_by_id",
            "description": "Fetch a single node by its exact ID (e.g. 'DOID:0001816').",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to look up.",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search Wikipedia for biological context about an entity. "
                "Returns article titles, snippets, and introductory extracts. "
                "Use this to gather context for populating schema fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (entity name + optional context like 'protein function' or 'disease symptoms').",
                    },
                },
                "required": ["query"],
            },
        },
    },
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
                "Save the final refined schema as schema_final.json. Call this "
                "AFTER you have processed all 10 nodes and written the summary. "
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
                "Write the refinement summary to output/schema/refinement_summary.md. "
                "Call this BEFORE finalize_schema. The summary should document what "
                "changed, per-node notes, and vocabulary additions."
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
    {
        "type": "function",
        "function": {
            "name": "write_nodes",
            "description": (
                "Write the populated node context objects to output/nodes.json. "
                "Pass an array of objects, one per test node, with every schema "
                "field filled in (use null for fields that could not be determined). "
                "Call this BEFORE finalize_schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Array of node objects with all schema fields populated.",
                    },
                },
                "required": ["nodes"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


async def dispatch_tool(name: str, input_args: dict) -> str:
    """Call the appropriate tool function and return its JSON result."""
    try:
        if name == "get_node_by_id":
            result = get_node_by_id(input_args.get("node_id", ""))
        elif name == "web_search":
            result = await web_search(input_args.get("query", ""))
        elif name == "save_schema":
            schema = input_args.get("schema")
            version = input_args.get("version", "unknown")
            if not schema:
                result = {"error": "Missing required 'schema' argument. Pass the full schema object."}
            else:
                result = save_schema(schema=schema, version=version)
        elif name == "finalize_schema":
            schema = input_args.get("schema")
            if not schema:
                result = {"error": "Missing required 'schema' argument. Pass the full schema object."}
            else:
                result = finalize_schema(schema=schema)
        elif name == "write_summary":
            content = input_args.get("content", "")
            if not content:
                result = {"error": "Missing required 'content' argument."}
            else:
                result = write_summary(content=content)
        elif name == "write_nodes":
            nodes = input_args.get("nodes")
            if not nodes:
                result = {"error": "Missing required 'nodes' argument. Pass the array of node objects."}
            else:
                result = write_nodes(nodes=nodes)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": f"Tool '{name}' raised: {type(e).__name__}: {e}"}

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class CostTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def cost(self) -> float:
        return (
            self.total_input_tokens * INPUT_COST_PER_M / 1_000_000
            + self.total_output_tokens * OUTPUT_COST_PER_M / 1_000_000
        )

    @property
    def remaining(self) -> float:
        return self.budget - self.cost

    def record(self, usage):
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens

    def check(self) -> bool:
        return self.cost < self.budget

    def summary(self) -> str:
        return (
            f"Tokens — in: {self.total_input_tokens:,}  out: {self.total_output_tokens:,} | "
            f"Cost: ${self.cost:.4f} / ${self.budget:.2f}"
        )


def estimate_cost() -> float:
    est_input = 200_000
    est_output = 60_000
    est_cost = (
        est_input * INPUT_COST_PER_M / 1_000_000
        + est_output * OUTPUT_COST_PER_M / 1_000_000
    )
    return round(est_cost, 2)


# ---------------------------------------------------------------------------
# Node selection — pick 10 diverse nodes
# ---------------------------------------------------------------------------


def select_diverse_nodes() -> list[dict]:
    """Pick 10 diverse nodes: one per entity type + extras from large types."""
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
    selected = []
    for etype in entity_types:
        pool = _nodes_by_type.get(etype, [])
        if pool:
            # Pick one node that has a non-empty name (not just an ID-like string)
            candidates = [n for n in pool if len(n.get("name", "")) > 3]
            if candidates:
                selected.append(random.choice(candidates))
    # Add one extra high-degree node for variety
    all_with_xrefs = [n for n in _nodes if len(n.get("xrefs", "").split("|")) > 5]
    if all_with_xrefs:
        selected.append(random.choice(all_with_xrefs))
    return selected[:10]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def build_system_prompt(type_summary: str, starting_schema: dict, node_list: list[dict]) -> str:
    node_descriptions = "\n".join(
        f"  {i+1}. {n['id']} — \"{n['name']}\" (type: {n['label']})"
        for i, n in enumerate(node_list)
    )

    schema_json = json.dumps(starting_schema, indent=2)

    return f"""You are a schema refinement agent for a biomedical knowledge graph.

## Your goal
You have a starting schema with 21 controlled-vocabulary biological context
fields. Your job is to VALIDATE and REFINE this schema by attempting to
populate it for 10 real nodes using web search (Wikipedia), discovering gaps,
adding new controlled vocabulary terms, and producing an improved schema.

## What you know about the graph
{type_summary}

## The 10 nodes you must process
{node_descriptions}

## Starting schema
```json
{schema_json}
```

## Your process — for EACH of the 10 nodes:
1. Call web_search with the node's name to get biological context.
   - If the first search is too generic, try a more specific query like
     "{{name}} {{type}} biology" or "{{name}} function".
2. Read the search results and determine what values each schema field
   should take for this node.
3. For each field, pick the best matching term from the controlled vocabulary.
4. If no existing term fits but the concept clearly belongs, NOTE the new
   term to add to that vocabulary.
5. If a field is genuinely inapplicable to this node, note that too.
6. Track your findings mentally — you'll need them for the summary.

## After processing all 10 nodes:
1. Update the schema:
   - ADD new terms to controlled vocabularies where you found gaps
   - Adjust field descriptions if they were unclear
   - Add new fields if you discovered an important dimension not covered
   - Do NOT remove existing fields or vocabulary terms
2. Call write_summary with a markdown document covering:
   - Overview of what changed
   - Per-node notes (what was easy/hard to populate)
   - List of all new vocabulary terms added and why
   - Any new fields added and rationale
3. Save at least 2 versioned checkpoints along the way
4. Call finalize_schema with the complete updated schema

## Rules
- Use web_search ONLY — do NOT use lookup_wikidata
- Process all 10 nodes — do not skip any
- When you call save_schema, finalize_schema, or write_summary, you MUST
  pass the required arguments (full schema object, content string)
- Only ADD to vocabularies — never remove existing terms
- Only ADD fields — never remove existing ones
- Be thorough but efficient — 1-2 web searches per node is usually enough"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def run_agent(budget: float) -> None:
    """Run the schema refinement agent loop."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    tracker = CostTracker(budget)

    # Archive previous run outputs
    archive_result = archive_previous_outputs()
    if archive_result["archived"]:
        print(f"Archived {len(archive_result['archived'])} files from run {archive_result['run_number'] - 1}:")
        for f in archive_result["archived"]:
            print(f"  → {f}")
    else:
        print("No previous outputs to archive.")

    # Load starting schema
    if not STARTING_SCHEMA_PATH.exists():
        print(f"ERROR: Starting schema not found at {STARTING_SCHEMA_PATH}")
        sys.exit(1)
    starting_schema = json.loads(STARTING_SCHEMA_PATH.read_text(encoding="utf-8"))

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

    # Select 10 diverse nodes
    diverse_nodes = select_diverse_nodes()
    print(f"\nSelected {len(diverse_nodes)} diverse nodes:")
    for i, n in enumerate(diverse_nodes):
        print(f"  {i+1}. {n['id']} — \"{n['name']}\" ({n['label']})")
    print()

    system = build_system_prompt(type_summary, starting_schema, diverse_nodes)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Begin refining the schema. Process each of the 10 nodes: "
                "search the web for each one, determine which schema fields "
                "can be populated, note missing vocabulary terms, and track "
                "your findings. After all 10 nodes, update the schema, write "
                "the refinement summary, save checkpoints, and finalize."
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
        print(f"Turn {turn}/{MAX_TURNS}  |  {tracker.summary()}")
        print(f"{'='*60}")

        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=16384,
            tools=TOOLS,
            messages=messages,
        )
        tracker.record(response.usage)

        choice = response.choices[0]
        message = choice.message

        if message.content:
            print(f"\n[Agent] {message.content}")

        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls or []
        tool_messages = []

        for tc in tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            print(f"\n[Tool call] {func_name}({json.dumps(func_args, ensure_ascii=False)[:200]})")
            result_str = await dispatch_tool(func_name, func_args)
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
                    "tools. Search the web for the next node, then continue "
                    "processing. When all 10 are done, write the summary and "
                    "finalize the schema."
                ),
            })

    else:
        print(f"\n{'='*60}")
        print(f"Reached max turns ({MAX_TURNS}).")
        print(tracker.summary())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    est = estimate_cost()
    print("=" * 60)
    print("Knowledge Graph Schema Refinement Agent")
    print("=" * 60)
    print(f"\nEstimated cost for a full run: ${est:.2f}")
    print(f"  (Based on ~{MAX_TURNS} turns with {MODEL})")
    print(f"  Input: ${INPUT_COST_PER_M}/M tokens | Output: ${OUTPUT_COST_PER_M}/M tokens")
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
    print("Starting agent...\n")

    asyncio.run(run_agent(budget))


if __name__ == "__main__":
    main()
