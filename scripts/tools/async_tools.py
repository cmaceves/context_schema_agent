"""
Async direct-API tools for Phase 1 (summarization) and Phase 2 (population).

Drop-in alternative to batch_tools.py: same prompts, same output formats,
but uses AsyncOpenAI with concurrent requests instead of the Batch API.
"""

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from tools.batch_tools import (
    MODEL,
    _SUMMARIZE_PROMPT_TEMPLATE,
    _BATCH_INPUTS_DIR,
    _BATCH_OUTPUTS_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONCURRENT = 20  # semaphore limit for API calls

# Standard pricing (no batch discount)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60


# ---------------------------------------------------------------------------
# Phase 1: Async summarization
# ---------------------------------------------------------------------------


async def _summarize_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    node: dict,
    custom_id: str,
) -> dict:
    """Send a single summarization request. Returns a result dict matching batch output format."""
    prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(
        entity_name=node["name"],
        entity_type=node["label"],
    )
    async with sem:
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
    choice = response.choices[0]
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{"message": {"content": choice.message.content}}],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        },
    }


async def async_phase1_summarize(
    nodes: list[dict],
    client: AsyncOpenAI,
) -> tuple[list[dict], dict[str, str]]:
    """Run Phase 1 summarization with concurrent async requests.

    Returns (raw_results, summaries_by_custom_id) — same shape as batch mode.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        tasks.append(_summarize_one(client, sem, node, cid))

    print(f"[async] Phase 1: sending {len(tasks)} summarization requests (concurrency={MAX_CONCURRENT})...")
    results = await asyncio.gather(*tasks)

    # Parse into summaries dict (same as parse_phase1_results)
    summaries: dict[str, str] = {}
    for item in results:
        cid = item["custom_id"]
        choices = item["response"]["body"].get("choices", [])
        summaries[cid] = choices[0]["message"]["content"] if choices else ""

    success = sum(1 for s in summaries.values() if s)
    print(f"[async] Phase 1 complete: {success}/{len(nodes)} summaries obtained.")
    return list(results), summaries


# ---------------------------------------------------------------------------
# Phase 2: Async population
# ---------------------------------------------------------------------------


def _build_populate_prompt(node: dict, summary_text: str, schema: dict) -> str:
    """Build the populate prompt — mirrors batch_tools.build_populate_request."""
    fields = schema.get("fields", [])
    vocabs = schema.get("controlled_vocabularies", {})

    field_specs = []
    for f in fields:
        if f.get("field_type") != "controlled":
            continue
        vocab_name = f.get("controlled_vocabulary", "")
        terms = vocabs.get(vocab_name, [])
        field_specs.append(
            f'- {f["name"]}: {f.get("description", "")}  '
            f"Terms: {json.dumps(terms)}"
        )
    fields_block = "\n".join(field_specs)

    return (
        f"You are mapping biological context to schema fields.\n\n"
        f'Entity: "{node["name"]}" (ID: {node["id"]}, type: {node["label"]})\n\n'
        f"Summary:\n{summary_text}\n\n"
        f"Schema fields and their allowed controlled-vocabulary terms:\n{fields_block}\n\n"
        f"For EACH field listed above, return a JSON object where:\n"
        f"- Each key is a field name (use EXACTLY the field names listed above — do not add new fields)\n"
        f"- Each value is a LIST of matching vocabulary terms (the entity may match multiple)\n"
        f"- Use null if the field does not apply or cannot be determined\n"
        f'- Do NOT use placeholder values like "not_applicable", "unknown", "none", '
        f'"not_specified", "none_known", or similar — use null instead\n'
        f"- Only use terms from the provided vocabulary lists\n"
        f'- If a concept fits but no existing term matches, use the CLOSEST existing term '
        f'and also add a "suggested_additions" key mapping field names to suggested new vocabulary terms\n\n'
        f"Return valid JSON only. Example:\n"
        f'{{"organism": ["Homo sapiens"], "tissue_location": ["blood", "liver"], '
        f'"cell_type": null, "suggested_additions": {{"tissue_location": ["bone_marrow_stroma"]}}}}'
    )


async def _populate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    node: dict,
    summary_text: str,
    schema: dict,
    custom_id: str,
) -> dict:
    """Send a single population request. Returns a result dict matching batch output format."""
    prompt = _build_populate_prompt(node, summary_text, schema)
    async with sem:
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
    choice = response.choices[0]
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{"message": {"content": choice.message.content}}],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        },
    }


async def async_phase2_populate(
    nodes: list[dict],
    summaries: dict[str, str],
    schema: dict,
    client: AsyncOpenAI,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Run Phase 2 population with concurrent async requests.

    Returns (raw_results, populated_nodes, suggestions) — same shape as batch mode.
    """
    from tools.batch_tools import parse_phase2_results

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        summary_text = summaries.get(cid, "")
        if not summary_text:
            continue
        tasks.append(_populate_one(client, sem, node, summary_text, schema, cid))

    print(f"[async] Phase 2: sending {len(tasks)} population requests (concurrency={MAX_CONCURRENT})...")
    results = await asyncio.gather(*tasks)

    populated, suggestions = parse_phase2_results(list(results))

    # Restore identity fields using _custom_id set by parse_phase2_results
    node_by_cid = {f"{i:04d}_{n['id']}": n for i, n in enumerate(nodes)}
    for p in populated:
        cid = p.pop("_custom_id", "")
        if cid in node_by_cid:
            p["id"] = node_by_cid[cid]["id"]
            p["name"] = node_by_cid[cid]["name"]
            p["label"] = node_by_cid[cid]["label"]

    print(f"[async] Phase 2 complete: {len(populated)} nodes populated.")
    return list(results), populated, suggestions


# ---------------------------------------------------------------------------
# Cost estimation (standard pricing, no batch discount)
# ---------------------------------------------------------------------------


def estimate_async_cost(num_nodes: int) -> float:
    """Estimate total cost for Phase 1 + Phase 2 at standard (non-batch) pricing."""
    # Phase 1: ~300 input, ~800 output per node
    p1_in = num_nodes * 300
    p1_out = num_nodes * 800
    # Phase 2: ~3000 input, ~600 output per node
    p2_in = num_nodes * 3000
    p2_out = num_nodes * 600

    total_in = p1_in + p2_in
    total_out = p1_out + p2_out

    return (
        total_in * INPUT_COST_PER_M / 1_000_000
        + total_out * OUTPUT_COST_PER_M / 1_000_000
    )
