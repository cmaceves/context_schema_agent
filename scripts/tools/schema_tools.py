"""
Schema management tools: loading, checkpointing, and finalizing.

All schema artifacts live in output/archive/ with run-number suffixes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "archive"
_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

_current_run_number: int | None = None

MAX_VOCAB_SIZE = 30
NULL_LIKE_TERMS = frozenset({
    "not_applicable", "unknown", "none", "not_specified",
    "none_known", "unclassified", "other", "n/a", "na",
    "not_a_drug", "not_organism_specific",
})


def is_null_like(term: str) -> bool:
    """Check if a vocabulary term is a null-like placeholder."""
    if term is None:
        return True
    return term.lower().strip() in NULL_LIKE_TERMS


def normalize_term(term: str) -> str:
    """Lowercase a term and replace spaces with underscores."""
    return term.strip().lower().replace(" ", "_")


def _clean_vocabularies(schema: dict) -> list[str]:
    """Normalize terms, remove null-like terms, and enforce term cap.

    Modifies schema in-place. Returns a list of warning strings.
    """
    warnings = []
    vocabs = schema.get("controlled_vocabularies", {})
    for vocab_name, terms in vocabs.items():
        # Normalize all terms
        normalized = [normalize_term(t) for t in terms]
        # Remove null-like terms
        cleaned = [t for t in normalized if not is_null_like(t)]
        # Deduplicate (normalization may create duplicates)
        seen = set()
        deduped = []
        for t in cleaned:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        cleaned = deduped
        removed_nulls = sorted(set(normalized) - set(cleaned) - set(t for t in normalized if is_null_like(t)))
        stripped_nulls = [t for t in normalized if is_null_like(t)]
        if stripped_nulls:
            warnings.append(
                f"{vocab_name}: stripped null-like terms: {sorted(set(stripped_nulls))}"
            )
        # Enforce cap
        if len(cleaned) > MAX_VOCAB_SIZE:
            overflow = cleaned[MAX_VOCAB_SIZE:]
            warnings.append(
                f"{vocab_name}: truncated from {len(cleaned)} to {MAX_VOCAB_SIZE} "
                f"(dropped: {[t for t in overflow]})"
            )
            cleaned = cleaned[:MAX_VOCAB_SIZE]
        vocabs[vocab_name] = cleaned
    return warnings


def set_run_number(n: int) -> None:
    """Set the run number for the current pipeline execution."""
    global _current_run_number
    _current_run_number = n


def load_latest_schema() -> tuple[dict, int]:
    """Load the schema_final_N.json with the highest N from the archive.

    Returns
    -------
    (schema_dict, N) where N is the run number of the loaded schema.
    Raises FileNotFoundError if no schema exists in the archive.
    """
    existing = list(_ARCHIVE_DIR.glob("schema_final_*.json"))
    nums: list[tuple[int, Path]] = []
    for p in existing:
        stem = p.stem  # e.g. schema_final_3
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            nums.append((int(parts[1]), p))

    if not nums:
        print("[schema] ERROR: No schema_final_N.json found in output/archive/")
        raise FileNotFoundError("No schema found in output/archive/")

    nums.sort(key=lambda x: x[0])
    latest_n, latest_path = nums[-1]
    schema = json.loads(latest_path.read_text(encoding="utf-8"))
    print(f"[schema] Loaded schema from {latest_path} (run {latest_n})")
    return schema, latest_n


def save_schema(schema: dict, version: str) -> dict:
    """Save a schema draft to disk as a versioned checkpoint."""
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    warnings = _clean_vocabularies(schema)
    for w in warnings:
        print(f"[checkpoint] WARNING: {w}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"schema_checkpoint_v{version}_{ts}.json"
    path = _ARCHIVE_DIR / filename
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[checkpoint] Saved schema version {version} → {path}")
    result = {"saved": True, "path": str(path), "version": version}
    if warnings:
        result["warnings"] = warnings
    return result


def finalize_schema(schema: dict) -> dict:
    """Save the final schema as schema_final_N.json in the archive."""
    if _current_run_number is None:
        return {"error": "Run number not set. Call set_run_number() first."}
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    warnings = _clean_vocabularies(schema)
    for w in warnings:
        print(f"[finalize] WARNING: {w}")
    path = _ARCHIVE_DIR / f"schema_final_{_current_run_number}.json"
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[finalize] Saved final schema → {path}")
    result = {"finalized": True, "path": str(path), "run_number": _current_run_number}
    if warnings:
        result["warnings"] = warnings
    return result


def write_summary(content: str) -> dict:
    """Write the refinement summary to output/archive/refinement_summary_N.md."""
    if _current_run_number is None:
        return {"error": "Run number not set. Call set_run_number() first."}
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARCHIVE_DIR / f"refinement_summary_{_current_run_number}.md"
    path.write_text(content, encoding="utf-8")
    print(f"[summary] Saved refinement summary → {path}")
    return {"saved": True, "path": str(path), "run_number": _current_run_number}


def write_nodes(nodes: list[dict]) -> dict:
    """Write populated node context objects to output/archive/nodes_N.json."""
    if _current_run_number is None:
        return {"error": "Run number not set. Call set_run_number() first."}
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARCHIVE_DIR / f"nodes_{_current_run_number}.json"
    path.write_text(
        json.dumps(nodes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[nodes] Saved {len(nodes)} populated nodes → {path}")
    return {"saved": True, "path": str(path), "count": len(nodes)}


def cleanup_checkpoints() -> dict:
    """Remove schema checkpoint intermediates from the archive directory."""
    removed = []
    for p in _ARCHIVE_DIR.glob("schema_checkpoint_*.json"):
        p.unlink()
        removed.append(p.name)
        print(f"[cleanup] Removed checkpoint {p.name}")
    return {"removed": removed, "count": len(removed)}
