"""Labeled validation set loading and synthetic-cohort generation.

The validation set is described declaratively in `labels.json` so it can grow
to the 100-1000 file scale without code changes: add files to a directory, add
one `collections` entry pointing at it, and the harness picks them up.

Three cohorts are kept separate on purpose, because averaging them produces a
number that means nothing:

* `third_party_ai`  - real generator output. Tests whether we can read provenance
                      somebody else wrote.
* `human`           - human-created audio. Tests the false-positive rate.
* `audiomark_marked`- audio this tool watermarked itself. Tests the embed/detect
                      round trip only; it says nothing about third-party
                      detection, so it is never folded into the headline metric.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LABELS_PATH = Path(__file__).resolve().parent / "labels.json"

# Ground-truth classes. `ai_assisted` is deliberately distinct: a human-produced
# track that used an AI tool somewhere is a different statutory question from a
# fully generated one, and folding it into "ai" would misstate both.
LABELS = ("ai", "human", "ai_assisted")


@dataclass
class Item:
    id: str
    path: Path
    label: str
    cohort: str
    provider: str | None = None
    system: str | None = None
    source: str | None = None
    expected_provenance: str = "unknown"
    label_confidence: str = "reported"
    notes: str = ""
    exists: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "label": self.label,
            "cohort": self.cohort,
            "provider": self.provider,
            "system": self.system,
            "source": self.source,
            "expected_provenance": self.expected_provenance,
            "label_confidence": self.label_confidence,
            "notes": self.notes,
            "exists": self.exists,
        }


def _resolve_roots(spec: dict[str, Any], overrides: dict[str, str] | None = None) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name, raw in (spec.get("roots") or {}).items():
        value = (overrides or {}).get(name) or os.environ.get(f"AUDIOMARK_EVAL_ROOT_{name.upper()}") or raw
        roots[name] = Path(value).expanduser()
    roots.setdefault("repo", Path(__file__).resolve().parents[1])
    return roots


def load_items(
    labels_path: Path = LABELS_PATH,
    root_overrides: dict[str, str] | None = None,
    include_missing: bool = False,
) -> list[Item]:
    """Load every labeled item, expanding `collections` globs."""
    spec = json.loads(labels_path.read_text(encoding="utf-8"))
    roots = _resolve_roots(spec, root_overrides)
    items: list[Item] = []

    for entry in spec.get("items", []):
        root = roots.get(entry.get("root", "repo"), roots["repo"])
        path = root / entry["path"]
        item = Item(
            id=entry["id"],
            path=path,
            label=entry["label"],
            cohort=entry.get("cohort", "third_party_ai"),
            provider=entry.get("provider"),
            system=entry.get("system"),
            source=entry.get("source"),
            expected_provenance=entry.get("expected_provenance", "unknown"),
            label_confidence=entry.get("label_confidence", "reported"),
            notes=entry.get("notes", ""),
            exists=path.exists(),
        )
        if item.exists or include_missing:
            items.append(item)

    for collection in spec.get("collections", []):
        root = roots.get(collection.get("root", "repo"), roots["repo"])
        pattern = collection["glob"]
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            items.append(
                Item(
                    id=f"{collection.get('id_prefix', 'item')}_{path.stem}",
                    path=path,
                    label=collection["label"],
                    cohort=collection.get("cohort", "third_party_ai"),
                    provider=collection.get("provider"),
                    system=collection.get("system"),
                    source=collection.get("source"),
                    expected_provenance=collection.get("expected_provenance", "unknown"),
                    label_confidence=collection.get("label_confidence", "reported"),
                    notes=collection.get("notes", ""),
                    exists=True,
                )
            )

    seen: set[str] = set()
    unique: list[Item] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        unique.append(item)
    return unique


def missing_items(labels_path: Path = LABELS_PATH, root_overrides: dict[str, str] | None = None) -> list[Item]:
    """Items declared in the label file whose audio is not on disk."""
    return [i for i in load_items(labels_path, root_overrides, include_missing=True) if not i.exists]


def cohort_summary(items: list[Item]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = summary.setdefault(item.cohort, {})
        bucket[item.label] = bucket.get(item.label, 0) + 1
    return summary
