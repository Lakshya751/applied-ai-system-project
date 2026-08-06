"""Plan schema, parsing, and validation.

This module is the system's guardrail. Model output is treated as untrusted
input: it is parsed defensively, checked field by field, checked for citations
it invented, and checked for scheduling conflicts against the live schedule
before a single Task object is created.

Validation failures are not fatal. They are fed back to the model as a repair
instruction, which is the "check its own work" step of the agent loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

VALID_FREQUENCIES = ("once", "daily", "weekly")

# Enforces zero-padded 24-hour time. This is not cosmetic: Scheduler.sort_by_time()
# sorts these values as plain strings, so an unpadded "8:00" would silently sort
# after "18:00". Validating the format here protects that existing invariant.
TIME_PATTERN = re.compile(r"^([01][0-9]|2[0-3]):[0-5][0-9]$")

MAX_DESCRIPTION_LENGTH = 80
MAX_TASKS_PER_PLAN = 12


class PlanParseError(ValueError):
    """Raised when model output cannot be read as a plan at all."""


@dataclass
class ProposedTask:
    """One task the model wants to add. Not yet trusted, not yet committed."""

    pet: str
    description: str
    time: str
    frequency: str
    reason: str = ""
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pet": self.pet,
            "description": self.description,
            "time": self.time,
            "frequency": self.frequency,
            "reason": self.reason,
            "sources": list(self.sources),
        }


@dataclass
class ValidationIssue:
    """One problem found in a proposed plan."""

    kind: str  # "schema" | "conflict" | "citation"
    message: str

    def __str__(self) -> str:
        return f"({self.kind}) {self.message}"


@dataclass
class ValidationReport:
    """Result of checking a plan: what survived, and what was wrong with it."""

    tasks: List[ProposedTask] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    caveats: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """A plan is valid when it has at least one task and no outstanding issues."""
        return bool(self.tasks) and not self.issues

    def issues_by_kind(self, kind: str) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.kind == kind]

    def repair_instruction(self) -> str:
        """Render the issues as a correction prompt for the model."""
        lines = [f"- {issue}" for issue in self.issues]
        return (
            "Your previous plan was rejected by the validator for these reasons:\n"
            + "\n".join(lines)
            + "\n\nReturn a corrected plan as JSON in the same schema. "
            "Fix every issue listed. Do not repeat the rejected values."
        )


def _strip_code_fences(raw: str) -> str:
    """Remove Markdown fences models add even when asked for raw JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_plan(raw: str) -> Dict[str, Any]:
    """Parse model output into a plan dict, tolerating common formatting noise.

    Falls back to extracting the outermost ``{...}`` span when the model wraps
    JSON in prose, which happens even with a JSON mime type requested.
    """
    text = _strip_code_fences(raw)
    if not text:
        raise PlanParseError("Model returned an empty response.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise PlanParseError(f"No JSON object found in response: {text[:120]!r}")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise PlanParseError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def _coerce_confidence(value: Any) -> float:
    """Clamp confidence into 0.0-1.0, defaulting to 0.0 when unusable."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def validate_plan(
    plan: Dict[str, Any],
    known_pets: Sequence[str],
    existing_times: Optional[Iterable[str]] = None,
    allowed_citations: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Check a parsed plan against the schema, the real pets, and the real schedule.

    ``known_pets``        - pet names that actually exist; anything else is a hallucination.
    ``existing_times``    - times already booked, so we can catch conflicts before committing.
    ``allowed_citations`` - citations actually retrieved; anything else was invented.
    """
    report = ValidationReport(
        summary=str(plan.get("summary", "")).strip(),
        confidence=_coerce_confidence(plan.get("confidence")),
        caveats=[str(c) for c in plan.get("caveats", []) if str(c).strip()],
    )

    pet_lookup = {name.lower(): name for name in known_pets}
    booked: Set[str] = set(existing_times or [])
    citation_whitelist: Set[str] = set(allowed_citations or [])

    raw_tasks = plan.get("tasks")
    if not isinstance(raw_tasks, list):
        report.issues.append(
            ValidationIssue("schema", "Top-level 'tasks' must be a list of task objects.")
        )
        return report

    if not raw_tasks:
        report.issues.append(ValidationIssue("schema", "The plan contained no tasks."))
        return report

    if len(raw_tasks) > MAX_TASKS_PER_PLAN:
        report.issues.append(
            ValidationIssue(
                "schema",
                f"Plan proposed {len(raw_tasks)} tasks; the maximum is {MAX_TASKS_PER_PLAN}.",
            )
        )
        return report

    proposed_times: Set[str] = set()

    for index, entry in enumerate(raw_tasks, start=1):
        label = f"task {index}"
        if not isinstance(entry, dict):
            report.issues.append(ValidationIssue("schema", f"{label} is not an object."))
            continue

        pet_raw = str(entry.get("pet", "")).strip()
        description = str(entry.get("description", "")).strip()
        time_value = str(entry.get("time", "")).strip()
        frequency = str(entry.get("frequency", "")).strip().lower()

        # --- pet must exist -------------------------------------------------
        if not pet_raw:
            report.issues.append(ValidationIssue("schema", f"{label} is missing 'pet'."))
            continue
        if pet_raw.lower() not in pet_lookup:
            report.issues.append(
                ValidationIssue(
                    "schema",
                    f"{label} targets unknown pet {pet_raw!r}. "
                    f"Valid pets: {', '.join(known_pets) or '(none)'}.",
                )
            )
            continue
        pet_name = pet_lookup[pet_raw.lower()]

        # --- description ----------------------------------------------------
        if not description:
            report.issues.append(ValidationIssue("schema", f"{label} is missing 'description'."))
            continue
        if len(description) > MAX_DESCRIPTION_LENGTH:
            report.issues.append(
                ValidationIssue(
                    "schema",
                    f"{label} description is {len(description)} chars; "
                    f"keep it under {MAX_DESCRIPTION_LENGTH}.",
                )
            )
            continue

        # --- time -----------------------------------------------------------
        if not TIME_PATTERN.match(time_value):
            report.issues.append(
                ValidationIssue(
                    "schema",
                    f"{label} has time {time_value!r}; expected zero-padded 24-hour HH:MM.",
                )
            )
            continue

        # --- frequency ------------------------------------------------------
        if frequency not in VALID_FREQUENCIES:
            report.issues.append(
                ValidationIssue(
                    "schema",
                    f"{label} has frequency {frequency!r}; "
                    f"expected one of {', '.join(VALID_FREQUENCIES)}.",
                )
            )
            continue

        # --- conflicts ------------------------------------------------------
        # Checked against both the live schedule and earlier tasks in this same
        # plan, so the model cannot double-book itself.
        if time_value in booked:
            report.issues.append(
                ValidationIssue(
                    "conflict",
                    f"{label} ({description}) is at {time_value}, "
                    "which already has a task scheduled. Choose a different time.",
                )
            )
            continue
        if time_value in proposed_times:
            report.issues.append(
                ValidationIssue(
                    "conflict",
                    f"{label} ({description}) reuses {time_value}, "
                    "already taken by an earlier task in this same plan.",
                )
            )
            continue

        # --- citations ------------------------------------------------------
        # Any citation outside the retrieved set was invented by the model.
        raw_sources = entry.get("sources") or []
        sources = [str(s).strip() for s in raw_sources if str(s).strip()]
        if citation_whitelist:
            invented = [s for s in sources if s not in citation_whitelist]
            if invented:
                report.issues.append(
                    ValidationIssue(
                        "citation",
                        f"{label} cites {invented!r}, which was not in the retrieved "
                        f"context. Cite only: {sorted(citation_whitelist)}.",
                    )
                )
                continue

        proposed_times.add(time_value)
        report.tasks.append(
            ProposedTask(
                pet=pet_name,
                description=description,
                time=time_value,
                frequency=frequency,
                reason=str(entry.get("reason", "")).strip(),
                sources=sources,
            )
        )

    return report
