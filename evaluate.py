"""Evaluation harness for the PawPal+ AI layer.

Run:  python evaluate.py            (uses whatever provider .env configures)
      PAWPAL_PROVIDER=stub python evaluate.py   (offline baseline)

Runs three suites over the cases in evaluation/eval_cases.json:

  RETRIEVAL  - does the right document surface in the top-k, and are
               out-of-domain queries correctly rejected?
  VALIDATION - do the guardrails accept good plans and reject each specific
               category of bad one?
  PLANNING   - does the full agent loop produce a schedule that is internally
               consistent: valid times, real pets, no double-bookings?

Prints a summary table and writes evaluation/results.md so the numbers are
readable without re-running anything.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pawpal_ai.agent import CarePlanAgent
from pawpal_ai.config import load_settings
from pawpal_ai.retriever import Retriever
from pawpal_ai.schemas import TIME_PATTERN, validate_plan
from pawpal_system import Owner, Pet, Scheduler, Task

CASES_PATH = Path(__file__).parent / "evaluation" / "eval_cases.json"
RESULTS_PATH = Path(__file__).parent / "evaluation" / "results.md"

TOP_K = 3


@dataclass
class CaseResult:
    """Outcome of one evaluation case."""

    suite: str
    case_id: str
    description: str
    passed: bool
    detail: str = ""
    confidence: Optional[float] = None

    @property
    def mark(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class Report:
    results: List[CaseResult] = field(default_factory=list)

    def add(self, result: CaseResult) -> None:
        self.results.append(result)
        print(f"  [{result.mark}] {result.case_id:<4} {result.description}")
        if not result.passed and result.detail:
            print(f"         -> {result.detail}")

    def for_suite(self, suite: str) -> List[CaseResult]:
        return [r for r in self.results if r.suite == suite]

    def tally(self, suite: Optional[str] = None) -> str:
        subset = self.for_suite(suite) if suite else self.results
        return f"{sum(1 for r in subset if r.passed)}/{len(subset)}"

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def mean_confidence(self) -> Optional[float]:
        scores = [r.confidence for r in self.results if r.confidence is not None]
        return round(sum(scores) / len(scores), 2) if scores else None


# --- suites -----------------------------------------------------------------


def run_retrieval(cases: List[Dict[str, Any]], report: Report) -> None:
    """Check that the expected document appears in the top-k results."""
    print("\nRETRIEVAL — does the right passage surface?")
    retriever = Retriever()

    for case in cases:
        query = case["query"]
        expected = set(case.get("expect_sources", []))
        hits = retriever.search(query, top_k=TOP_K)
        found = [hit.chunk.source for hit in hits]
        top_score = hits[0].score if hits else 0.0

        if not expected:
            # Out-of-domain: retrieving anything at all is the failure.
            passed = not hits
            detail = f"expected no results, got {found}"
        else:
            passed = bool(expected.intersection(found))
            detail = f"expected one of {sorted(expected)}, got {found}"

        report.add(
            CaseResult(
                suite="retrieval",
                case_id=case["id"],
                description=f'"{query[:52]}"',
                passed=passed,
                detail=detail,
                confidence=round(top_score, 2) if expected else None,
            )
        )


def run_validation(cases: List[Dict[str, Any]], report: Report) -> None:
    """Check the guardrails accept good plans and reject each bad category."""
    print("\nVALIDATION — do the guardrails hold?")

    for case in cases:
        result = validate_plan(
            case["plan"],
            known_pets=case.get("known_pets", []),
            existing_times=case.get("existing_times"),
            allowed_citations=case.get("allowed_citations"),
        )
        should_pass = case["should_pass"]
        passed = result.is_valid == should_pass

        # When we expect a rejection, also check it was rejected for the right reason.
        expected_issue = case.get("expect_issue")
        if passed and expected_issue and not result.issues_by_kind(expected_issue):
            passed = False
            detail = (
                f"rejected, but not for a '{expected_issue}' reason: "
                f"{[str(i) for i in result.issues]}"
            )
        else:
            detail = (
                f"expected {'accept' if should_pass else 'reject'}, "
                f"got {'accept' if result.is_valid else 'reject'} "
                f"({[str(i) for i in result.issues]})"
            )

        report.add(
            CaseResult(
                suite="validation",
                case_id=case["id"],
                description=case["name"],
                passed=passed,
                detail=detail,
            )
        )


def run_planning(cases: List[Dict[str, Any]], report: Report, agent: CarePlanAgent) -> None:
    """Run the full loop and assert the resulting schedule is internally consistent."""
    print("\nPLANNING — is the produced schedule internally consistent?")

    for case in cases:
        owner = Owner(case["owner"])
        for name, species in case["pets"]:
            owner.add_pet(Pet(name, species))
        for description, time in case.get("existing_tasks", []):
            owner.pets[0].add_task(Task(description, time))

        result = agent.plan(owner, case["request"])
        failures: List[str] = []

        if not result.ok:
            failures.append(f"agent returned no usable plan: {result.error}")
        else:
            known = {pet.name for pet in owner.pets}
            for task in result.tasks:
                if not TIME_PATTERN.match(task.time):
                    failures.append(f"invalid time {task.time!r}")
                if task.pet not in known:
                    failures.append(f"unknown pet {task.pet!r}")

            if len(result.tasks) < case.get("min_tasks", 1):
                failures.append(
                    f"only {len(result.tasks)} task(s), expected >= {case.get('min_tasks', 1)}"
                )

            # Commit, then ask the existing Scheduler whether it double-booked.
            agent.commit(owner, result)
            conflicts = Scheduler(owner).detect_conflicts()
            if conflicts:
                failures.append(f"schedule has conflicts: {conflicts}")

        report.add(
            CaseResult(
                suite="planning",
                case_id=case["id"],
                description=case["name"],
                passed=not failures,
                detail="; ".join(failures),
                confidence=result.confidence,
            )
        )


# --- reporting --------------------------------------------------------------


def write_results(report: Report, provider_label: str, degraded: bool) -> None:
    """Write a Markdown summary so results are readable without re-running."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mean_conf = report.mean_confidence()

    lines = [
        "# Evaluation Results",
        "",
        "Generated by `python evaluate.py`. Do not edit by hand.",
        "",
        f"- **Provider:** {provider_label}",
        f"- **Overall:** {report.tally()} cases passed",
        f"- **Retrieval:** {report.tally('retrieval')}",
        f"- **Validation:** {report.tally('validation')}",
        f"- **Planning:** {report.tally('planning')}",
    ]
    if mean_conf is not None:
        lines.append(f"- **Mean confidence / top retrieval score:** {mean_conf}")
    if degraded:
        lines += [
            "",
            "> **Note:** this run used the offline stub provider, so the planning suite "
            "measures the deterministic fallback rather than a language model. It is the "
            "floor the live model should beat, not a measure of model quality.",
        ]

    lines += ["", "## Cases", "", "| Suite | ID | Case | Result | Detail |", "|---|---|---|---|---|"]
    for result in report.results:
        detail = "" if result.passed else result.detail.replace("|", "\\|")[:150]
        description = result.description.replace("|", "\\|")
        lines.append(
            f"| {result.suite} | {result.case_id} | {description} | {result.mark} | {detail} |"
        )

    lines.append("")
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(Path(__file__).parent)}")


def main() -> int:
    if not CASES_PATH.exists():
        print(f"Cannot find {CASES_PATH}")
        return 2

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    settings = load_settings()
    agent = CarePlanAgent(settings=settings)

    print("=" * 74)
    print("PawPal+ evaluation")
    print("=" * 74)
    print(f"Provider: {settings.describe()}")

    report = Report()
    run_retrieval(cases["retrieval"], report)
    run_validation(cases["validation"], report)
    run_planning(cases["planning"], report, agent)

    degraded = not settings.is_live
    print("\n" + "=" * 74)
    print(
        f"RESULT: {report.tally()} cases passed  "
        f"(retrieval {report.tally('retrieval')}, "
        f"validation {report.tally('validation')}, "
        f"planning {report.tally('planning')})"
    )
    mean_conf = report.mean_confidence()
    if mean_conf is not None:
        print(f"Mean confidence / top retrieval score: {mean_conf}")
    if degraded:
        print("NOTE: offline stub provider — planning results measure the fallback, not a model.")
    print("=" * 74)

    write_results(report, settings.describe(), degraded)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
