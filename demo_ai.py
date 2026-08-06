"""End-to-end walkthrough of the PawPal+ AI layer.

Run:  python demo_ai.py

Prints three scenarios that together exercise every part of the system:
retrieval, planning, validation, self-repair, human approval, grounded question
answering, and the out-of-domain guardrail.

The output of this script is what is pasted into README.md as sample
interactions, so it is written to be readable rather than clever.
"""

from __future__ import annotations

import sys

from pawpal_ai.agent import AgentResult, CarePlanAgent
from pawpal_ai.config import load_settings
from pawpal_system import Owner, Pet, Scheduler, Task

RULE = "=" * 74


def header(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_proposal(result: AgentResult) -> None:
    """Render a proposal the way the UI does: plan, evidence, then trace."""
    if result.error and not result.tasks:
        print(f"  ERROR: {result.error}")
        return

    print(f"\n  Summary: {result.summary or '(none)'}")
    print(
        f"  Confidence: {result.confidence:.2f}"
        f"   Repairs: {result.repairs}"
        f"   Mode: {'OFFLINE FALLBACK' if result.degraded else 'live model'}"
    )

    print("\n  Proposed tasks (awaiting approval):")
    if not result.tasks:
        print("    (none survived validation)")
    for task in sorted(result.tasks, key=lambda t: t.time):
        print(f"    {task.time}  {task.description:<32} {task.pet:<10} [{task.frequency}]")
        if task.reason:
            print(f"            why: {task.reason}")
        if task.sources:
            print(f"            source: {', '.join(task.sources)}")

    if result.caveats:
        print("\n  Caveats:")
        for caveat in result.caveats:
            print(f"    - {caveat}")

    print("\n  Agent trace:")
    for line in result.trace.as_lines():
        print(f"    {line}")


def scenario_one(agent: CarePlanAgent) -> None:
    """Plan from scratch, then approve, showing the human-in-the-loop gate."""
    header("SCENARIO 1 — Plan a day for a puppy and a cat")

    owner = Owner("Jordan")
    biscuit = Pet("Biscuit", "dog")
    biscuit.add_task(Task("Vet checkup", "11:00", frequency="once"))
    owner.add_pet(biscuit)
    owner.add_pet(Pet("Mochi", "cat"))

    request = (
        "Biscuit is a 9-month-old puppy and Mochi is an adult indoor cat. "
        "I work 09:00 to 17:00 on weekdays. Plan a realistic daily routine "
        "that doesn't leave Biscuit alone too long."
    )
    print(f"\n  Owner: {owner.name}")
    print(f"  Pets:  {', '.join(f'{p.name} ({p.species})' for p in owner.pets)}")
    print(f"  Already scheduled: 11:00 Vet checkup (Biscuit)")
    print(f"\n  Request: {request}")

    result = agent.plan(owner, request)
    show_proposal(result)

    if result.ok:
        created = agent.commit(owner, result)
        print(f"\n  >>> Human approved. {len(created)} task(s) committed.\n")
        print(Scheduler(owner).todays_schedule())


def scenario_two(agent: CarePlanAgent) -> None:
    """Grounded question answering with citations."""
    header("SCENARIO 2 — Ask a grounded pet-care question")

    owner = Owner("Priya")
    owner.add_pet(Pet("Luna", "cat"))

    for question in [
        "How often should I brush a long-haired cat?",
        "My cat keeps waking me up at 4am. What can I change?",
    ]:
        print(f"\n  Q: {question}")
        answer = agent.answer(question, owner)
        print(f"\n  A: {answer.answer}")
        print(f"\n     Sources: {', '.join(answer.sources) or '(none)'}")
        print(f"     Confidence: {answer.confidence:.2f}")


def scenario_three(agent: CarePlanAgent) -> None:
    """Guardrails: an out-of-domain question, and a plan with no pets."""
    header("SCENARIO 3 — Guardrails")

    owner = Owner("Sam")

    print("\n  3a. Out-of-domain question (should refuse, not improvise)")
    print("      Q: How do I fix my car engine?")
    answer = agent.answer("How do I fix my car engine?", owner)
    print(f"\n      A: {answer.answer}")
    print(f"      Confidence: {answer.confidence:.2f}")

    print("\n  3b. Planning with no pets registered (should refuse cleanly)")
    result = agent.plan(owner, "Plan my day")
    print(f"      -> {result.error}")

    print("\n  3c. Empty request (should refuse cleanly)")
    owner.add_pet(Pet("Rex", "dog"))
    result = agent.plan(owner, "   ")
    print(f"      -> {result.error}")


def main() -> int:
    settings = load_settings()
    print(RULE)
    print("PawPal+ AI layer demo")
    print(RULE)
    print(f"Provider: {settings.describe()}")
    if not settings.is_live:
        print(
            "\nNOTE: running on the offline stub, so plans come from fixed rules\n"
            "and ignore the owner's constraints. Add GEMINI_API_KEY to .env for\n"
            "live model output."
        )

    agent = CarePlanAgent(settings=settings)
    scenario_one(agent)
    scenario_two(agent)
    scenario_three(agent)

    print(f"\n{RULE}\nDemo complete. Traces written to logs/traces.jsonl\n{RULE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
