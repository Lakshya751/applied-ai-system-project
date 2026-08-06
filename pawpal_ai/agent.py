"""The PawPal+ care-planning agent.

A five-step loop over the existing scheduler:

    retrieve -> plan -> validate -> repair (loop) -> propose

The agent never writes to the schedule. ``plan()`` returns a *proposal*; a human
approves it and only then does ``commit()`` create real Task objects. That
boundary is deliberate — an AI that silently rewrites your pet's medication
schedule is a worse product than one that asks first.

Everything the agent does is recorded in a RunTrace, so any plan can be audited
after the fact: what was retrieved, what the model proposed, what the validator
rejected, and how many repairs it took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pawpal_ai.config import Settings, load_settings
from pawpal_ai.logging_setup import RunTrace, get_logger
from pawpal_ai.providers import (
    LLMProvider,
    ProviderError,
    StubProvider,
    build_provider,
)
from pawpal_ai.retriever import Retriever, format_context
from pawpal_ai.schemas import (
    PlanParseError,
    ProposedTask,
    ValidationIssue,
    ValidationReport,
    parse_plan,
    validate_plan,
)
from pawpal_system import Owner, Pet, Scheduler, Task

logger = get_logger("pawpal_ai.agent")


PLANNER_SYSTEM_PROMPT = """\
You are the scheduling engine inside PawPal+, a pet-care planner.

Your job is to turn an owner's request into concrete, scheduled care tasks.

Rules you must follow:
1. Use ONLY the pets listed under PETS. Never invent a pet.
2. Ground every task in the RETRIEVED GUIDANCE. In each task's "sources" array,
   list the exact citation labels (like "dog_care.md#daily-exercise-needs") of the
   passages you relied on. If no passage supports a task, use an empty array.
   NEVER cite a label that does not appear in the RETRIEVED GUIDANCE.
3. Times must be zero-padded 24-hour "HH:MM" (08:00, not 8:00 or 8am).
4. Never schedule two tasks at the same time, including against the CURRENT
   SCHEDULE. Space tasks at least 15 minutes apart.
5. Respect the owner's stated constraints (work hours, availability, the pet's age).
6. Prefer few, high-value tasks over many trivial ones. Maximum 12.
7. Set "confidence" honestly: high only when the guidance directly supports your
   plan and the owner's constraints were clear.
8. You are not a veterinarian. Never propose a dosage, a diagnosis, or a
   treatment. For anything medical, schedule a reminder to consult a vet and say
   so in "caveats".

Respond with JSON only, in exactly this shape:
{
  "tasks": [
    {
      "pet": "<name from PETS>",
      "description": "<short task name, under 80 characters>",
      "time": "HH:MM",
      "frequency": "once" | "daily" | "weekly",
      "reason": "<one sentence on why this task, at this time>",
      "sources": ["<citation label>"]
    }
  ],
  "summary": "<2-3 sentences explaining the shape of the plan>",
  "confidence": <number between 0 and 1>,
  "caveats": ["<anything the owner should verify>"]
}
"""

ANSWER_SYSTEM_PROMPT = """\
You are the pet-care assistant inside PawPal+.

Answer the owner's question using ONLY the RETRIEVED GUIDANCE below. Cite the
passages you use inline, like [dog_care.md#daily-exercise-needs].

If the guidance does not cover the question, say so plainly rather than guessing.
You are not a veterinarian: for symptoms, dosages, or diagnosis, tell the owner to
contact a vet.

Keep the answer under 150 words.
"""


@dataclass
class AgentResult:
    """The outcome of one agent run: a proposal, not yet applied."""

    request: str
    tasks: List[ProposedTask] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    caveats: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    repairs: int = 0
    degraded: bool = False
    error: Optional[str] = None
    trace: Optional[RunTrace] = None
    committed: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.tasks) and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "tasks": [t.to_dict() for t in self.tasks],
            "summary": self.summary,
            "confidence": self.confidence,
            "caveats": self.caveats,
            "sources": self.sources,
            "repairs": self.repairs,
            "degraded": self.degraded,
            "error": self.error,
            "committed": self.committed,
        }


@dataclass
class AnswerResult:
    """A grounded answer to a pet-care question."""

    question: str
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    degraded: bool = False
    error: Optional[str] = None
    trace: Optional[RunTrace] = None


class CarePlanAgent:
    """Plans pet-care tasks, grounded in the knowledge base and self-checked."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[LLMProvider] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.provider = provider or build_provider(self.settings)
        self.retriever = retriever if retriever is not None else Retriever()

    # --- prompt assembly ---------------------------------------------------

    @staticmethod
    def _describe_pets(owner: Owner) -> str:
        """Render pets in a fixed '- Name (species)' form.

        The offline stub provider parses exactly this shape, so the degraded path
        reads the same prompt the live model does.
        """
        if not owner.pets:
            return "(no pets registered)"
        return "\n".join(f"- {pet.name} ({pet.species})" for pet in owner.pets)

    @staticmethod
    def _describe_schedule(owner: Owner) -> str:
        pairs = Scheduler(owner).sort_by_time()
        if not pairs:
            return "(nothing scheduled yet)"
        return "\n".join(
            f"- {task.time} {task.description} ({pet.name}, {task.frequency})"
            for pet, task in pairs
        )

    @staticmethod
    def _existing_times(owner: Owner) -> List[str]:
        return [task.time for _pet, task in owner.all_tasks()]

    def _build_plan_prompt(self, owner: Owner, request: str, context: str) -> str:
        return (
            f"OWNER: {owner.name}\n\n"
            f"PETS:\n{self._describe_pets(owner)}\n\n"
            f"CURRENT SCHEDULE:\n{self._describe_schedule(owner)}\n\n"
            f"RETRIEVED GUIDANCE:\n{context}\n\n"
            f"REQUEST:\n{request}\n\n"
            "Produce the JSON plan now."
        )

    # --- retrieval ---------------------------------------------------------

    def _retrieve(self, owner: Owner, request: str, trace: RunTrace) -> Tuple[str, List[str], Dict[str, float]]:
        """Search the knowledge base, widening the query with pet species.

        Adding species to the query matters: "plan his day" alone retrieves
        nothing useful, while "plan his day dog" pulls the dog routines.
        """
        species = " ".join(sorted({pet.species for pet in owner.pets}))
        query = f"{request} {species}".strip()

        results = self.retriever.search(
            query,
            top_k=self.settings.top_k,
            min_relevance=self.settings.min_relevance,
        )
        context = format_context(results)
        citations = [r.chunk.citation for r in results]
        scores = {r.chunk.citation: r.score for r in results}

        trace.add(
            "retrieve",
            f"{len(results)} passage(s) retrieved for query {query[:60]!r}",
            citations=citations,
            scores=scores,
        )
        if not results:
            trace.add(
                "retrieve",
                "No passage cleared the relevance threshold; the plan will be ungrounded.",
            )
        return context, citations, scores

    # --- model call with degradation ---------------------------------------

    def _call_model(
        self,
        prompt: str,
        system: str,
        trace: RunTrace,
        json_mode: bool,
    ) -> Tuple[str, bool]:
        """Call the provider, degrading to the offline stub on failure.

        Returns (text, degraded).
        """
        try:
            response = self.provider.generate(prompt, system=system, json_mode=json_mode)
            trace.add(
                "model",
                f"{response.provider}:{response.model} replied in {response.latency_ms}ms",
                tokens=response.token_total,
                degraded=response.degraded,
            )
            return response.text, response.degraded
        except ProviderError as exc:
            logger.error("Provider failed (%s): %s", type(exc).__name__, exc)
            trace.add("model", f"Provider failed: {type(exc).__name__}: {exc}")

            if isinstance(self.provider, StubProvider):
                raise  # already degraded; nothing left to fall back to

            trace.add("fallback", "Falling back to the offline deterministic planner.")
            response = StubProvider().generate(prompt, system=system, json_mode=json_mode)
            return response.text, True

    # --- confidence --------------------------------------------------------

    def _score_confidence(
        self,
        reported: float,
        sources: Sequence[str],
        repairs: int,
        degraded: bool,
    ) -> float:
        """Temper the model's self-reported confidence with observed signals.

        A model's own confidence is not trustworthy on its own, so it is
        discounted by things we can actually measure: whether the plan was
        grounded in retrieved text, how many times it failed validation, and
        whether a real model produced it at all.
        """
        if degraded:
            return round(min(reported, 0.3), 2)

        score = reported
        if not sources:
            score *= 0.6  # ungrounded: nothing in the corpus supported this
        score *= max(0.5, 1.0 - 0.2 * repairs)  # each repair costs 20%, floored
        return round(max(0.0, min(1.0, score)), 2)

    # --- the loop ----------------------------------------------------------

    def plan(self, owner: Owner, request: str) -> AgentResult:
        """Run the full agent loop and return a proposal for human approval."""
        trace = RunTrace(request=request, provider=self.provider.name)
        result = AgentResult(request=request, trace=trace)

        if not owner.pets:
            result.error = "No pets registered. Add a pet before asking for a plan."
            trace.add("guard", result.error)
            trace.persist()
            return result

        if not request.strip():
            result.error = "Empty request. Describe what you want scheduled."
            trace.add("guard", result.error)
            trace.persist()
            return result

        trace.add("start", f"Planning for {owner.name} with {len(owner.pets)} pet(s)")

        # 1. RETRIEVE
        context, citations, scores = self._retrieve(owner, request, trace)
        result.retrieval_scores = scores

        # 2. PLAN
        prompt = self._build_plan_prompt(owner, request, context)
        known_pets = [pet.name for pet in owner.pets]
        existing_times = self._existing_times(owner)

        report: Optional[ValidationReport] = None
        degraded = False

        for attempt in range(self.settings.max_repair_attempts + 1):
            try:
                raw, degraded_now = self._call_model(
                    prompt, PLANNER_SYSTEM_PROMPT, trace, json_mode=True
                )
            except ProviderError as exc:
                result.error = f"All providers failed: {exc}"
                trace.add("abort", result.error)
                trace.persist()
                return result

            degraded = degraded or degraded_now

            # 3. VALIDATE
            try:
                parsed = parse_plan(raw)
            except PlanParseError as exc:
                trace.add("validate", f"Unparseable output: {exc}")
                report = ValidationReport()
                report.issues.append(
                    ValidationIssue("schema", f"Response was not valid JSON: {exc}")
                )
            else:
                # The stub cites nothing, so citation checking would reject it for
                # the wrong reason; only enforce the whitelist on live output.
                report = validate_plan(
                    parsed,
                    known_pets=known_pets,
                    existing_times=existing_times,
                    allowed_citations=None if degraded_now else citations,
                )
                trace.add(
                    "validate",
                    f"{len(report.tasks)} task(s) accepted, {len(report.issues)} issue(s)",
                    issues=[str(i) for i in report.issues],
                )

            if report.is_valid:
                break

            # 4. REPAIR
            if attempt < self.settings.max_repair_attempts:
                result.repairs += 1
                trace.add(
                    "repair",
                    f"Attempt {attempt + 1} rejected; asking the model to fix "
                    f"{len(report.issues)} issue(s)",
                )
                prompt = self._build_plan_prompt(owner, request, context) + (
                    "\n\n" + report.repair_instruction()
                )
            else:
                trace.add(
                    "repair",
                    "Repair budget exhausted; keeping whatever tasks passed validation.",
                )

        # 5. PROPOSE
        assert report is not None  # loop always assigns
        result.tasks = report.tasks
        result.summary = report.summary
        result.caveats = list(report.caveats)
        result.sources = sorted({s for task in report.tasks for s in task.sources}) or citations
        result.degraded = degraded
        result.confidence = self._score_confidence(
            report.confidence, result.sources, result.repairs, degraded
        )

        if not result.tasks:
            result.error = "No valid tasks survived validation."
            trace.add("abort", result.error)
        else:
            trace.add(
                "propose",
                f"{len(result.tasks)} task(s) proposed for approval "
                f"(confidence {result.confidence}, repairs {result.repairs})",
            )

        trace.persist()
        return result

    # --- commit ------------------------------------------------------------

    def commit(self, owner: Owner, result: AgentResult) -> List[Task]:
        """Apply an approved proposal to the real schedule.

        This is the only place the AI layer mutates owner state, and it runs only
        after a human approves. Tasks are created through the existing
        ``Pet.add_task``, so the logic layer stays the single source of truth.
        """
        if not result.ok:
            logger.warning("Refusing to commit a result that is not ok.")
            return []

        by_name = {pet.name.lower(): pet for pet in owner.pets}
        created: List[Task] = []

        for proposed in result.tasks:
            pet: Optional[Pet] = by_name.get(proposed.pet.lower())
            if pet is None:
                # Validation should have caught this; belt and braces.
                logger.error("Skipping task for unknown pet %r at commit time.", proposed.pet)
                continue
            task = Task(
                description=proposed.description,
                time=proposed.time,
                frequency=proposed.frequency,
            )
            pet.add_task(task)
            created.append(task)

        result.committed = True
        logger.info("Committed %d task(s) to %s's schedule.", len(created), owner.name)
        if result.trace:
            result.trace.add("commit", f"{len(created)} task(s) added to the schedule")
        return created

    # --- grounded question answering ---------------------------------------

    def answer(self, question: str, owner: Optional[Owner] = None) -> AnswerResult:
        """Answer a pet-care question strictly from retrieved guidance."""
        trace = RunTrace(request=question, provider=self.provider.name)
        result = AnswerResult(question=question, trace=trace)

        if not question.strip():
            result.error = "Empty question."
            trace.add("guard", result.error)
            return result

        species = ""
        if owner and owner.pets:
            species = " ".join(sorted({pet.species for pet in owner.pets}))

        results = self.retriever.search(
            f"{question} {species}".strip(),
            top_k=self.settings.top_k,
            min_relevance=self.settings.min_relevance,
        )
        trace.add("retrieve", f"{len(results)} passage(s) retrieved")

        if not results:
            # Refusing to answer is the correct behaviour here: without grounding
            # the model would be free-associating about someone's animal.
            result.answer = (
                "I don't have guidance on that in my knowledge base, so I'd rather not "
                "guess. The knowledge base covers dog and cat routines, puppy and kitten "
                "care, senior pets, and medication timing. For anything medical, please "
                "ask your vet."
            )
            result.confidence = 0.0
            trace.add("refuse", "No grounding available; declined to answer.")
            trace.persist()
            return result

        context = format_context(results)
        prompt = f"RETRIEVED GUIDANCE:\n{context}\n\nQUESTION:\n{question}"

        try:
            text, degraded = self._call_model(prompt, ANSWER_SYSTEM_PROMPT, trace, json_mode=False)
        except ProviderError as exc:
            result.error = f"Could not answer: {exc}"
            trace.add("abort", result.error)
            trace.persist()
            return result

        result.answer = text
        result.degraded = degraded
        result.sources = [r.chunk.citation for r in results]
        # Retrieval strength is a reasonable proxy for how well-grounded the answer is.
        best = max(r.score for r in results)
        result.confidence = 0.0 if degraded else round(min(1.0, best * 2.5), 2)
        trace.add("answer", f"Answered from {len(results)} passage(s)")
        trace.persist()
        return result
