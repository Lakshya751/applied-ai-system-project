"""Tests for the PawPal+ AI layer.

Every test here runs offline and deterministically. Nothing calls a real model:
the live provider is exercised through injected fakes, which is what lets the
agent's failure and repair paths be tested at all — you cannot reliably make a
real model emit malformed JSON on demand.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from pawpal_ai.agent import CarePlanAgent
from pawpal_ai.config import Settings
from pawpal_ai.providers import (
    LLMProvider,
    LLMResponse,
    ProviderError,
    StubProvider,
    build_provider,
)
from pawpal_ai.retriever import Chunk, Retriever, format_context, split_document, tokenize
from pawpal_ai.schemas import (
    PlanParseError,
    parse_plan,
    validate_plan,
)
from pawpal_system import Owner, Pet, Task


# --- fakes ------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    """Returns a fixed sequence of responses, so failure paths are reproducible."""

    name = "scripted"
    model = "scripted-v1"

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[str] = []

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResponse:
        self.calls.append(prompt)
        text = self.replies.pop(0) if self.replies else "{}"
        return LLMResponse(text=text, provider=self.name, model=self.model, latency_ms=1)


class ExplodingProvider(LLMProvider):
    """Always fails, to exercise the degradation path."""

    name = "exploding"
    model = "none"

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResponse:
        raise ProviderError("simulated outage")


def valid_plan_json(time: str = "07:30", pet: str = "Biscuit") -> str:
    return json.dumps(
        {
            "tasks": [
                {
                    "pet": pet,
                    "description": "Morning walk",
                    "time": time,
                    "frequency": "daily",
                    "reason": "Puppies need short frequent walks.",
                    "sources": [],
                }
            ],
            "summary": "A single morning walk.",
            "confidence": 0.9,
            "caveats": [],
        }
    )


@pytest.fixture
def owner() -> Owner:
    owner = Owner("Jordan")
    pet = Pet("Biscuit", "dog")
    pet.add_task(Task("Vet checkup", "11:00"))
    owner.add_pet(pet)
    return owner


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="stub", max_repair_attempts=2)


# --- retriever: tokenization and stemming -----------------------------------


def test_tokenizer_drops_stopwords_and_punctuation():
    assert tokenize("How often should I walk the dog?") == ["often", "walk", "dog"]


def test_stemmer_folds_plurals():
    assert tokenize("puppies")[0] == tokenize("puppy")[0]


def test_stemmer_folds_verb_endings():
    """'walking' must match 'walk' — the corpus and the user phrase it differently."""
    assert tokenize("walking")[0] == tokenize("walk")[0]
    assert tokenize("feeding")[0] == tokenize("feed")[0]


def test_stemmer_leaves_short_words_alone():
    assert tokenize("bed cat") == ["bed", "cat"]


# --- retriever: chunking ----------------------------------------------------


def test_split_document_splits_on_h2_headings():
    raw = "# Title\nintro text\n\n## First\nalpha\n\n## Second\nbeta\n"
    chunks = split_document("test.md", raw)
    assert [c.heading for c in chunks] == ["First", "Second"]
    # Text before the first ## is dropped: it has no heading to be retrieved by.
    assert "intro text" not in " ".join(c.text for c in chunks)


def test_chunk_citation_is_slugified():
    chunk = Chunk(source="dog_care.md", heading="Daily exercise needs", text="x")
    assert chunk.citation == "dog_care.md#daily-exercise-needs"


# --- retriever: search ------------------------------------------------------


def test_search_ranks_relevant_chunk_first():
    retriever = Retriever()
    hits = retriever.search("what should I feed my kitten", top_k=1)
    assert hits, "expected at least one hit"
    assert hits[0].chunk.source == "puppy_and_kitten_care.md"


def test_search_rejects_out_of_domain_query():
    """The domain gate must reject queries the corpus cannot answer."""
    retriever = Retriever()
    assert retriever.search("how do I fix my car engine") == []
    assert retriever.search("what is the capital of France") == []


def test_search_on_empty_corpus_returns_nothing():
    assert Retriever(chunks=[]).search("anything") == []


def test_format_context_reports_absence_explicitly():
    assert "no relevant guidance" in format_context([])


# --- schema parsing ---------------------------------------------------------


def test_parse_plan_strips_markdown_fences():
    assert parse_plan('```json\n{"tasks": []}\n```') == {"tasks": []}


def test_parse_plan_extracts_json_wrapped_in_prose():
    raw = 'Sure! Here is your plan:\n{"tasks": [], "confidence": 0.5}\nHope that helps.'
    assert parse_plan(raw)["confidence"] == 0.5


def test_parse_plan_rejects_empty_and_malformed():
    with pytest.raises(PlanParseError):
        parse_plan("")
    with pytest.raises(PlanParseError):
        parse_plan("no json here at all")
    with pytest.raises(PlanParseError):
        parse_plan('{"tasks": [oops}')


def test_parse_plan_rejects_non_object():
    with pytest.raises(PlanParseError):
        parse_plan("[1, 2, 3]")


# --- validation -------------------------------------------------------------


def test_validate_accepts_a_well_formed_plan():
    report = validate_plan(json.loads(valid_plan_json()), known_pets=["Biscuit"])
    assert report.is_valid
    assert report.tasks[0].description == "Morning walk"


def test_validate_rejects_unknown_pet():
    """A task for a pet that does not exist is a hallucination, not a typo."""
    plan = json.loads(valid_plan_json(pet="Rex"))
    report = validate_plan(plan, known_pets=["Biscuit"])
    assert not report.is_valid
    assert "unknown pet" in str(report.issues[0])


@pytest.mark.parametrize("bad_time", ["8:00", "08:60", "25:00", "8am", "", "0800"])
def test_validate_rejects_bad_time_formats(bad_time):
    """Zero-padded HH:MM is required — Scheduler.sort_by_time() sorts these as strings."""
    plan = json.loads(valid_plan_json(time=bad_time))
    report = validate_plan(plan, known_pets=["Biscuit"])
    assert not report.is_valid
    assert report.issues_by_kind("schema")


def test_validate_rejects_bad_frequency():
    plan = json.loads(valid_plan_json())
    plan["tasks"][0]["frequency"] = "hourly"
    report = validate_plan(plan, known_pets=["Biscuit"])
    assert not report.is_valid


def test_validate_rejects_overlong_description():
    plan = json.loads(valid_plan_json())
    plan["tasks"][0]["description"] = "x" * 200
    report = validate_plan(plan, known_pets=["Biscuit"])
    assert not report.is_valid


def test_validate_detects_conflict_with_existing_schedule():
    plan = json.loads(valid_plan_json(time="11:00"))
    report = validate_plan(plan, known_pets=["Biscuit"], existing_times=["11:00"])
    assert not report.is_valid
    assert report.issues_by_kind("conflict")


def test_validate_detects_conflict_within_the_same_plan():
    """The model must not double-book itself inside one response."""
    plan = json.loads(valid_plan_json())
    plan["tasks"].append(dict(plan["tasks"][0], description="Second walk"))
    report = validate_plan(plan, known_pets=["Biscuit"])
    assert report.issues_by_kind("conflict")


def test_validate_rejects_invented_citation():
    """Citing a document that was never retrieved is the RAG failure mode we care about."""
    plan = json.loads(valid_plan_json())
    plan["tasks"][0]["sources"] = ["made_up_doc.md#invented"]
    report = validate_plan(
        plan, known_pets=["Biscuit"], allowed_citations=["dog_care.md#daily-exercise-needs"]
    )
    assert not report.is_valid
    assert report.issues_by_kind("citation")


def test_validate_accepts_a_real_citation():
    plan = json.loads(valid_plan_json())
    plan["tasks"][0]["sources"] = ["dog_care.md#daily-exercise-needs"]
    report = validate_plan(
        plan, known_pets=["Biscuit"], allowed_citations=["dog_care.md#daily-exercise-needs"]
    )
    assert report.is_valid


def test_validate_rejects_empty_task_list():
    report = validate_plan({"tasks": []}, known_pets=["Biscuit"])
    assert not report.is_valid


def test_validate_rejects_non_list_tasks():
    report = validate_plan({"tasks": "walk the dog"}, known_pets=["Biscuit"])
    assert not report.is_valid


def test_validate_clamps_confidence_out_of_range():
    plan = json.loads(valid_plan_json())
    plan["confidence"] = 4.2
    assert validate_plan(plan, known_pets=["Biscuit"]).confidence == 1.0
    plan["confidence"] = "not a number"
    assert validate_plan(plan, known_pets=["Biscuit"]).confidence == 0.0


# --- providers --------------------------------------------------------------


def test_stub_provider_is_deterministic():
    prompt = "PETS:\n- Biscuit (dog)\n"
    first = StubProvider().generate(prompt, json_mode=True).text
    second = StubProvider().generate(prompt, json_mode=True).text
    assert first == second


def test_stub_provider_reads_pets_from_the_prompt():
    plan = json.loads(StubProvider().generate("PETS:\n- Mochi (cat)\n", json_mode=True).text)
    assert {t["pet"] for t in plan["tasks"]} == {"Mochi"}


def test_stub_provider_marks_itself_degraded():
    assert StubProvider().generate("PETS:\n- Biscuit (dog)\n").degraded is True


def test_build_provider_falls_back_when_no_key():
    """A missing key must degrade to the stub, not crash the app."""
    provider = build_provider(Settings(provider="gemini", api_key=None))
    assert isinstance(provider, StubProvider)


# --- agent: guardrails ------------------------------------------------------


def test_agent_refuses_to_plan_without_pets(settings):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.plan(Owner("Jordan"), "plan my day")
    assert not result.ok
    assert "No pets" in result.error


def test_agent_refuses_empty_request(settings, owner):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.plan(owner, "   ")
    assert not result.ok
    assert "Empty request" in result.error


def test_agent_refuses_to_commit_a_failed_result(settings, owner):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.plan(Owner("Nobody"), "plan my day")  # fails: no pets
    assert agent.commit(owner, result) == []


# --- agent: the loop --------------------------------------------------------


def test_agent_produces_a_proposal_without_committing(settings, owner):
    """plan() must never mutate the schedule — approval is a separate step."""
    before = len(owner.all_tasks())
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.plan(owner, "plan a day for my dog")
    assert result.tasks
    assert len(owner.all_tasks()) == before, "plan() must not mutate the owner"


def test_commit_applies_tasks_to_the_right_pet(settings, owner):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.plan(owner, "plan a day for my dog")
    created = agent.commit(owner, result)
    assert created
    assert all(task in owner.pets[0].tasks for task in created)
    assert result.committed


def test_agent_repairs_invalid_output_then_succeeds(settings, owner):
    """The self-check loop: bad plan -> validator rejects -> model corrects."""
    bad = json.dumps({"tasks": [{"pet": "Ghost", "description": "Walk",
                                 "time": "8am", "frequency": "hourly"}]})
    provider = ScriptedProvider([bad, valid_plan_json()])
    agent = CarePlanAgent(settings=settings, provider=provider)

    result = agent.plan(owner, "plan a walk")

    assert result.ok
    assert result.repairs == 1
    assert len(provider.calls) == 2
    # The repair prompt must actually tell the model what was wrong.
    assert "rejected by the validator" in provider.calls[1]


def test_agent_gives_up_after_the_repair_budget(owner):
    """Persistent garbage must not loop forever."""
    settings = Settings(provider="stub", max_repair_attempts=2)
    provider = ScriptedProvider(["not json", "still not json", "nope"])
    agent = CarePlanAgent(settings=settings, provider=provider)

    result = agent.plan(owner, "plan a walk")

    assert not result.ok
    assert result.repairs == 2
    assert len(provider.calls) == 3  # initial + 2 repairs


def test_agent_avoids_conflicting_with_existing_tasks(settings, owner):
    """A plan colliding with the 11:00 vet checkup is rejected, then repaired."""
    provider = ScriptedProvider([valid_plan_json(time="11:00"), valid_plan_json(time="07:30")])
    agent = CarePlanAgent(settings=settings, provider=provider)

    result = agent.plan(owner, "plan a walk")

    assert result.ok
    assert result.repairs == 1
    assert result.tasks[0].time == "07:30"


def test_agent_degrades_to_stub_when_provider_fails(settings, owner):
    """An outage must produce a usable degraded plan, not an exception."""
    agent = CarePlanAgent(settings=settings, provider=ExplodingProvider())
    result = agent.plan(owner, "plan a day for my dog")
    assert result.ok
    assert result.degraded
    assert result.confidence <= 0.3


# --- agent: confidence ------------------------------------------------------


def test_confidence_is_capped_in_degraded_mode(settings):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    assert agent._score_confidence(0.95, ["a"], repairs=0, degraded=True) <= 0.3


def test_confidence_drops_when_ungrounded(settings):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    grounded = agent._score_confidence(0.9, ["dog_care.md#x"], repairs=0, degraded=False)
    ungrounded = agent._score_confidence(0.9, [], repairs=0, degraded=False)
    assert ungrounded < grounded


def test_confidence_drops_with_each_repair(settings):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    clean = agent._score_confidence(0.9, ["x"], repairs=0, degraded=False)
    repaired = agent._score_confidence(0.9, ["x"], repairs=2, degraded=False)
    assert repaired < clean


# --- agent: grounded answering ----------------------------------------------


def test_answer_refuses_when_nothing_is_retrieved(settings):
    """No grounding must produce a refusal, not an improvised answer."""
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.answer("how do I fix my car engine")
    assert result.confidence == 0.0
    assert "knowledge base" in result.answer
    assert result.sources == []


def test_answer_returns_sources_when_grounded(settings):
    agent = CarePlanAgent(settings=settings, provider=StubProvider())
    result = agent.answer("how often should I brush a long haired cat")
    assert result.sources
    assert all("#" in source for source in result.sources)
