# AI Interactions Log

Reasoning traces from the PawPal+ agent, plus the retrieval experiments behind the
current tuning.

Every trace below is real output, not illustration. The agent writes one JSON record
per run to `logs/traces.jsonl` (gitignored, since it is a runtime artifact); the traces
quoted here were copied from that file and from stdout.

---

## Agent Workflow (SF7)

### What the agent does

`CarePlanAgent.plan()` is a five-step decision chain, not a single model call:

| Step | What happens | Where |
|------|--------------|-------|
| 1. RETRIEVE | Query the TF-IDF index; widen the query with the owner's pet species | `agent._retrieve` |
| 2. PLAN | Prompt the model with the retrieved passages, the current schedule and the request | `agent.plan` |
| 3. VALIDATE | Check schema, pet names, time format, conflicts and citations | `schemas.validate_plan` |
| 4. REPAIR | Feed the specific validation errors back and re-prompt, up to a bounded budget | `agent.plan` loop |
| 5. SCORE | Discount the model's self-reported confidence by observed signals | `agent._score_confidence` |

The agent then stops. Committing to the schedule requires a human pressing Approve.

### Trace A — a clean live run

Captured from `python demo_ai.py` against `gemini-3.6-flash`, with two pets and one task
already booked at 11:00.

```text
[start]    Planning for Jordan with 2 pet(s)
[retrieve] 3 passage(s) retrieved for query 'Biscuit is a 9-month-old puppy and Mochi is an adult indoor '
[model]    gemini:gemini-3.6-flash replied in 6410ms
[validate] 6 task(s) accepted, 0 issue(s)
[propose]  6 task(s) proposed for approval (confidence 0.85, repairs 0)
[commit]   6 task(s) added to the schedule
```

Note the ordering: `propose` and `commit` are separate events, with the human decision
between them.

The same request on the offline provider, for comparison — structurally identical, and
capped at 0.3 confidence because the fallback ignores the owner's constraints:

```text
[model]    stub:deterministic-rules-v1 replied in 0ms
[validate] 8 task(s) accepted, 0 issue(s)
[propose]  8 task(s) proposed for approval (confidence 0.3, repairs 0)
```

### Trace B — the repair loop under fault injection

A real model rarely produces exactly the failure you want to demonstrate, so this run
injects three scripted responses through the same `LLMProvider` interface the live
client implements: first a plan with four separate violations, then a plan that collides
with an existing appointment, then a valid one.

Reproduce it with the snippet in "How to regenerate these traces" below.

```text
[start]    Planning for Jordan with 1 pet(s)
[retrieve] 3 passage(s) retrieved for query 'Plan a morning walk for Biscuit. dog'
[model]    scripted:scripted-v1 replied in 1ms
[validate] 0 task(s) accepted, 1 issue(s)
[repair]   Attempt 1 rejected; asking the model to fix 1 issue(s)
[model]    scripted:scripted-v1 replied in 1ms
[validate] 0 task(s) accepted, 1 issue(s)
[repair]   Attempt 2 rejected; asking the model to fix 1 issue(s)
[model]    scripted:scripted-v1 replied in 1ms
[validate] 1 task(s) accepted, 0 issue(s)
[propose]  1 task(s) proposed for approval (confidence 0.54, repairs 2)

final: ok=True repairs=2 confidence=0.54 time=07:30
```

What each rejection caught:

1. **Attempt 1** — `{"pet": "Rex", "time": "8am", "frequency": "hourly", "sources": ["made_up.md#x"]}`.
   Rex is not a registered pet, `8am` is not zero-padded `HH:MM`, `hourly` is not a valid
   frequency, and `made_up.md#x` was never retrieved. Validation stops at the first fatal
   problem per task, so this reports one issue and the task is dropped entirely.
2. **Attempt 2** — a well-formed task at `11:00`, which collides with the vet appointment
   already on the schedule. Caught by the conflict check, not the schema check.
3. **Attempt 3** — valid, accepted at `07:30`.

Confidence fell from the model's self-reported `0.9` to `0.54`: two repairs cost 20% each,
and the surviving task cited nothing.

The repair prompt is not generic. It contains the exact validator messages:

```text
Your previous plan was rejected by the validator for these reasons:
- (conflict) task 1 (Morning walk) is at 11:00, which already has a task
  scheduled. Choose a different time.

Return a corrected plan as JSON in the same schema. Fix every issue listed.
Do not repeat the rejected values.
```

### Trace C — the real failure, found on first contact with the live API

Every offline test passed. The first live run failed three times in a row. Verbatim from
`logs/traces.jsonl`:

```text
[start]    Planning for Jordan with 2 pet(s)
[retrieve] 3 passage(s) retrieved
[model]    gemini:gemini-3.6-flash replied in 8877ms   (tokens=3039)
[validate] Unparseable output: No JSON object found in response:
           '{\n  "tasks": [\n    {\n      "pet": "Biscuit",\n
             "description": "Morning meal for Biscuit",\n      "time": "07:00",\n   '
[repair]   Attempt 1 rejected; asking the model to fix 1 issue(s)
[model]    gemini:gemini-3.6-flash replied in 8573ms   (tokens=3145)
[validate] Unparseable output: Malformed JSON: Expecting ',' delimiter: line 52 column 6
[repair]   Attempt 2 rejected; asking the model to fix 1 issue(s)
[model]    gemini:gemini-3.6-flash replied in 8648ms   (tokens=3114)
[validate] Unparseable output: Malformed JSON: Expecting ',' delimiter: line 42 column 6
[repair]   Repair budget exhausted
[abort]    No valid tasks survived validation.
```

The JSON was not malformed. Look at the first payload: it is *well-formed and simply stops*.
The output was truncated.

Probing the API directly explained why:

```text
finishReason : STOP
usage        : {"promptTokenCount": 27, "candidatesTokenCount": 481,
                "thoughtsTokenCount": 856, "totalTokenCount": 1364}
```

`thoughtsTokenCount: 856`. Gemini 3.x reasons before answering, and those tokens are billed
against the same `maxOutputTokens` budget as the answer. With the cap at 2048, hidden
reasoning consumed most of it and the plan was cut off mid-object.

Measuring the options:

| `thinkingConfig` | Result | Thought tokens |
|---|---|---|
| omitted | STOP | 752 |
| `thinkingLevel: low` | STOP | **0** |
| `thinkingBudget: 0` | **HTTP 400**, invalid argument | — |
| `thinkingBudget: 256` | STOP | 0 |

Three changes followed. `maxOutputTokens` 2048 → 8192. `thinkingLevel: low`, which also cut
latency from ~8.8s to ~6.4s — this task is structured extraction from supplied context, not
open-ended reasoning. And the change that mattered most: detect `finishReason: MAX_TOKENS`
and surface it as *truncation*, with its own repair instruction ("return a SHORTER plan"),
so the trace stops accusing the parser of a fault three files away.

`test_agent_treats_truncated_output_as_repairable` locks the behaviour in, and asserts the
trace contains the words "cut off" rather than a JSON error.

**The transferable lesson:** mocked providers cannot exhaust a token budget, so no amount
of offline testing would have found this. Some bugs live only at the real boundary.

### How to regenerate these traces

```bash
# Trace A
PAWPAL_PROVIDER=stub python demo_ai.py          # traces land in logs/traces.jsonl

# Trace B (fault injection through the provider interface)
python - <<'PY'
import json, sys; sys.path.insert(0, 'tests')
from test_ai_layer import ScriptedProvider, valid_plan_json
from pawpal_ai.agent import CarePlanAgent
from pawpal_ai.config import Settings
from pawpal_system import Owner, Pet, Task

owner = Owner('Jordan')
pet = Pet('Biscuit', 'dog'); pet.add_task(Task('Vet checkup', '11:00')); owner.add_pet(pet)

bad = json.dumps({'tasks': [{'pet': 'Rex', 'description': 'Walk', 'time': '8am',
                             'frequency': 'hourly', 'sources': ['made_up.md#x']}]})
agent = CarePlanAgent(settings=Settings(provider='stub', max_repair_attempts=2),
                      provider=ScriptedProvider([bad, valid_plan_json(time='11:00'),
                                                 valid_plan_json(time='07:30')]))
result = agent.plan(owner, 'Plan a morning walk for Biscuit.')
print('\n'.join(result.trace.as_lines()))
PY
```

---

## Retrieval tuning experiments

The retriever was tuned by measurement, not by feel. Each change below was kept or
reverted based on the eval cases in `evaluation/eval_cases.json`.

### Experiment 1 — verb stemming

**Problem.** Asking *"How often should I walk a puppy?"* returned senior-dog guidance
first, and the correct passage (`puppy_and_kitten_care.md#puppy-exercise-limits`) did not
appear at all. The corpus says *"five minutes of formal **walking** per month of age"*
while the owner says *"**walk**"*. The original normalizer folded plurals only, so those
two tokens never met. Senior-dog guidance won purely because it happens to say *"walks"*.

**Change.** Added `-ing` / `-ed` stripping with doubled-consonant collapse.

| | Before | After |
|---|---|---|
| Rank of the correct puppy passage | absent from top 2 | rank 2 |
| Score of the correct passage | 0.14 (wrong document) | 0.19 |

**Kept**, with a caveat: the stemmer is not linguistically correct. `during` stems to
`dur`. That is harmless only because the query side stems identically — consistency
matters more than correctness here.

### Experiment 2 — heading weighting

**Problem.** Long passages were penalised by L2 normalisation, so a passage literally
titled *"Puppy exercise limits"* lost to a longer one that merely said *"walk"* often.

**Change.** Repeat each chunk's heading three times when indexing.

| Query | Before | After |
|---|---|---|
| "what should I feed my kitten" | 0.43 | **0.56** |
| "how often should I walk a puppy" (correct passage) | 0.19 | **0.21** |

**Kept.**

### Experiment 3 — the regression that experiment 1 caused

**Problem.** After stemming, the out-of-domain query *"how do I fix my car engine"*
started retrieving medication guidance at 0.16. Cause: the corpus phrase *"a **fixed**
daily event"* stems to `fix`, which collides with the user's `fix`. One accidental term
match was enough to clear the relevance threshold.

**Change.** Added a domain gate: if fewer than half a query's words exist in the corpus
vocabulary at all, reject the query outright instead of scoring it. `fix my car engine`
has 1 of 3 known terms.

| | Before | After |
|---|---|---|
| Out-of-domain queries retrieving something | 3 of 3 leaked | 0 of 3 |
| In-domain queries still matching | 7 of 7 | 7 of 7 |

**Kept.** This is why `evaluate.py` includes out-of-domain cases (R11, R12) that assert
*nothing* is returned — a retriever that always answers is not a feature.

### Experiment 4 — the one that is still unsolved

Phrasing sensitivity remains. The same intent, worded three ways:

| Query | Top result |
|---|---|
| "my cat keeps waking me at night" | `cat_care.md#play-and-enrichment` ✅ |
| "my cat keeps waking me up at four in the morning" | `senior_pet_care.md#senior-dog-exercise` ❌ (correct passage at rank 2) |
| "My cat keeps waking me up at 4am. What can I change?" | `cat_care.md#litter-tray` ❌ (correct passage absent) |

Two causes. `4am` tokenizes to `am`, because the token pattern requires a leading letter
and silently discards the digit — so the most informative word in the sentence is
destroyed. And the query shares only the stem `wak` with the corpus phrase *"reduces
night-time waking"*; the strong signal, `night`, is simply not in the user's wording.

This is the honest ceiling of lexical retrieval: TF-IDF matches words, not meaning.
Embedding-based retrieval would handle all three phrasings. It was not adopted here
because it would add a heavyweight dependency or a second paid API to a project whose
retrieval must stay free, offline and deterministic — but the limitation is real and is
recorded in [`model_card.md`](model_card.md).

Partial mitigation: `top_k = 3` means the model usually still sees a relevant cat passage
even when ranking is imperfect.

---

## Prompt Comparison (SF11)

Not attempted. The comparison worth running is the current planner system prompt against
a minimal one, measured on the same eval cases, and that requires a live model to be
meaningful — the offline provider ignores prompts entirely by design.
