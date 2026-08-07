# Model Card — PawPal+

Reflection and responsible-AI documentation for the PawPal+ applied AI system.

| | |
|---|---|
| **System** | PawPal+ — a RAG-grounded, self-checking pet-care scheduling agent |
| **Base project** | PawPal+ (Modules 1–3), a deterministic Streamlit pet-care planner |
| **Model** | Google Gemini via REST (`PAWPAL_MODEL`, default `gemini-3.6-flash`), swappable |
| **Retrieval** | Local TF-IDF over 5 Markdown documents / 31 passages. No external index |
| **Fallback** | Deterministic offline planner, used when no key is set or the API fails |
| **Human oversight** | Required. The agent proposes; a person approves before anything is scheduled |

## Intended use

Helping a pet owner turn a plain-English description of their situation into a concrete
daily care routine, and answering general pet-care routine questions grounded in a small
local corpus.

## Out of scope

Veterinary advice of any kind: diagnosis, symptom interpretation, dosages, or whether a
given medication is safe. The system is instructed to refuse these and to schedule a
reminder to contact a vet instead. It is also not a medication reminder you should rely
on — see "Could this be misused" below.

---

## What are the limitations or biases in your system?

**The corpus is the ceiling.** The system can only be as good as five documents I wrote
myself. They are general, widely-published rules of thumb, not clinical guidance from a
named veterinary authority. If the corpus is wrong, the agent will confidently schedule
around the wrong advice, because every guardrail I built checks *structure* — is this a
real pet, a valid time, a real citation — and none check whether the underlying advice is
medically sound.

**Species bias.** The corpus covers dogs and cats. A rabbit, bird, reptile, or horse gets
the `_STUB_DEFAULT` routine — two vague "care check" tasks — because no passage matches.
The UI cheerfully offers "other" as a species, which overpromises.

**Cultural and lifestyle bias.** The guidance assumes UK/US pet-care conventions: indoor
cats, walked dogs, a household with a garden or a safe street, and an owner whose day has
predictable structure. The example prompts assume a 9-to-5 office worker. An owner doing
shift work, living somewhere dogs are not walked on leads, or without reliable daytime
access to their home is served worse, and nothing in the system signals that.

**Lexical retrieval misses paraphrases.** TF-IDF matches words, not meaning. Measured, on
three phrasings of one question:

| Query | Correct passage retrieved? |
|---|---|
| "my cat keeps waking me at night" | Yes, rank 1 |
| "my cat keeps waking me up at four in the morning" | Yes, but rank 2 |
| "My cat keeps waking me up at 4am. What can I change?" | No |

The third fails partly because `4am` tokenizes to `am` — the token pattern requires a
leading letter and drops the digit, destroying the most informative word in the sentence.
Embeddings would fix this; they were rejected to keep retrieval free, offline and
deterministic. The tradeoff is documented rather than hidden.

**Conflict detection is exact-match only.** Inherited from the Module 1–3 `Scheduler`:
tasks are points in time, not intervals, so 08:00 and 08:15 never conflict even if the
first takes half an hour. The agent is told to space tasks 15 minutes apart, but that is a
prompt instruction, not an enforced rule.

**No persistence.** State lives in `st.session_state` and dies when the app restarts.

**Confidence is a heuristic, not a probability.** It starts from the model's own
self-report — which is not calibrated — and is discounted for repairs, missing grounding,
and degraded mode. A 0.72 does not mean 72% of such plans are correct. It is a relative
signal for comparing runs, and I would not want it read as more than that.

---

## Could your AI be misused, and how would you prevent that?

**Mistaken for veterinary advice.** The most likely real harm. Someone asks whether they
can give their dog ibuprofen, and a fluent, cited answer reads as authoritative.

*Mitigations in the code:* both system prompts forbid dosages, diagnosis and treatment and
require deferring to a vet; the corpus itself carries an explicit scope disclaimer;
`medication_and_safety.md` states plainly that human painkillers are toxic; and when
retrieval finds nothing, the agent refuses rather than improvising — verified by
`test_answer_refuses_when_nothing_is_retrieved` and eval cases R11/R12.

*Not mitigated:* nothing classifies a question as medical and hard-blocks it. A question
that happens to retrieve well would still be answered by the model, and my only defence is
prompt instructions, which are not a security boundary.

**Over-trusted for medication timing.** The system will happily schedule "Give arthritis
tablet, 08:00, daily". If someone relies on it as a medication reminder and it silently
drops a task, that is a real-world consequence for an animal.

*Mitigations:* the human approval gate means no medication task is ever added without a
person reading it. Confidence and sources are shown next to the plan, not buried. Degraded
runs are labelled with an explicit warning that constraints were ignored.

**Invented citations lending false authority.** A fabricated source like
`vet_journal_2024.md#dosage` would make output look better-sourced than it is. The
validator rejects any citation outside the retrieved set (eval case V8), and the model is
re-prompted rather than allowed to keep it.

**Automation creep.** The most likely way this system becomes harmful is a future
maintainer removing the approval step for convenience. It is load-bearing and is why
`plan()` and `commit()` are separate methods, with a test asserting `plan()` does not
mutate the owner.

---

## What surprised you while testing your AI's reliability?

**My own fix caused a regression, and only the eval caught it.** Adding verb stemming
fixed the puppy-walk query. It also made the corpus phrase *"a **fixed** daily event"*
stem to `fix`, so the deliberately absurd query *"how do I fix my car engine"* started
retrieving medication guidance at a respectable score. One accidental term match was
enough. That led to the domain gate, and to keeping out-of-domain cases in the eval suite
permanently — a retriever that always returns *something* looks confident and is worse.

**The offline fallback scores 25/25 on the eval, and that is a warning, not a win.** The
deterministic planner passes every planning case because those cases check *internal
consistency* — valid times, real pets, no double-bookings — and fixed routines are
trivially consistent. It scores well while completely ignoring the owner's request. Tests
that a dumb baseline passes are not measuring intelligence, and I would not have noticed
if I had not run the baseline deliberately.

**Ranking was more fragile than parsing.** I expected malformed JSON to be the hard
problem. It was not: fences, prose wrappers and JSON mode handle it in about thirty lines.
Retrieval ranking was where behaviour was genuinely unstable — three phrasings of the same
question gave three different top results.

**"Refuse" needed to be designed, not discovered.** My first retriever always returned its
top-k, so an off-topic query got the least-bad match and the model dutifully built an
answer on it. Returning an empty list had to become a deliberate, tested behaviour.

**Passing every offline test told me almost nothing about the live system.** The first real
API run failed outright — "No valid tasks survived validation", three attempts, all
reported as malformed JSON. The JSON was fine. Gemini 3.x spends output tokens on hidden
reasoning (~850 of my 2048-token budget), so the plan was cut off mid-object. My tests
could not have caught it: fake providers never run out of tokens.

Two lessons stuck. First, an entire category of failure lives only at the real network
boundary, and a suite that mocks that boundary is blind to it by construction. Second,
**the error message pointed at the wrong layer** — I was told "malformed JSON", so I looked
at the parser, when the fault was a configuration value three files away. The fix was not
just raising the budget; it was detecting `finishReason: MAX_TOKENS` and reporting
truncation as truncation, so the next person is not sent to the wrong place.

---

## Describe your collaboration with AI during this project

I used an AI coding assistant throughout: designing the layering, writing the retriever and
agent loop, and generating the test suite. My role was deciding what to build, what to
reject, and what to verify.

### One helpful suggestion

Before writing any client code, the assistant checked the current Gemini documentation
instead of writing from memory — and that changed the design. Two things surfaced that
would each have cost hours:

1. The official `google-genai` SDK now requires **Python ≥ 3.10**, and my default `python3`
   is **3.9.6**. Building against the SDK first would have produced a confusing install
   failure.
2. Google's own docs currently disagree with each other about the SDK's call signature —
   the quickstart shows an `interactions.create` API, the PyPI readme shows
   `models.generate_content`.

The resulting recommendation — call the documented REST endpoint with `requests` instead of
the SDK — is why the project has four small dependencies, runs on any Python from 3.9 up,
and is insulated from SDK churn. This was the single most valuable moment of the
collaboration, and it was valuable precisely because the assistant treated its own memory
as untrustworthy.

### One flawed suggestion

The first retriever it wrote normalized **plurals only**. That looked complete and passed a
casual read, but it silently mis-ranked queries: *"How often should I walk a puppy?"*
returned senior-dog guidance, because the corpus says *"walking"* and the query says
*"walk"*, and those never matched. Nothing failed loudly — I only found it because I
printed real retrieval scores for a handful of realistic questions instead of trusting that
the component worked.

The fix it then proposed introduced the `fix` / `"fixed daily event"` collision described
above. So the flawed suggestion was not one bad line but a pattern: **plausible code with
no measurement attached.** The assistant was fast and fluent at producing retrieval logic
and slow to volunteer that retrieval quality is an empirical question. What made the
difference was building `evaluate.py` early and re-running it after every change, which is
a judgement call the tooling did not make for me.

The same pattern produced the bug that broke the first live run. The assistant set
`maxOutputTokens: 2048` — a reasonable-looking default — without accounting for the fact
that the model it had just recommended spends output tokens on hidden reasoning. Every
offline test passed. The system failed on its first contact with the real API, and the
error it produced ("malformed JSON") pointed at the wrong layer entirely.

A smaller instance: it wrote a dynamic `__import__("pawpal_ai.schemas", fromlist=[...])`
inside the agent loop where a plain top-level import belonged — working code that would not
survive review.

### Takeaway

The assistant was strongest where the answer was verifiable — schema validation, error
handling, test scaffolding — and weakest where quality is a matter of measurement rather
than correctness. The parts of this system I trust most are the parts where I wrote a
failing case first.

---

## Testing summary

Full numbers in [`evaluation/results.md`](evaluation/results.md) and the README.

- **62/62 automated tests pass** (`python -m pytest`): 12 inherited from Modules 1–3, 50
  covering the AI layer.
- **25/25 evaluation cases pass** (`python evaluate.py`) against the live model
  (mean confidence 0.50) and against the offline baseline (0.36): retrieval 12/12,
  validation 10/10, planning 3/3.
- Failure paths are tested by injecting fake providers, because a real model cannot be
  asked to emit malformed JSON on demand — with the caveat, learned the hard way, that
  mocked providers cannot reproduce token-budget failures.
- Known-imperfect and deliberately left visible: retrieval ranking on paraphrased queries
  (Experiment 4 in [`ai_interactions.md`](ai_interactions.md)).
