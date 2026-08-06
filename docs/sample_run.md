# Sample Run — Reproducible Execution Evidence

Verbatim output of `python demo_ai.py`, captured on the offline stub provider
(no API key set). Re-run it yourself with:

```bash
PAWPAL_PROVIDER=stub python demo_ai.py
```

With a `GEMINI_API_KEY` in `.env`, the same command produces model-generated
plans that respond to the owner's stated constraints; see README.md.

## Transcript

```text
==========================================================================
PawPal+ AI layer demo
==========================================================================
Provider: stub provider (offline, deterministic)

NOTE: running on the offline stub, so plans come from fixed rules
and ignore the owner's constraints. Add GEMINI_API_KEY to .env for
live model output.

==========================================================================
SCENARIO 1 — Plan a day for a puppy and a cat
==========================================================================

  Owner: Jordan
  Pets:  Biscuit (dog), Mochi (cat)
  Already scheduled: 11:00 Vet checkup (Biscuit)

  Request: Biscuit is a 9-month-old puppy and Mochi is an adult indoor cat. I work 09:00 to 17:00 on weekdays. Plan a realistic daily routine that doesn't leave Biscuit alone too long.

  Summary: Offline fallback plan built from fixed species routines. It does not account for your stated constraints.
  Confidence: 0.30   Repairs: 0   Mode: OFFLINE FALLBACK

  Proposed tasks (awaiting approval):
    07:15  Morning meal                     Mochi      [daily]
            why: Standard cat routine from the local knowledge base.
    07:30  Morning walk                     Biscuit    [daily]
            why: Standard dog routine from the local knowledge base.
    08:00  Breakfast                        Biscuit    [daily]
            why: Standard dog routine from the local knowledge base.
    09:00  Scoop litter tray                Mochi      [daily]
            why: Standard cat routine from the local knowledge base.
    17:30  Evening walk                     Biscuit    [daily]
            why: Standard dog routine from the local knowledge base.
    18:30  Dinner                           Biscuit    [daily]
            why: Standard dog routine from the local knowledge base.
    19:30  Evening play session             Mochi      [daily]
            why: Standard cat routine from the local knowledge base.
    20:00  Evening meal                     Mochi      [daily]
            why: Standard cat routine from the local knowledge base.

  Caveats:
    - Generated without a language model, so the owner's specific constraints were not considered.

  Agent trace:
    [start] Planning for Jordan with 2 pet(s)
    [retrieve] 3 passage(s) retrieved for query 'Biscuit is a 9-month-old puppy and Mochi is an adult indoor '
    [model] stub:deterministic-rules-v1 replied in 0ms
    [validate] 8 task(s) accepted, 0 issue(s)
    [propose] 8 task(s) proposed for approval (confidence 0.3, repairs 0)

  >>> Human approved. 8 task(s) committed.

Today's Schedule:
  07:15  Morning meal (Mochi) [daily] [todo]
  07:30  Morning walk (Biscuit) [daily] [todo]
  08:00  Breakfast (Biscuit) [daily] [todo]
  09:00  Scoop litter tray (Mochi) [daily] [todo]
  11:00  Vet checkup (Biscuit) [once] [todo]
  17:30  Evening walk (Biscuit) [daily] [todo]
  18:30  Dinner (Biscuit) [daily] [todo]
  19:30  Evening play session (Mochi) [daily] [todo]
  20:00  Evening meal (Mochi) [daily] [todo]

==========================================================================
SCENARIO 2 — Ask a grounded pet-care question
==========================================================================

  Q: How often should I brush a long-haired cat?

  A: The offline planner is active, so this answer is not model-generated. It applies fixed species routines from the local knowledge base. Set GEMINI_API_KEY in .env for a real grounded answer.

     Sources: cat_care.md#grooming, dog_care.md#grooming, senior_pet_care.md#senior-dog-exercise
     Confidence: 0.00

  Q: My cat keeps waking me up at 4am. What can I change?

  A: The offline planner is active, so this answer is not model-generated. It applies fixed species routines from the local knowledge base. Set GEMINI_API_KEY in .env for a real grounded answer.

     Sources: cat_care.md#litter-tray, cat_care.md#water, cat_care.md#grooming
     Confidence: 0.00

==========================================================================
SCENARIO 3 — Guardrails
==========================================================================

  3a. Out-of-domain question (should refuse, not improvise)
      Q: How do I fix my car engine?

      A: I don't have guidance on that in my knowledge base, so I'd rather not guess. The knowledge base covers dog and cat routines, puppy and kitten care, senior pets, and medication timing. For anything medical, please ask your vet.
      Confidence: 0.00

  3b. Planning with no pets registered (should refuse cleanly)
      -> No pets registered. Add a pet before asking for a plan.

  3c. Empty request (should refuse cleanly)
      -> Empty request. Describe what you want scheduled.

==========================================================================
Demo complete. Traces written to logs/traces.jsonl
==========================================================================
```
