# Sample Run — Reproducible Execution Evidence

Verbatim output of `python demo_ai.py`, captured against the live model
(`gemini-3.6-flash`, free tier). Reproduce with:

```bash
cp .env.example .env      # add your free key from https://aistudio.google.com
python check_setup.py     # verifies the key and lists reachable models
python demo_ai.py
```

Every run differs slightly — the model is sampled at temperature 0.2, not zero.
For a byte-identical run, force the offline planner with `PAWPAL_PROVIDER=stub`.

## Transcript

```text
==========================================================================
PawPal+ AI layer demo
==========================================================================
Provider: gemini:gemini-3.6-flash

==========================================================================
SCENARIO 1 — Plan a day for a puppy and a cat
==========================================================================

  Owner: Jordan
  Pets:  Biscuit (dog), Mochi (cat)
  Already scheduled: 11:00 Vet checkup (Biscuit)

  Request: Biscuit is a 9-month-old puppy and Mochi is an adult indoor cat. I work 09:00 to 17:00 on weekdays. Plan a realistic daily routine that doesn't leave Biscuit alone too long.

  Summary: This plan structures daily feeding and exercise routines around your 09:00 to 17:00 work schedule. Biscuit receives twice-daily meals and walks suited for a 9-month-old, while Mochi is fed in parallel morning and evening. The plan avoids conflict with the scheduled 11:00 vet checkup.
  Confidence: 0.85   Repairs: 0   Mode: live model

  Proposed tasks (awaiting approval):
    07:30  Morning walk and toilet break    Biscuit    [daily]
            why: Provides morning exercise and a toilet break before the work day begins.
            source: puppy_and_kitten_care.md#puppy-exercise-limits, puppy_and_kitten_care.md#puppy-toileting-frequency
    08:00  Morning meal                     Biscuit    [daily]
            why: Provides the first of two daily meals appropriate for a 9-month-old dog.
            source: puppy_and_kitten_care.md#puppy-feeding
    08:15  Morning meal and fresh water     Mochi      [daily]
            why: Ensures Mochi is fed and has fresh water before work hours.
    17:30  Evening walk and toilet break    Biscuit    [daily]
            why: Gives Biscuit exercise and relief immediately following your workday.
            source: puppy_and_kitten_care.md#puppy-exercise-limits, puppy_and_kitten_care.md#puppy-toileting-frequency
    18:00  Evening meal                     Biscuit    [daily]
            why: Provides the second daily meal for Biscuit.
            source: puppy_and_kitten_care.md#puppy-feeding
    18:30  Evening meal and play session    Mochi      [daily]
            why: Provides Mochi's evening food and indoor enrichment.

  Caveats:
    - Verify whether a mid-day dog walker or drop-in visit is needed for Biscuit while you are at work from 09:00 to 17:00.
    - Consult your vet during the 11:00 checkup regarding exact daily caloric needs and exercise tolerances for Biscuit as growth plates continue closing.

  Agent trace:
    [start] Planning for Jordan with 2 pet(s)
    [retrieve] 3 passage(s) retrieved for query 'Biscuit is a 9-month-old puppy and Mochi is an adult indoor '
    [model] gemini:gemini-3.6-flash replied in 6410ms
    [validate] 6 task(s) accepted, 0 issue(s)
    [propose] 6 task(s) proposed for approval (confidence 0.85, repairs 0)

  >>> Human approved. 6 task(s) committed.

Today's Schedule:
  07:30  Morning walk and toilet break (Biscuit) [daily] [todo]
  08:00  Morning meal (Biscuit) [daily] [todo]
  08:15  Morning meal and fresh water (Mochi) [daily] [todo]
  11:00  Vet checkup (Biscuit) [once] [todo]
  17:30  Evening walk and toilet break (Biscuit) [daily] [todo]
  18:00  Evening meal (Biscuit) [daily] [todo]
  18:30  Evening meal and play session (Mochi) [daily] [todo]

==========================================================================
SCENARIO 2 — Ask a grounded pet-care question
==========================================================================

  Q: How often should I brush a long-haired cat?

  A: Long-haired cats, such as Persians and Maine Coons, need daily brushing to prevent painful matting [cat_care.md#grooming].

     Sources: cat_care.md#grooming, dog_care.md#grooming, senior_pet_care.md#senior-dog-exercise
     Confidence: 1.00

  Q: My cat keeps waking me up at 4am. What can I change?

  A: I am sorry, but the provided guidance does not cover information on how to address a cat waking you up early in the morning or changes to make for early-morning behavior. 

If this is a new or sudden change in your cat's behavior, please contact a veterinarian to rule out any underlying medical conditions or discomfort.

     Sources: cat_care.md#litter-tray, cat_care.md#water, cat_care.md#grooming
     Confidence: 0.49

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
