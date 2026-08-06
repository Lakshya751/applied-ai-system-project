# Knowledge Base

The retrieval corpus for the PawPal+ AI layer. Every file here is plain Markdown, split into
chunks on its `##` headings by [`pawpal_ai/retriever.py`](../pawpal_ai/retriever.py) and scored
with TF-IDF cosine similarity at query time.

| File | Covers |
|------|--------|
| `dog_care.md` | Adult dog exercise, feeding, toileting, enrichment, grooming |
| `cat_care.md` | Adult cat feeding, hydration, litter, play, grooming |
| `puppy_and_kitten_care.md` | Age-based toileting and exercise limits, socialisation, meal frequency |
| `senior_pet_care.md` | Senior thresholds, gentler routines, monitoring, medication adherence |
| `medication_and_safety.md` | Dose timing, interactions, toxic substances, emergency signs |

## Provenance and scope

These documents were written for this project as a compact, self-contained corpus of
widely-published general pet-care guidance. They are **not** veterinary advice, are not
sourced from a specific clinical authority, and deliberately favour commonly-cited rules of
thumb (for example "five minutes of walking per month of age") because those are the kind of
scheduling heuristics the planner needs.

This matters for interpreting the system's output: PawPal+ can only be as good as this corpus.
Its known gaps are documented in [`../model_card.md`](../model_card.md) — notably that the
corpus is dog- and cat-centric, uses UK/US pet-care conventions, and contains no breed-specific
medical guidance.

## Adding your own documents

Drop any `.md` file into this folder and it is picked up automatically on the next run — the
retriever globs the directory, so no registration step or re-index command is needed. Use `##`
headings to control chunk boundaries.
