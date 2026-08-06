"""Retrieval over the local pet-care knowledge base.

A deliberately dependency-free TF-IDF retriever. No embedding model, no vector
database, no network call — which means retrieval is fully deterministic and
unit-testable, and the repo installs from four small packages.

Documents are split on Markdown ``##`` headings so each chunk is one coherent
topic ("Puppy exercise limits", "Litter tray"), which keeps retrieved context
tight enough to fit comfortably in a prompt.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pawpal_ai.config import KNOWLEDGE_BASE_DIR
from pawpal_ai.logging_setup import get_logger

logger = get_logger("pawpal_ai.retriever")

# Words carrying no topical signal. Kept short on purpose: an aggressive list
# would strip meaningful pet-care terms.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can do does for from had has have how i if in into is it
    its me my no not of on or our should so than that the their them then there these they this
    to was we were what when where which who will with you your
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9']*")

# How many times a chunk's heading is repeated when indexing it. Tuned by hand
# against the queries in evaluation/eval_cases.json; see the retrieval notes in
# model_card.md for the before/after numbers.
_HEADING_WEIGHT = 3

# Minimum fraction of a query's words that must exist in the corpus vocabulary
# before we treat the query as in-domain at all. This is a domain gate, not a
# relevance threshold, and it exists because stemming creates false friends:
# the corpus phrase "a fixed daily event" stems to "fix", so "fix my car engine"
# would otherwise retrieve medication guidance off a single accidental match.
_MIN_QUERY_COVERAGE = 0.5


def _undouble(base: str) -> str:
    """Collapse the doubled consonant left by stripping a suffix: runn -> run."""
    if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "sl":
        return base[:-1]
    return base


def _normalize(token: str) -> str:
    """Fold inflected forms so query and document tokens meet in the middle.

    Handles plurals ("puppies" -> "puppy") and verb endings ("walking" -> "walk").
    The verb rule matters more than it looks: the corpus says "five minutes of
    formal walking" while owners ask "how often should I walk him", and without
    stemming those two never match.

    Linguistic correctness is not the goal — consistency is. "during" stems to
    "dur", which is wrong but harmless, because the query side stems identically.
    """
    if len(token) > 5 and token.endswith("ing"):
        base = _undouble(token[:-3])
        if len(base) >= 3:
            return base
    if len(token) > 4 and token.endswith("ed"):
        base = _undouble(token[:-2])
        if len(base) >= 3:
            return base
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("es") and token[-3] in "sxzh":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords, fold plurals."""
    return [
        _normalize(match)
        for match in _TOKEN_RE.findall(text.lower())
        if match not in _STOPWORDS and len(match) > 1
    ]


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage: a single ``##`` section of a source document."""

    source: str  # filename, e.g. "dog_care.md"
    heading: str  # e.g. "Daily exercise needs"
    text: str

    @property
    def citation(self) -> str:
        """Short label the model is told to cite, e.g. dog_care.md#daily-exercise-needs."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.heading.lower()).strip("-")
        return f"{self.source}#{slug}"


@dataclass
class RetrievalResult:
    """A chunk plus how well it matched the query."""

    chunk: Chunk
    score: float


def split_document(source: str, raw: str) -> List[Chunk]:
    """Split one Markdown document into chunks on its ``##`` headings.

    Text appearing before the first ``##`` (typically just the ``#`` title) is
    attached to nothing and dropped, since it carries no standalone content.
    """
    chunks: List[Chunk] = []
    heading: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        if heading and buffer:
            body = "\n".join(buffer).strip()
            if body:
                chunks.append(Chunk(source=source, heading=heading, text=body))

    for line in raw.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buffer = []
        elif heading is not None:
            buffer.append(line)

    flush()
    return chunks


def load_chunks(directory: Optional[Path] = None) -> List[Chunk]:
    """Load and chunk every Markdown file in the knowledge base directory.

    ``README.md`` is skipped: it documents the corpus rather than being part of it.
    """
    target = directory or KNOWLEDGE_BASE_DIR
    if not target.exists():
        logger.warning("Knowledge base directory not found: %s", target)
        return []

    chunks: List[Chunk] = []
    for path in sorted(target.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(split_document(path.name, path.read_text(encoding="utf-8")))

    logger.debug("Loaded %d chunks from %s", len(chunks), target)
    return chunks


class Retriever:
    """TF-IDF retriever with cosine similarity scoring."""

    def __init__(self, chunks: Optional[Sequence[Chunk]] = None) -> None:
        self.chunks: List[Chunk] = list(chunks) if chunks is not None else load_chunks()
        self._idf: Dict[str, float] = {}
        self._vectors: List[Dict[str, float]] = []
        self._build_index()

    # --- indexing ----------------------------------------------------------

    def _build_index(self) -> None:
        """Compute IDF across the corpus, then an L2-normalized vector per chunk."""
        if not self.chunks:
            logger.warning("Retriever built with an empty corpus; searches return nothing.")
            return

        # The heading is repeated so its terms carry more weight than body terms.
        # Without this, a long chunk's L2 norm dilutes every term it contains, and
        # a passage literally titled "Puppy exercise limits" loses a query about
        # walking a puppy to a longer passage that merely says "walk" a lot.
        tokenized = [
            tokenize(" ".join([c.heading] * _HEADING_WEIGHT) + " " + c.text)
            for c in self.chunks
        ]
        n_docs = len(tokenized)

        doc_freq: Counter = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))

        # Smoothed IDF, so a term appearing in every chunk still gets a small
        # positive weight rather than exactly zero.
        self._idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in doc_freq.items()
        }
        self._vectors = [self._vectorize(tokens) for tokens in tokenized]
        logger.debug("Indexed %d chunks, %d unique terms", n_docs, len(self._idf))

    def _vectorize(self, tokens: Iterable[str]) -> Dict[str, float]:
        """Turn tokens into an L2-normalized TF-IDF vector."""
        counts = Counter(tokens)
        if not counts:
            return {}

        # Sublinear TF: a term appearing 10 times is not 10x as important.
        vector = {
            term: (1.0 + math.log(count)) * self._idf.get(term, 0.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm == 0.0:
            return {}
        return {term: weight / norm for term, weight in vector.items()}

    # --- querying ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_relevance: float = 0.0,
    ) -> List[RetrievalResult]:
        """Return the top_k chunks most similar to the query, best first.

        Chunks scoring at or below ``min_relevance`` are dropped, so an
        off-topic query returns nothing rather than the least-bad match. That
        distinction is what lets the agent say "I have no grounding for this"
        instead of confidently citing an irrelevant document.
        """
        if not self._vectors:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        # Domain gate: if most of the query's words are absent from the corpus,
        # this is not a question the knowledge base can answer, and any score it
        # produces comes from an incidental term match rather than real relevance.
        known = [token for token in tokens if token in self._idf]
        coverage = len(known) / len(tokens)
        if not known or (len(tokens) >= 3 and coverage < _MIN_QUERY_COVERAGE):
            logger.debug(
                "Query %r rejected as out-of-domain (%d/%d terms known)",
                query[:60],
                len(known),
                len(tokens),
            )
            return []

        query_vector = self._vectorize(tokens)
        if not query_vector:
            return []

        scored: List[Tuple[float, int]] = []
        for index, chunk_vector in enumerate(self._vectors):
            # Both vectors are normalized, so the dot product is the cosine.
            # Iterate the shorter vector for speed.
            if len(query_vector) < len(chunk_vector):
                score = sum(w * chunk_vector.get(t, 0.0) for t, w in query_vector.items())
            else:
                score = sum(w * query_vector.get(t, 0.0) for t, w in chunk_vector.items())
            if score > min_relevance:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        results = [
            RetrievalResult(chunk=self.chunks[index], score=round(score, 4))
            for score, index in scored[:top_k]
        ]

        logger.debug(
            "Query %r matched %d/%d chunks; returning %d",
            query[:60],
            len(scored),
            len(self.chunks),
            len(results),
        )
        return results


def format_context(results: Sequence[RetrievalResult]) -> str:
    """Render retrieved chunks as a citable context block for the prompt.

    Each passage is labelled with the exact citation string the model is
    instructed to reuse, which is what makes citation checking possible later.
    """
    if not results:
        return "(no relevant guidance found in the knowledge base)"

    blocks = []
    for result in results:
        blocks.append(
            f"[{result.chunk.citation}] (relevance {result.score:.2f})\n"
            f"{result.chunk.heading}\n{result.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)
