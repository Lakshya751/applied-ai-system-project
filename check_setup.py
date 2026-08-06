"""Verify a PawPal+ install before you rely on it.

Run this first:  python check_setup.py

Checks dependencies, configuration, the knowledge base index, and — if a key is
present — makes one real API call and lists the models the key can reach. This
exists because the most common failure for someone cloning this repo is a missing
or misnamed model ID, and a 404 buried in a Streamlit stack trace is a bad way to
discover that.
"""

from __future__ import annotations

import sys

from pawpal_ai.config import load_settings

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def check_dependencies() -> bool:
    """Confirm the third-party imports the app needs are installed."""
    print("Dependencies")
    ok = True
    for module, why in [
        ("requests", "calling the Gemini REST API"),
        ("streamlit", "the web UI"),
        ("pytest", "the test suite"),
    ]:
        try:
            __import__(module)
            print(f"{PASS} {module} available ({why})")
        except ImportError:
            print(f"{FAIL} {module} missing - run: pip install -r requirements.txt")
            ok = False

    try:
        __import__("dotenv")
        print(f"{PASS} python-dotenv available (.env loading)")
    except ImportError:
        print(f"{WARN} python-dotenv missing - .env will not load automatically")
    return ok


def check_knowledge_base() -> bool:
    """Confirm the corpus loads, indexes, and answers a known query."""
    print("\nKnowledge base")
    from pawpal_ai.retriever import Retriever

    retriever = Retriever()
    if not retriever.chunks:
        print(f"{FAIL} No passages indexed - is knowledge_base/ present?")
        return False

    sources = sorted({chunk.source for chunk in retriever.chunks})
    print(f"{PASS} {len(retriever.chunks)} passages from {len(sources)} documents")
    print(f"         {', '.join(sources)}")

    hits = retriever.search("how often should I feed a kitten", top_k=1)
    if hits:
        print(f"{PASS} Sample query resolved to {hits[0].chunk.citation} ({hits[0].score:.2f})")
    else:
        print(f"{FAIL} Sample query returned nothing - the index looks broken")
        return False

    off_topic = retriever.search("how do I fix my car engine", top_k=1)
    if off_topic:
        print(f"{WARN} Off-topic query matched {off_topic[0].chunk.citation}; domain gate is loose")
    else:
        print(f"{PASS} Off-topic query correctly rejected")
    return True


def check_model(settings) -> bool:
    """Confirm the configured model is reachable, or explain the offline path."""
    print("\nModel provider")
    print(f"         Configured: {settings.describe()}")

    if settings.provider == "stub":
        print(f"{PASS} Stub provider selected; no API access needed")
        return True

    if not settings.api_key:
        print(f"{WARN} No GEMINI_API_KEY found - the app will run on the offline stub")
        print("         Get a free key at https://aistudio.google.com (no card required),")
        print("         then: cp .env.example .env  and paste it into .env")
        return True  # not a failure: the system is designed to degrade

    from pawpal_ai.providers import GeminiProvider, ProviderError

    try:
        models = GeminiProvider.list_models(settings)
    except ProviderError as exc:
        print(f"{FAIL} Could not list models: {exc}")
        return False

    print(f"{PASS} Key is valid; {len(models)} models support generateContent")
    if settings.model in models:
        print(f"{PASS} Configured model '{settings.model}' is available")
    else:
        print(f"{FAIL} Configured model '{settings.model}' is NOT available to this key.")
        print("         Set PAWPAL_MODEL in .env to one of:")
        for name in models[:12]:
            print(f"           - {name}")
        return False

    try:
        provider = GeminiProvider(settings)
        response = provider.generate("Reply with the single word: ready", system="Be terse.")
        print(f"{PASS} Live call succeeded in {response.latency_ms}ms: {response.text[:40]!r}")
    except ProviderError as exc:
        print(f"{FAIL} Live call failed: {exc}")
        return False
    return True


def main() -> int:
    print("=" * 68)
    print("PawPal+ setup check")
    print("=" * 68)
    print(f"Python {sys.version.split()[0]}")

    settings = load_settings()
    results = [
        check_dependencies(),
        check_knowledge_base(),
        check_model(settings),
    ]

    print("\n" + "=" * 68)
    if all(results):
        print("All checks passed. Run:  streamlit run app.py")
        return 0
    print("Some checks failed. Fix the [FAIL] lines above and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
