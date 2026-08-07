"""Configuration for the PawPal+ AI layer.

Every tunable lives here and is overridable from the environment (or a local
``.env`` file), so no secret or endpoint is hard-coded into the logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # python-dotenv is convenient but not required to run.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only hit when the extra isn't installed
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
LOG_DIR = PROJECT_ROOT / "logs"

# Gemini's REST surface. Pinned to the documented v1beta path rather than the
# Python SDK: the SDK requires Python >= 3.10 and its call signature has churned
# recently, while this endpoint is stable and works on any Python with requests.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back on malformed input."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    """Resolved runtime configuration for the AI layer."""

    provider: str = "gemini"
    model: str = "gemini-3.6-flash"
    api_key: Optional[str] = None

    # Generation controls. Temperature is deliberately low: this system asks the
    # model for structured schedule data, not prose, so determinism beats flair.
    temperature: float = 0.2
    # Generous, because Gemini 3.x is a thinking model and reasoning tokens are
    # billed against this same budget. At 2048 the plan itself was being cut off
    # mid-object after ~850 tokens of thinking, which surfaced as "malformed JSON".
    max_output_tokens: int = 8192
    # "low" spends no tokens on visible reasoning. This task is structured data
    # extraction against supplied context, not open-ended problem solving, so the
    # thinking budget bought latency (8.8s -> ~3s) rather than quality.
    thinking_level: str = "low"
    request_timeout: int = 60
    max_retries: int = 2

    # Retrieval controls.
    top_k: int = 3
    min_relevance: float = 0.05

    # Agent controls: how many times the agent may repair invalid output before
    # it gives up and falls back to a deterministic plan.
    max_repair_attempts: int = 2

    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR
    log_dir: Path = LOG_DIR

    @property
    def is_live(self) -> bool:
        """True when a real model will be called (stub provider needs no key)."""
        if self.provider == "stub":
            return False
        return bool(self.api_key)

    def describe(self) -> str:
        """Human-readable one-liner for logs and the UI status line."""
        if self.provider == "stub":
            return "stub provider (offline, deterministic)"
        if not self.api_key:
            return f"{self.provider}:{self.model} (NO API KEY - will fall back to stub)"
        return f"{self.provider}:{self.model}"


def load_settings() -> Settings:
    """Build Settings from the environment.

    ``PAWPAL_PROVIDER=stub`` forces the offline provider, which is how the test
    suite and the eval harness run without a network or a key.
    """
    provider = _env_str("PAWPAL_PROVIDER", "gemini").lower()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    return Settings(
        provider=provider,
        model=_env_str("PAWPAL_MODEL", "gemini-3.6-flash"),
        api_key=api_key.strip() if api_key else None,
        temperature=_env_float("PAWPAL_TEMPERATURE", 0.2),
        max_output_tokens=_env_int("PAWPAL_MAX_TOKENS", 8192),
        thinking_level=_env_str("PAWPAL_THINKING", "low").lower(),
        request_timeout=_env_int("PAWPAL_TIMEOUT", 60),
        max_retries=_env_int("PAWPAL_MAX_RETRIES", 2),
        top_k=_env_int("PAWPAL_TOP_K", 3),
        max_repair_attempts=_env_int("PAWPAL_MAX_REPAIRS", 2),
    )
