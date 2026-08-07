"""Model providers behind one small interface.

The agent never talks to a vendor SDK directly. It talks to ``LLMProvider``,
which has two implementations:

* ``GeminiProvider`` — the real thing, over Google's documented REST endpoint.
* ``StubProvider``   — a deterministic offline planner used by the test suite,
  by the eval harness, and as the automatic fallback when the live call fails.

Because the fallback *is* the stub provider, the degraded path is exercised by
every test run rather than being untested emergency code.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pawpal_ai.config import GEMINI_API_BASE, Settings
from pawpal_ai.logging_setup import get_logger

logger = get_logger("pawpal_ai.providers")


# --- errors -----------------------------------------------------------------


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class AuthError(ProviderError):
    """Bad or missing credentials. Never worth retrying."""


class RateLimitError(ProviderError):
    """Quota exhausted or too many requests. Retryable with backoff."""


class TransientError(ProviderError):
    """Timeout, connection failure, or 5xx. Retryable."""


class ContentBlockedError(ProviderError):
    """The model returned no candidate, typically a safety filter."""


# --- response ---------------------------------------------------------------


@dataclass
class LLMResponse:
    """One model reply, plus the metadata the trace and logs care about."""

    text: str
    provider: str
    model: str
    latency_ms: int
    usage: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False  # True when this came from the fallback, not the live model
    truncated: bool = False  # True when the model hit the output cap mid-answer

    @property
    def token_total(self) -> int:
        return int(self.usage.get("totalTokenCount", 0) or 0)


class LLMProvider(ABC):
    """Minimal contract the agent depends on."""

    name: str = "abstract"
    model: str = "none"

    @abstractmethod
    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResponse:
        """Return the model's reply to ``prompt`` under ``system`` instructions."""


# --- Gemini -----------------------------------------------------------------


class GeminiProvider(LLMProvider):
    """Google Gemini via the v1beta generateContent REST endpoint.

    Calling REST rather than the ``google-genai`` SDK is deliberate: the SDK
    requires Python >= 3.10 and its call signature changed recently, while this
    endpoint is stable and needs only ``requests``.
    """

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise AuthError("GEMINI_API_KEY is not set; cannot build GeminiProvider.")
        self.settings = settings
        self.model = settings.model

    def _endpoint(self) -> str:
        return f"{GEMINI_API_BASE}/models/{self.model}:generateContent"

    def _payload(self, prompt: str, system: str, json_mode: bool) -> Dict[str, Any]:
        generation_config: Dict[str, Any] = {
            "temperature": self.settings.temperature,
            "maxOutputTokens": self.settings.max_output_tokens,
        }
        if json_mode:
            # Ask the API itself to guarantee JSON. We still validate the result:
            # a syntactically valid JSON document can still be semantically wrong.
            generation_config["responseMimeType"] = "application/json"

        # Gemini 3.x spends output tokens on hidden reasoning before it answers.
        # Left unbounded it consumed ~850 tokens per call and truncated the plan.
        if self.settings.thinking_level:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": self.settings.thinking_level
            }

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResponse:
        """Call the API with bounded retries on transient failures."""
        import requests  # imported here so the package imports without requests installed

        last_error: Optional[ProviderError] = None

        for attempt in range(self.settings.max_retries + 1):
            started = time.monotonic()
            try:
                response = requests.post(
                    self._endpoint(),
                    headers={
                        "Content-Type": "application/json",
                        # Header auth, not ?key= in the URL, so the key cannot
                        # leak into logs, proxies, or error messages.
                        "x-goog-api-key": self.settings.api_key or "",
                    },
                    json=self._payload(prompt, system, json_mode),
                    timeout=self.settings.request_timeout,
                )
            except requests.Timeout as exc:
                last_error = TransientError(f"Request timed out after {self.settings.request_timeout}s")
                logger.warning("Attempt %d timed out: %s", attempt + 1, exc)
            except requests.RequestException as exc:
                last_error = TransientError(f"Network error: {exc}")
                logger.warning("Attempt %d network error: %s", attempt + 1, exc)
            else:
                latency_ms = int((time.monotonic() - started) * 1000)
                try:
                    return self._parse(response, latency_ms)
                except (RateLimitError, TransientError) as exc:
                    last_error = exc
                    logger.warning("Attempt %d retryable failure: %s", attempt + 1, exc)
                # AuthError / ContentBlockedError / ProviderError propagate: retrying
                # a bad key or a blocked prompt just burns quota.

            if attempt < self.settings.max_retries:
                backoff = 2.0 ** attempt
                logger.info("Retrying in %.1fs", backoff)
                time.sleep(backoff)

        raise last_error or TransientError("Request failed for an unknown reason")

    def _parse(self, response: Any, latency_ms: int) -> LLMResponse:
        """Turn an HTTP response into an LLMResponse or a typed error."""
        status = response.status_code

        if status in (401, 403):
            raise AuthError(f"Authentication failed ({status}). Check GEMINI_API_KEY.")
        if status == 429:
            raise RateLimitError("Rate limit or quota exceeded (429).")
        if status >= 500:
            raise TransientError(f"Server error ({status}).")
        if status == 404:
            raise ProviderError(
                f"Model '{self.model}' not found (404). "
                "Run `python check_setup.py` to list models your key supports."
            )
        if status != 200:
            detail = self._error_detail(response)
            raise ProviderError(f"Request failed ({status}): {detail}")

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"Response was not valid JSON: {exc}") from exc

        candidates = body.get("candidates") or []
        if not candidates:
            feedback = body.get("promptFeedback", {})
            raise ContentBlockedError(f"No candidates returned. promptFeedback={feedback}")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        finish_reason = candidate.get("finishReason", "unknown")

        if not text:
            raise ContentBlockedError(f"Empty response (finishReason={finish_reason}).")

        usage = body.get("usageMetadata", {}) or {}

        # Report truncation as truncation. Without this the caller sees a JSON
        # object with no closing brace and reports "malformed JSON", which sends
        # you looking for a parser bug instead of an output-budget problem.
        truncated = finish_reason == "MAX_TOKENS"
        if truncated:
            logger.warning(
                "Response hit the output cap (maxOutputTokens=%d, thoughts=%s). "
                "The answer is incomplete.",
                self.settings.max_output_tokens,
                usage.get("thoughtsTokenCount", "?"),
            )

        logger.info(
            "gemini call ok: model=%s latency=%dms tokens=%s thoughts=%s finish=%s",
            self.model,
            latency_ms,
            usage.get("totalTokenCount", "?"),
            usage.get("thoughtsTokenCount", 0),
            finish_reason,
        )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=body.get("modelVersion", self.model),
            latency_ms=latency_ms,
            usage=usage,
            truncated=truncated,
        )

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            return str(response.json().get("error", {}).get("message", response.text[:200]))
        except Exception:  # noqa: BLE001 - error paths must never raise
            return response.text[:200]

    @staticmethod
    def list_models(settings: Settings) -> List[str]:
        """List model IDs this key can call. Used by check_setup.py."""
        import requests

        response = requests.get(
            f"{GEMINI_API_BASE}/models",
            headers={"x-goog-api-key": settings.api_key or ""},
            timeout=settings.request_timeout,
        )
        if response.status_code in (401, 403):
            raise AuthError(f"Authentication failed ({response.status_code}).")
        if response.status_code != 200:
            raise ProviderError(f"Could not list models ({response.status_code}).")

        models = response.json().get("models", [])
        return [
            m.get("name", "").replace("models/", "")
            for m in models
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        ]


# --- offline stub -----------------------------------------------------------

_PET_LINE = re.compile(r"^\s*-\s+(?P<name>[^(]+?)\s+\((?P<species>[a-zA-Z ]+?)\)", re.MULTILINE)

# Species-appropriate default routines, drawn from the same knowledge base the
# live model reads. Times are spread so the stub does not self-conflict.
_STUB_ROUTINES: Dict[str, List[Dict[str, str]]] = {
    "dog": [
        {"description": "Morning walk", "time": "07:30", "frequency": "daily"},
        {"description": "Breakfast", "time": "08:00", "frequency": "daily"},
        {"description": "Evening walk", "time": "17:30", "frequency": "daily"},
        {"description": "Dinner", "time": "18:30", "frequency": "daily"},
    ],
    "cat": [
        {"description": "Morning meal", "time": "07:15", "frequency": "daily"},
        {"description": "Scoop litter tray", "time": "09:00", "frequency": "daily"},
        {"description": "Evening play session", "time": "19:30", "frequency": "daily"},
        {"description": "Evening meal", "time": "20:00", "frequency": "daily"},
    ],
}

_STUB_DEFAULT = [
    {"description": "Morning care check", "time": "08:15", "frequency": "daily"},
    {"description": "Evening care check", "time": "18:15", "frequency": "daily"},
]


class StubProvider(LLMProvider):
    """Deterministic offline provider.

    Applies fixed species-based routines instead of calling a model. It exists so
    that (a) the test suite and eval harness run with no key and no network, and
    (b) the agent has something safe to degrade to when the live call fails.

    It is genuinely dumb — it does not read the owner's constraints, only which
    pets exist. That gap is the point: the eval harness measures how much the live
    model actually beats this floor.
    """

    name = "stub"
    model = "deterministic-rules-v1"

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> LLMResponse:
        started = time.monotonic()

        pets = [
            (m.group("name").strip(), m.group("species").strip().lower())
            for m in _PET_LINE.finditer(prompt)
        ]

        if json_mode:
            text = json.dumps(self._plan(pets), indent=2)
        else:
            text = (
                "The offline planner is active, so this answer is not model-generated. "
                "It applies fixed species routines from the local knowledge base. "
                "Set GEMINI_API_KEY in .env for a real grounded answer."
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        logger.info("stub provider used (offline, deterministic)")
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            degraded=True,
        )

    @staticmethod
    def _plan(pets: List[Any]) -> Dict[str, Any]:
        """Build a schema-shaped plan, nudging times apart to avoid collisions."""
        tasks: List[Dict[str, Any]] = []
        used_times: set = set()

        for offset, (name, species) in enumerate(pets):
            for routine in _STUB_ROUTINES.get(species, _STUB_DEFAULT):
                time_slot = routine["time"]
                # Each additional pet shifts its routine later, so two pets do not
                # land every task at the same minute.
                if time_slot in used_times:
                    hour, minute = (int(part) for part in time_slot.split(":"))
                    minute += 10 * offset
                    hour, minute = hour + minute // 60, minute % 60
                    time_slot = f"{hour % 24:02d}:{minute:02d}"
                used_times.add(time_slot)

                tasks.append(
                    {
                        "pet": name,
                        "description": routine["description"],
                        "time": time_slot,
                        "frequency": routine["frequency"],
                        "reason": f"Standard {species} routine from the local knowledge base.",
                        "sources": [],
                    }
                )

        return {
            "tasks": tasks,
            "summary": (
                "Offline fallback plan built from fixed species routines. "
                "It does not account for your stated constraints."
            ),
            "confidence": 0.3,
            "caveats": [
                "Generated without a language model, so the owner's specific "
                "constraints were not considered."
            ],
        }


# --- factory ----------------------------------------------------------------


def build_provider(settings: Settings) -> LLMProvider:
    """Pick a provider from settings, degrading to the stub rather than crashing."""
    if settings.provider == "stub":
        logger.info("Provider: stub (explicitly selected)")
        return StubProvider()

    if settings.provider == "gemini":
        if not settings.api_key:
            logger.warning(
                "No GEMINI_API_KEY found — falling back to the offline stub provider. "
                "Copy .env.example to .env and add a key for live AI."
            )
            return StubProvider()
        logger.info("Provider: gemini (%s)", settings.model)
        return GeminiProvider(settings)

    logger.error("Unknown provider %r; using stub.", settings.provider)
    return StubProvider()
