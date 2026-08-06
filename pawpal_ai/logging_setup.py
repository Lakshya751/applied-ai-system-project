"""Structured logging and run tracing for the AI layer.

Two separate concerns:

* ``get_logger`` - ordinary human-readable logging to stderr and a rolling file.
* ``RunTrace``   - a machine-readable record of one agent run (every step, every
  model call, every validation failure). Traces are what make the agent's
  reasoning auditable after the fact; they are written to ``logs/traces.jsonl``
  and are the source for the reasoning traces quoted in ai_interactions.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pawpal_ai.config import LOG_DIR

_CONFIGURED = False


def _ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def get_logger(name: str = "pawpal_ai") -> logging.Logger:
    """Return the shared logger, configuring handlers exactly once."""
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        log_dir = _ensure_log_dir()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        file_handler = logging.FileHandler(log_dir / "pawpal_ai.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        root = logging.getLogger("pawpal_ai")
        root.setLevel(logging.DEBUG)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
        root.propagate = False
        _CONFIGURED = True

    return logger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TraceStep:
    """One recorded step in an agent run."""

    step: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "detail": self.detail,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class RunTrace:
    """An append-only record of a single agent run."""

    request: str
    provider: str
    steps: List[TraceStep] = field(default_factory=list)
    started_at: str = field(default_factory=_now)

    def add(self, step: str, detail: str, **data: Any) -> None:
        """Record a step and mirror it to the log at DEBUG level."""
        self.steps.append(TraceStep(step=step, detail=detail, data=data))
        get_logger().debug("[%s] %s", step, detail)

    def as_lines(self) -> List[str]:
        """Render the trace as readable lines, for the UI and CLI."""
        return [f"[{s.step}] {s.detail}" for s in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "provider": self.provider,
            "started_at": self.started_at,
            "steps": [s.to_dict() for s in self.steps],
        }

    def persist(self, path: Optional[Path] = None) -> Path:
        """Append this trace to the JSONL trace log and return the file path."""
        target = path or (_ensure_log_dir() / "traces.jsonl")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        return target
