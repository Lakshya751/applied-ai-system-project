"""PawPal+ AI layer.

Sits on top of the existing logic layer in ``pawpal_system.py``. Nothing in this
package mutates a schedule directly: the agent proposes tasks, the validator
checks them against the same rules the Scheduler enforces, and only validated
tasks are committed through ``Pet.add_task``.
"""

from pawpal_ai.config import Settings, load_settings

__all__ = ["Settings", "load_settings"]
