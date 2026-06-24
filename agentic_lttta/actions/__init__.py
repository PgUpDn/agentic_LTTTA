"""Bounded action space for the LTTTA controller.

Importing this package registers every action into ``ACTION_REGISTRY``.
"""

from __future__ import annotations

from .base import (
    ACTION_REGISTRY,
    ActionResult,
    BlockContext,
    action_names,
    register,
)

# Import side-effect: each module registers its action into ACTION_REGISTRY.
from . import skip      # noqa: F401,E402
from . import observe   # noqa: F401,E402
from . import recalibrate  # noqa: F401,E402
from . import adapter   # noqa: F401,E402
from . import memory    # noqa: F401,E402
from . import expert    # noqa: F401,E402

# Canonical ordering of the bounded action space.
BOUNDED_ACTIONS = [
    "skip_update",
    "recalibrate",
    "update_adapter",
    "observe",
    "retrieve_memory",
    "select_expert",
]

__all__ = [
    "ACTION_REGISTRY",
    "ActionResult",
    "BlockContext",
    "BOUNDED_ACTIONS",
    "action_names",
    "register",
]
