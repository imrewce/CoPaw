# -*- coding: utf-8 -*-
"""Proactive memory submodule for CoPaw agents."""

from .proactive_types import (
    ProactiveConfig,
    ProactiveStatus,
    ProactiveTask,
    ProactiveQueryResult,
    ProactiveMemoryContext,
)
from .proactive_trigger import (
    enable_proactive_for_session,
    proactive_trigger_loop,
    proactive_tasks,
    proactive_configs,
)
from .proactive_responder import generate_proactive_response
from .proactive_utils import extract_content

__all__ = [
    "ProactiveConfig",
    "ProactiveStatus",
    "ProactiveTask",
    "ProactiveQueryResult",
    "ProactiveMemoryContext",
    "enable_proactive_for_session",
    "proactive_trigger_loop",
    "proactive_tasks",
    "proactive_configs",
    "generate_proactive_response",
    "extract_content",
]
