# -*- coding: utf-8 -*-
"""Memory management module for CoPaw agents."""

from .agent_md_manager import AgentMdManager
from .memory_manager import MemoryManager
from .proactive_types import ProactiveConfig, ProactiveTask, ProactiveQueryResult, ProactiveMemoryContext

# Import essential proactive functions directly to avoid dynamic imports
from .proactive_trigger import enable_proactive_for_session, proactive_trigger_loop
from .proactive_responder import generate_proactive_response
from .proactive_utils import extract_content

__all__ = [
    "AgentMdManager",
    "MemoryManager",
    "ProactiveConfig",
    "ProactiveTask",
    "ProactiveQueryResult",
    "ProactiveMemoryContext",
    "enable_proactive_for_session",
    "proactive_trigger_loop",
    "generate_proactive_response",
    "extract_content"
]
