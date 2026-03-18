# -*- coding: utf-8 -*-
"""Memory management module for CoPaw agents."""

from .agent_md_manager import AgentMdManager
from .memory_manager import MemoryManager
from .proactive_trigger import enable_proactive_for_session, proactive_trigger_loop, update_last_interaction_time, get_proactive_config, reset_proactive_session
from .proactive_responder import generate_proactive_response
from .proactive_types import ProactiveConfig, ProactiveTask, ProactiveQueryResult, ProactiveMemoryContext
from .proactive_utils import get_recent_memory_file_paths, build_proactive_memory_context, load_json_safely

__all__ = [
    "AgentMdManager",
    "MemoryManager",
    "enable_proactive_for_session",
    "proactive_trigger_loop",
    "generate_proactive_response",
    "ProactiveConfig",
    "ProactiveTask",
    "ProactiveQueryResult",
    "ProactiveMemoryContext",
    "get_recent_memory_file_paths",
    "build_proactive_memory_context",
    "update_last_interaction_time",
    "get_proactive_session",
    "get_proactive_config",
    "reset_proactive_session",
    "load_json_safely",
]
