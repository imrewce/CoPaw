# -*- coding: utf-8 -*-
"""Responder logic for proactive conversation feature."""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from agentscope.message import Msg


if TYPE_CHECKING:
    from ..memory import MemoryManager
    from reme.memory.file_based import ReMeInMemoryMemory

from .proactive_types import (
    ProactiveTask,
    ProactiveQueryResult,
)
from .proactive_prompts import (
    PROACTIVE_TASK_EXTRACTION_PROMPT,
    PROACTIVE_USER_FACING_MESSAGE_PROMPT
)
from .proactive_utils import (
    get_recent_memory_file_paths,
    build_proactive_memory_context,
    load_json_safely
)
from ..tools import (
    browser_use,
    read_file,
    execute_shell_command
)
from copaw.constant import WORKING_DIR

logger = logging.getLogger(__name__)


async def generate_proactive_response(
    session_id: str,
    memory_manager: Optional['MemoryManager'] = None,
    in_memory: Optional['ReMeInMemoryMemory'] = None
) -> Optional[Msg]:
    """Main function to generate proactive response based on memory.

    Args:
        session_id: The session identifier
        memory_manager: Optional memory manager instance to access real memory
        in_memory: Optional in-memory instance to access session history

    Returns:
        A proactive message if successful, None otherwise
    """
    try:
        baseline_timestamp = datetime.now()

        # Get session context
        session_context = await _get_session_context(session_id, in_memory=in_memory)

        # Read recent file-based memories (last 2 days)
        file_memory_contents = await _get_recent_file_memories(WORKING_DIR)

        # Combine and build memory context
        memory_context_str = build_proactive_memory_context(
            session_context=session_context,
            file_memories=file_memory_contents
        )

        # Check for interruption
        if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
            logger.info("Proactive response generation interrupted")
            return None

        # Extract tasks and queries from memory context
        tasks = await _extract_tasks_from_memory(memory_context_str)

        # Check for interruption again
        if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
            logger.info("Proactive response generation interrupted")
            return None

        # Execute highest priority queries
        results = []
        for task in tasks[:3]:  # Limit to top 3 tasks
            if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
                logger.info("Proactive response generation interrupted")
                return None

            result = await _execute_query(task.query)
            results.append(result)

            # If we got useful information, stop and use it
            if result.success and result.data:
                break

        # Generate final proactive message
        if results and any(r.success for r in results):
            message_content = await _generate_final_message(
                memory_context_str,
                results
            )

            if message_content:
                # Create a message with proactive indicator in metadata
                return Msg(
                    name="Friday",
                    role="assistant",
                    content=message_content,
                    metadata={"is_proactive": True, "timestamp": datetime.now()}
                )

    except Exception as e:
        logger.error(f"Error generating proactive response: {e}")

    return None


async def _get_session_context(session_id: str, in_memory: Optional['ReMeInMemoryMemory'] = None) -> str:
    """Get the current session context."""
    if not in_memory:
        return f"Session context for {session_id}. This would normally include recent conversation history and state."

    try:
        messages = await in_memory.get_memory(
            exclude_mark=None,
            prepend_summary=False,
        )

        if not messages:
            return f"No conversation history found for session {session_id}."

        context_parts = []
        for msg in messages:
            sender = getattr(msg, 'name', 'Unknown')
            role = getattr(msg, 'role', 'unknown')

            # Extract content regardless of format
            content = _extract_content(msg.content)
            context_parts.append(f"[{role.upper()} {sender}]: {content}")

        return "\n".join(context_parts)

    except Exception as e:
        logger.warning(f"Could not access real session memory: {e}")
        return f"Session context for {session_id}. Could not access real session memory: {e}"


def _extract_content(content) -> str:
    """Helper to extract string content from various formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if 'text' in block:
                    text_parts.append(block['text'])
                elif 'content' in block:
                    text_parts.append(block['content'])
            elif hasattr(block, 'text'):
                text_parts.append(getattr(block, 'text', ''))
            elif hasattr(block, 'content'):
                text_parts.append(getattr(block, 'content', ''))
            else:
                text_parts.append(str(block))
        return ' '.join(text_parts)
    elif hasattr(content, 'text'):
        return getattr(content, 'text', '')
    elif hasattr(content, 'content'):
        return getattr(content, 'content', '')
    else:
        return str(content)


async def _get_recent_file_memories(working_dir: str) -> List[str]:
    """Get recent file-based memories from the working directory."""
    file_paths = get_recent_memory_file_paths(working_dir, num_days=2)
    memories = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                memories.append(content)
        except Exception as e:
            logger.warning(f"Could not read memory file {file_path}: {e}")

    return memories


async def _extract_tasks_from_memory(memory_context: str) -> List[ProactiveTask]:
    """Extract likely user tasks from memory context."""
    from ..react_agent import CoPawAgent

    try:
        temp_agent = CoPawAgent(max_iters=10)
        temp_agent._sys_prompt = PROACTIVE_TASK_EXTRACTION_PROMPT + f"\n\nMemory:\n{memory_context}"
        temp_agent.name = "ProactiveTaskExtractor"
        logger.info(f"Proactive task extraction prompt: {memory_context}")
        response = await temp_agent(Msg(
            name="User",
            role="user",
            content=f"Extract tasks from this memory context:\n{memory_context}"
        ))

        logger.info(f"Proactive task extraction response: {response}")
        if not response or not response.content:
            return []

        # Handle both string and structured content formats
        text_content = _extract_content(response.content)

        # Try to parse JSON directly
        parsed_data = load_json_safely(text_content)

        if parsed_data and "tasks" in parsed_data:
            return _create_tasks_from_data(parsed_data["tasks"])

        # If direct parsing fails, try regex extraction
        import re
        json_match = re.search(r'\{.*\}', text_content, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            parsed_data = load_json_safely(json_str)

            if parsed_data and "tasks" in parsed_data:
                return _create_tasks_from_data(parsed_data["tasks"])

    except Exception as e:
        logger.error(f"Error extracting tasks from memory: {e}")

    return []


def _create_tasks_from_data(tasks_data: List[Dict]) -> List[ProactiveTask]:
    """Helper to create ProactiveTask instances from data."""
    tasks = []
    for i, task_data in enumerate(tasks_data):
        if "task" in task_data and "query" in task_data:
            tasks.append(ProactiveTask(
                task=task_data["task"],
                query=task_data["query"],
                priority=i + 1,  # Higher index = lower priority
                reason=task_data.get("why", "")
            ))
    return tasks


async def _execute_query(query: str) -> ProactiveQueryResult:
    """Execute a query using available tools."""
    try:
        from agentscope.tool import Toolkit
        from ..react_agent import CoPawAgent

        toolkit = Toolkit()
        toolkit.register_tool_function(browser_use)
        toolkit.register_tool_function(read_file)
        toolkit.register_tool_function(write_file)
        toolkit.register_tool_function(edit_file)
        toolkit.register_tool_function(execute_shell_command)

        query_agent = CoPawAgent(max_iters=10)
        query_agent._sys_prompt = f"You are a helpful assistant. Execute this query: {query}"
        query_agent.name = "ProactiveQueryAgent"
        query_agent.toolkit = toolkit

        response = await query_agent(Msg(
            name="User",
            role="user",
            content=query
        ))

        return ProactiveQueryResult(
            query=query,
            success=True,
            data=response.content if response else None
        )
    except Exception as e:
        logger.error(f"Error executing query '{query}': {e}")
        return ProactiveQueryResult(
            query=query,
            success=False,
            error=str(e)
        )


async def _generate_final_message(
    memory_context: str,
    query_results: List[ProactiveQueryResult]
) -> Optional[str]:
    """Generate the final proactive message for the user."""
    try:
        # Prepare gathered info from successful queries
        gathered_info = ""
        for result in query_results:
            if result.success and result.data:
                gathered_info += f"Query: {result.query}\nResult: {result.data}\n\n"

        if not gathered_info.strip():
            return None

        from ..react_agent import CoPawAgent

        message_agent = CoPawAgent(max_iters=10)
        message_agent._sys_prompt = PROACTIVE_USER_FACING_MESSAGE_PROMPT.format(
            session_context=memory_context,
            gathered_info=gathered_info
        )
        message_agent.name = "ProactiveMessageGenerator"

        response = await message_agent(Msg(
            name="User",
            role="user",
            content="Generate a helpful proactive message based on the context and information."
        ))

        if response and response.content:
            return response.content

    except Exception as e:
        logger.error(f"Error generating final proactive message: {e}")

    return None


async def _was_interrupted(baseline_timestamp: datetime, in_memory: Optional['ReMeInMemoryMemory'] = None) -> bool:
    """Check if the proactive process was interrupted by new user activity."""
    if not in_memory:
        return False

    try:
        messages = await in_memory.get_memory(
            exclude_mark=None,
            prepend_summary=False,
        )

        for msg in messages:
            if hasattr(msg, 'timestamp') and msg.role == 'user':
                msg_time = msg.timestamp
                if isinstance(msg_time, str):
                    from datetime import datetime
                    msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00'))

                if msg_time > baseline_timestamp:
                    return True

    except Exception as e:
        logger.warning(f"Could not check session memory for interruptions: {e}")

    return False