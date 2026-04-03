# -*- coding: utf-8 -*-
"""Responder logic for proactive conversation feature."""

import logging
from datetime import datetime
from typing import Optional, List, Dict, TYPE_CHECKING
from agentscope.message import Msg


if TYPE_CHECKING:
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
    build_proactive_memory_context,
    load_json_safely,
    extract_content
)
from ..tools import (
    browser_use,
    read_file,
    execute_shell_command
)


logger = logging.getLogger(__name__)


async def generate_proactive_response(
    session_id: str,
    in_memory: Optional['ReMeInMemoryMemory'] = None,
) -> Optional[Msg]:
    from ...app.multi_agent_manager import MultiAgentManager
    from ...app.agent_context import get_current_agent_id

    """Main function to generate proactive response based on memory.

    Args:
        session_id: The session identifier
        memory_manager: Optional memory manager instance to access real memory
        in_memory: Optional in-memory instance to access session history

    Returns:
        A proactive message if successful, None otherwise
    """
    baseline_timestamp = datetime.now()

    # Get the current active agent ID
    active_agent_id = get_current_agent_id()

    # Get the workspace for the active agent
    multi_agent_manager = MultiAgentManager()
    workspace = await multi_agent_manager.get_agent(active_agent_id)

    # Create a single agent instance for all operations
    agent = await _initialize_single_proactive_agent(session_id, workspace, active_agent_id)


    # Combine and build memory context
    memory_context_str = await build_proactive_memory_context(
        agent_workspace_path=str(workspace.workspace_dir),
        workspace=workspace
    )


    # Check for interruption
    if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
        logger.info("Proactive response generation interrupted")
        return None

    # Extract tasks and queries from memory context using the single agent
    tasks = await _extract_tasks_from_memory(memory_context_str, agent)

    # Check for interruption again
    if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
        logger.info("Proactive response generation interrupted")
        return None

    # Execute highest priority queries using the single agent
    results = []
    for task in tasks[:3]:  # Limit to top 3 tasks
        if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
            logger.info("Proactive response generation interrupted")
            return None

        result = await _execute_query(task.query, agent)
        results.append(result)

        # If we got useful information, stop and use it
        if result.success and result.data:
            break
    
    if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
        logger.info("Proactive response generation interrupted")
        return None

    # Generate final proactive message using the single agent
    if results:
        message_content = await _generate_final_message(
            results[-1], agent
        )

        if message_content:
            return message_content

    return None



async def _initialize_single_proactive_agent(session_id: str, workspace, agent_id: str = "proactive"):
    """Initialize a single proactive agent instance to be used across all operations."""
    from ..react_agent import CoPawAgent
    from ...config.config import load_agent_config
    from ...app.runner.utils import build_env_context
    from ...app.runner.models import ChatSpec
    from agentscope.tool import Toolkit


    # Load agent configuration
    agent_config = load_agent_config(agent_id)
    agent_config.running.max_iters = 30

    # Create environment context
    env_context = build_env_context(
        session_id=session_id,
        user_id="proactive_system",
        channel="console",
        working_dir=str(workspace.workspace_dir)
    )

    # Create agent with proper configuration and context
    agent = CoPawAgent(
        agent_config=agent_config,
        env_context=env_context,
        mcp_clients=await workspace.mcp_manager.get_clients() if workspace.mcp_manager else [],
        memory_manager=workspace.memory_manager,
        request_context={
            "session_id": session_id,
            "user_id": "proactive_system",
            "channel": "console",
            "agent_id": agent_id
        },
        workspace_dir=workspace.workspace_dir
    )

    # Complete agent setup
    await agent.register_mcp_clients()
    toolkit = Toolkit()
    toolkit.register_tool_function(browser_use)
    toolkit.register_tool_function(read_file)
    toolkit.register_tool_function(execute_shell_command)

    # Load session state if it exists
    try:
        await workspace.runner.session.load_session_state(
            session_id=session_id,
            user_id="default",
            agent=agent
        )
    except KeyError:
        pass

    agent.toolkit = toolkit
    # Update the memory to reflect the new toolkit
    if agent.memory_manager is not None:
        agent.memory = agent.memory_manager.get_in_memory_memory()

    return agent


async def _extract_tasks_from_memory(memory_context: str, agent) -> List[ProactiveTask]:
    """Extract likely user tasks from memory context using the shared agent."""

    # Process the memory context to extract tasks
    print("##########Memory context:", memory_context)
    response = await agent.reply(Msg(
        name="User",
        role="user",
        content=f"{PROACTIVE_TASK_EXTRACTION_PROMPT}\n#Contexts: {memory_context}"
    ))

    if not response or not response.content:
        return []

    # Handle both string and structured content formats
    text_content = extract_content(response.content)

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

async def _execute_query(query: str, agent) -> ProactiveQueryResult:
    """Execute a query using available tools with the shared agent."""
    import re

    response = await agent.reply(Msg(
        name="User",
        role="user",
        content = f"""Task: Answer: {query} using tools --
                    `browser_use` primary, `execute_shell_command`/`read_file` only if essential.
                    Self-check: Did you retrieve new, query-relevant data?
                    Output: Query answer and end strictly with `[SUCCESS]` (yes) or `[FAILURE]` (no).
                    ⚠️ CRITICAL: The flag MUST be the absolute last token. No trailing text."""
    ))

    success = False
    response_content = ""
    if response and response.content:
        if isinstance(response.content, list) and len(response.content) > 0:
            first_content = response.content[0]
            if isinstance(first_content, dict):
                response_content = first_content.get('text', '')
            elif isinstance(first_content, str):
                response_content = first_content
            else:
                response_content = str(first_content)
        else:
            response_content = str(response.content)

    if response_content:
        match = re.search(r'\[(SUCCESS|FAILURE)\]\s*$', response_content.strip())
        if match:
            success = (match.group(1) == 'SUCCESS')

    return ProactiveQueryResult(
        query=query,
        success=success,
        data=response_content,
    )


async def _generate_final_message(
    result: ProactiveQueryResult, agent
) -> Optional[str]:
    """Generate the final proactive message for the user."""
    # Prepare gathered info from successful queries
    gathered_info = ""

    if result.data:
        gathered_info += f"Query: {result.query}\nResult: {result.data}\n\n"

    if not gathered_info.strip():
        return None

    proactive_content = PROACTIVE_USER_FACING_MESSAGE_PROMPT.format(
        gathered_info=gathered_info
    )

    # Import required modules from agent_context
    from ...app.agent_context import get_current_agent_id

    # Get the current active agent ID
    active_agent_id = get_current_agent_id()

    # Send the proactive message using async HTTP request instead of subprocess
    return await send_proactive_message_via_http(
        active_agent_id=active_agent_id,
        proactive_content=proactive_content,
        timeout_seconds=300
    )


async def send_proactive_message_via_http(
    active_agent_id: str,
    proactive_content: str,
    base_url: str = "http://127.0.0.1:8088", # Default CoPaw address
    timeout_seconds: int = 300,
) -> str:
    """
    Send a proactive message by directly calling the CoPaw API.

    This is non-blocking and safe to use within an async Uvicorn context.
    """
    import aiohttp
    import asyncio

    # Construct request payload similar to what CLI does
    request_payload = {
        "session_id": f"proactive:{active_agent_id}:{int(asyncio.get_event_loop().time() * 1000)}", # Simple session ID
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"[Agent proactive_helper requesting] {proactive_content}"}],
            },
        ],
    }

    headers = {"X-Agent-Id": active_agent_id}

    timeout_config = aiohttp.ClientTimeout(total=timeout_seconds)
    clean_base = base_url.rstrip("/")
    if not clean_base.endswith("/api"):
        api_base_url = f"{clean_base}/api"
    else:
        api_base_url = clean_base

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{api_base_url.rstrip('/')}/agent/process"
            async with session.post(
                url,
                json=request_payload,
                headers=headers,
                timeout=timeout_config
            ) as resp:
                resp.raise_for_status()

                # Handle stream response (SSE) - get the last data event
                last_data = None
                async for line_bytes in resp.content:
                    line = line_bytes.decode('utf-8').strip()
                    if line.startswith("data: "):
                        try:
                            last_data = line[6:]  # Remove "data: "
                        except Exception:
                            continue

                if last_data:
                    logger.info("Proactive message sent successfully via direct HTTP call")
                else:
                    logger.warning("No valid SSE data received from agent")

    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout_seconds}s) calling CoPaw API for proactive message")
    except Exception as e:
        logger.error(f"Error calling CoPaw API for proactive message: {e}")

    # Return the proactive content to ensure the flow continues
    return proactive_content





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