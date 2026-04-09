# -*- coding: utf-8 -*-
"""Responder logic for proactive conversation feature."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict

import aiohttp
from agentscope.message import Msg
from agentscope.tool import Toolkit

from ....app.agent_context import get_current_agent_id
from ....app.multi_agent_manager import MultiAgentManager
from ....app.runner.utils import build_env_context
from ....config.config import load_agent_config
from ...react_agent import CoPawAgent
from ...tools import (
    browser_use,
    desktop_screenshot,
    execute_shell_command,
    read_file,
)
from .proactive_prompts import (
    PROACTIVE_TASK_EXTRACTION_PROMPT,
    PROACTIVE_USER_FACING_MESSAGE_PROMPT,
)
from .proactive_types import ProactiveQueryResult, ProactiveTask
from .proactive_utils import (
    build_proactive_memory_context,
    load_json_safely,
)

logger = logging.getLogger(__name__)


async def generate_proactive_response(
    session_id: str,
    in_memory: Optional[object] = None,
) -> Optional[Msg]:
    """Main function to generate proactive response based on memory."""
    baseline_timestamp = datetime.now()
    active_agent_id = get_current_agent_id()

    multi_agent_manager = MultiAgentManager()
    workspace = await multi_agent_manager.get_agent(active_agent_id)

    agent = await _initialize_single_proactive_agent(
        session_id, workspace, active_agent_id
    )

    memory_context_str = await build_proactive_memory_context(
        agent_workspace_path=str(workspace.workspace_dir),
        workspace=workspace,
        agent=agent,
    )

    if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
        logger.info("Proactive response generation interrupted")
        return None

    tasks = await _extract_tasks_from_memory(memory_context_str, agent)

    if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
        logger.info("Proactive response generation interrupted")
        return None

    results = []
    for task in tasks[:3]:
        if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
            logger.info("Proactive response generation interrupted")
            return None

        result = await _execute_query(task.query, agent)
        results.append(result)

        if result.success and result.data:
            break

    if await _was_interrupted(baseline_timestamp, in_memory=in_memory):
        logger.info("Proactive response generation interrupted")
        return None

    if results:
        message_content = await _generate_final_message(results[-1], agent)
        if message_content:
            return message_content

    return None


async def _initialize_single_proactive_agent(
    session_id: str, workspace, agent_id: str = "proactive"
) -> CoPawAgent:
    """Initialize a single proactive agent instance."""
    agent_config = load_agent_config(agent_id)
    agent_config.running.max_iters = 50

    env_context = build_env_context(
        session_id=session_id,
        user_id="proactive_system",
        channel="console",
        working_dir=str(workspace.workspace_dir),
    )

    mcp_clients = (
        await workspace.mcp_manager.get_clients()
        if workspace.mcp_manager
        else []
    )

    agent = CoPawAgent(
        agent_config=agent_config,
        env_context=env_context,
        mcp_clients=mcp_clients,
        memory_manager=workspace.memory_manager,
        request_context={
            "session_id": session_id,
            "user_id": "proactive_system",
            "channel": "console",
            "agent_id": agent_id,
        },
        workspace_dir=workspace.workspace_dir,
    )

    await agent.register_mcp_clients()

    toolkit = Toolkit()
    toolkit.register_tool_function(browser_use)
    toolkit.register_tool_function(read_file)
    toolkit.register_tool_function(execute_shell_command)
    toolkit.register_tool_function(desktop_screenshot)

    try:
        await workspace.runner.session.load_session_state(
            session_id=session_id,
            user_id="default",
            agent=agent,
        )
    except KeyError:
        pass

    agent.toolkit = toolkit
    if agent.memory_manager is not None:
        agent.memory = agent.memory_manager.get_in_memory_memory()

    return agent


async def _extract_tasks_from_memory(
    memory_context: str, agent: CoPawAgent
) -> List[ProactiveTask]:
    """Extract likely user tasks from memory context."""
    prompt = f"{PROACTIVE_TASK_EXTRACTION_PROMPT}\n#Contexts: {memory_context}"
    response = await agent.reply(Msg(name="User", role="user", content=prompt))

    if not response or not response.content:
        return []

    text_content = response.get_text_content()
    parsed_data = load_json_safely(text_content)

    if parsed_data and "tasks" in parsed_data:
        return _create_tasks_from_data(parsed_data["tasks"])

    json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
    if json_match:
        parsed_data = load_json_safely(json_match.group(0))
        if parsed_data and "tasks" in parsed_data:
            return _create_tasks_from_data(parsed_data["tasks"])

    return []


def _create_tasks_from_data(tasks_data: List[Dict]) -> List[ProactiveTask]:
    """Helper to create ProactiveTask instances from data."""
    tasks = []
    for i, task_data in enumerate(tasks_data):
        if "task" in task_data and "query" in task_data:
            tasks.append(
                ProactiveTask(
                    task=task_data["task"],
                    query=task_data["query"],
                    priority=i + 1,
                    reason=task_data.get("why", ""),
                )
            )
    return tasks


async def _execute_query(
    query: str, agent: CoPawAgent
) -> ProactiveQueryResult:
    """Execute a query using available tools."""
    prompt = (
        f"Task: Answer: {query} using tools -- "
        "`browser_use` primary, `execute_shell_command`/`read_file` "
        "only if essential.\n"
        "Self-check: Did you retrieve new, query-relevant data or "
        "complete given task?\n"
        "Output: Query answer and end strictly with `[SUCCESS]` "
        "(yes) or `[FAILURE]` (no).\n"
        "⚠️ CRITICAL: The flag MUST be the absolute last token. "
        "No trailing text."
    )

    response = await agent.reply(
        Msg(name="User", role="user", content=prompt)
    )

    success = False
    response_content = response.get_text_content()
    if response_content:
        match = re.search(r"\[(SUCCESS)\]\s*$", response_content.strip())
        if match:
            success = True

    return ProactiveQueryResult(
        query=query,
        success=success,
        data=response_content,
    )


async def _generate_final_message(
    result: ProactiveQueryResult, agent: CoPawAgent
) -> Optional[Msg]:
    """Generate the final proactive message for the user."""
    if not result.data:
        return None

    gathered_info = f"Query: {result.query}\nResult: {result.data}\n\n"

    active_agent_id = get_current_agent_id()
    agent_language = load_agent_config(active_agent_id).language
    proactive_content = PROACTIVE_USER_FACING_MESSAGE_PROMPT.format(
        gathered_info=gathered_info,
        language = agent_language,
    )
    proactive_msg_content = await send_proactive_message_via_http(
        active_agent_id=active_agent_id,
        proactive_content=proactive_content,
        timeout_seconds=300,
    )

    if proactive_msg_content:
        return Msg(
            name="Proactive_Assistant",
            role="assistant",
            content=f"[PROACTIVE] {proactive_msg_content}",
            timestamp=datetime.now(),
        )

    return None


async def send_proactive_message_via_http(
    active_agent_id: str,
    proactive_content: str,
    base_url: str = "http://127.0.0.1:8088",
    timeout_seconds: int = 300,
) -> str:
    """Send a proactive message by directly calling the CoPaw API."""
    session_id = (
        f"proactive_mode:{active_agent_id}"
    )

    request_payload = {
        "session_id": session_id,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "[Agent proactive_helper requesting] "
                            f"{proactive_content}"
                        ),
                    }
                ],
            }
        ],
    }

    headers = {"X-Agent-Id": active_agent_id}
    timeout_config = aiohttp.ClientTimeout(total=timeout_seconds)

    clean_base = base_url.rstrip("/")
    api_base_url = (
        f"{clean_base}/api"
        if not clean_base.endswith("/api")
        else clean_base
    )

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{api_base_url.rstrip('/')}/agent/process"
            async with session.post(
                url,
                json=request_payload,
                headers=headers,
                timeout=timeout_config,
            ) as resp:
                resp.raise_for_status()
                last_data = None
                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8").strip()
                    if line.startswith("data: "):
                        try:
                            last_data = line[6:]
                        except Exception:
                            continue

                if last_data:
                    logger.info(
                        "Proactive message sent successfully via HTTP"
                    )
                else:
                    logger.warning("No valid SSE data received from agent")

    except asyncio.TimeoutError:
        logger.error(
            "Timeout (%ds) calling CoPaw API for proactive message",
            timeout_seconds,
        )
    except Exception as e:
        logger.error(
            "Error calling CoPaw API for proactive message: %s", e
        )

    return proactive_content


async def _was_interrupted(
    baseline_timestamp: datetime,
    in_memory: Optional[object] = None,
) -> bool:
    """Check if the proactive process was interrupted by new user activity."""
    if not in_memory:
        return False

    try:
        messages = await in_memory.get_memory(
            exclude_mark=None,
            prepend_summary=False,
        )

        for msg in messages:
            if hasattr(msg, "timestamp") and msg.role == "user":
                msg_time = msg.timestamp
                if isinstance(msg_time, str):
                    msg_time = datetime.fromisoformat(
                        msg_time.replace("Z", "+00:00")
                    )

                if msg_time > baseline_timestamp:
                    return True

    except Exception as e:
        logger.warning(
            "Could not check session memory for interruptions: %s", e
        )

    return False
