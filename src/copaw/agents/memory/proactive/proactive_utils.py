# -*- coding: utf-8 -*-
"""Utility functions for proactive messaging features."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any
from agentscope.message import Msg
from ...react_agent import CoPawAgent

logger = logging.getLogger(__name__)


def ensure_tz_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware, convert naive datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def build_proactive_memory_context(
    agent_workspace_path: Optional[str] = None,
    max_session_messages: int = 100,
    max_session_chars: int = 50000,
    workspace=None,
    agent=None,
) -> str:
    """Build a combined memory context for proactive agent."""
    if agent_workspace_path is None:
        raise ValueError("agent_workspace_path must be provided")

    from ...prompt import get_active_model_supports_multimodal

    combined_context = ""

    # Capture screen if supported
    if agent and get_active_model_supports_multimodal():
        try:
            screen_analysis = await _analyze_screen_activity(agent)
            if screen_analysis:
                combined_context += "[SCREEN CONTEXT]\n"
                combined_context += screen_analysis
        except Exception as e:
            logger.warning("Failed to analyze screen activity: %s", e)

    combined_context += "[SESSION CONTEXT]\n"

    if workspace is None:
        from ....app.agent_context import get_current_agent_id
        from ....app.multi_agent_manager import MultiAgentManager

        active_agent_id = get_current_agent_id()
        multi_agent_manager = MultiAgentManager()
        workspace = await multi_agent_manager.get_agent(active_agent_id)

    sessions_to_read = await _read_chat_sessions_metadata(workspace)
    if not sessions_to_read:
        return combined_context if combined_context != "[SESSION CONTEXT]\n" else ""

    filtered_sessions = _filter_recent_sessions(sessions_to_read)
    all_messages = await _collect_session_messages(filtered_sessions, workspace)

    if all_messages:
        session_context = _format_session_messages(
            all_messages, max_session_messages, max_session_chars
        )
        print("######", session_context)
        if session_context:
            combined_context += session_context + "\n\n"

    return combined_context


def _clean_message_content(msg: "Msg") -> Optional["Msg"]:
    """
    Args:
        msg: The native AgentScope Msg object.
        
    Returns:
        A cleaned Msg object if valid, otherwise None.
    """

    if msg.role == "system":
        return None

    content = msg.content

    if isinstance(content, list):
        cleaned_blocks: List[Any] = []
        
        for block in content:
            block_type = ""
            
            if isinstance(block, dict):
                block_type = block.get("type", "")
            elif hasattr(block, 'type'):
                block_type = getattr(block, 'type', '')
            else:
                continue

            if block_type == "text":
                cleaned_blocks.append(block)
            
  
        if not cleaned_blocks:
            return None
            
    
        msg.content = cleaned_blocks
        return msg

    return None



async def _process_session_memory(
    session_id: str, user_id: str, workspace
) -> List[dict]:
    """Process a session's memory and return a list of messages."""
    from agentscope.memory import InMemoryMemory
    from ....app.runner.utils import agentscope_msg_to_message

    try:
        state = await workspace.runner.session.get_session_state_dict(
            session_id, user_id
        )
        if not state:
            return []

        memories_data = state.get("agent", {}).get("memory", [])
        if not memories_data:
            return []

        memory = InMemoryMemory()
        memory.load_state_dict(memories_data)
        messages = await memory.get_memory()
   
        processed_messages = []
        default_time = datetime.now(timezone.utc)
        
        for msg in messages:
            msg = _clean_message_content(msg)
            if msg is None:
                continue

            timestamp = default_time
            if msg.timestamp:
                try:
                    dt_obj = datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    # Ensure timezone aware (assuming UTC for consistency)
                    timestamp = dt_obj.replace(tzinfo=timezone.utc)
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp '{msg.timestamp}': {e}")
            
            processed_messages.append({
                "message": msg,
                "timestamp": timestamp,
            })

        # Sort by timestamp descending
        processed_messages.sort(key=lambda x: x["timestamp"], reverse=True)
        return processed_messages

    except Exception as e:
        logger.warning(
            "Could not read session memory for %s/%s: %s",
            session_id, user_id, e
        )
        return []


def load_json_safely(json_input) -> Optional[dict]:
    """Safely parse JSON string, returning None if invalid."""
    if not isinstance(json_input, str):
        return None

    cleaned_str = json_input.strip()

    # Handle code blocks
    if cleaned_str.startswith("```json"):
        cleaned_str = cleaned_str[7:].split("```")[0].strip()
    elif cleaned_str.startswith("```"):
        cleaned_str = cleaned_str[3:].split("```")[0].strip()

    try:
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object
    try:
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(json_input):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(json_input[start_idx : i + 1])
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return None


def extract_content(content) -> str:
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if hasattr(block, 'text'):
                text_parts.append(getattr(block, 'text', ''))
            else:
                text_parts.append(str(block))
        return ' '.join(text_parts)
    else:
        return str(content)


async def _analyze_screen_activity(
        agent:CoPawAgent,
) -> Optional[str]:
    """Analyze user's screen activity using multimodal capabilities."""
    from agentscope.message import Msg
    from ...tools.desktop_screenshot import desktop_screenshot

    try:
        screenshot_result = await desktop_screenshot()
        if (
            not screenshot_result
            or not hasattr(screenshot_result, 'content')
            or not screenshot_result.content
        ):
            return None

        content = screenshot_result.content
        if isinstance(content, list) and len(content) > 0:
            result_text = content[0].get('text', '')
        else:
            result_text = str(content)

        try:
            result_json = json.loads(result_text)
            if not result_json.get("ok", True):
                logger.warning(
                    "Screenshot failed: %s",
                    result_json.get('error', 'Unknown error')
                )
                return None

            screenshot_path = result_json.get("path", "")
            if not screenshot_path:
                return None

            analysis_prompt = (
                "Analyze this screenshot of the user's desktop. "
                "Identify what application or activity the user is "
                "currently engaged in."
            )

            screenshot_msg = Msg(
                name="System",
                role="user",
                content=[
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": screenshot_path
                        },
                    },
                ],
            )

            response = await agent.reply(screenshot_msg)
            analysis_result =  response.get_text_content()
            if analysis_result:
                return (
                    "[SCREEN ANALYSIS]\n"
                    "Analysis of user's current desktop activity:\n"
                    f"{analysis_result.strip()}\n"
                    "[SCREEN CONTEXT END]\n\n"
                )

        except json.JSONDecodeError:
            logger.warning("Could not parse screenshot result as JSON")

    except Exception as e:
        logger.warning("Could not capture screen for analysis: %s", e)

    return None


async def _read_chat_sessions_metadata(workspace) -> List[dict]:
    """Read chat sessions metadata from chat manager."""
    sessions_to_read = []
    try:
        chats = await workspace.chat_manager.list_chats()
        for chat in chats:
            user_id = chat.user_id
            session_id = chat.session_id
            updated_at_dt = ensure_tz_aware(chat.updated_at)

            filename = (
                f"{user_id.replace(':', '--')}_"
                f"{session_id.replace(':', '--')}.json"
            )
            sessions_to_read.append({
                'filename': filename,
                'user_id': user_id,
                'session_id': session_id,
                'mod_time': updated_at_dt,
            })
    except Exception as e:
        logger.warning("Could not read chats through chat_manager: %s", e)

    return sessions_to_read


def _filter_recent_sessions(
    sessions_to_read: List[dict], days: int = 7
) -> List[dict]:
    """Filter sessions to only include recent ones."""
    ts_date = ensure_tz_aware(datetime.now(timezone.utc)) - timedelta(
        days=days
    )

    filtered_sessions = [
        s for s in sessions_to_read if s['mod_time'] >= ts_date
    ]

    if len(filtered_sessions) < 5:
        sessions_to_read.sort(key=lambda x: x['mod_time'], reverse=True)
        filtered_sessions = sessions_to_read[:5]
    else:
        filtered_sessions.sort(key=lambda x: x['mod_time'], reverse=True)

    return filtered_sessions


async def _collect_session_messages(
    filtered_sessions: List[dict], workspace
) -> List[dict]:
    """Collect all messages from session memory."""
    all_messages = []
    for session_info in filtered_sessions:
        session_id = session_info['session_id']
        user_id = session_info['user_id']
        try:
            session_messages = await _process_session_memory(
                session_id, user_id, workspace
            )
            if session_messages:
                all_messages.extend(session_messages)
        except Exception as e:
            logger.warning(
                "Could not read session memory for %s/%s: %s",
                session_id, user_id, e
            )
    return all_messages


def _format_session_messages(
    all_messages: List[dict],
    max_messages: int = 100,
    max_chars: int = 50000,
) -> str:
    """Format collected session messages into a context string."""
    all_messages.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_messages = all_messages[:max_messages]

    context_text = "\n"
    for msg_info in recent_messages:
        msg = msg_info["message"]
        role = msg.role if hasattr(msg, 'role') else "unknown"
        content = extract_content(msg.content)
        clean_text = content.replace('\n', ' ')
        concat_text = f"[{role}]: {clean_text}\n" + context_text
        if len(concat_text) > max_chars:
            return context_text
        context_text = concat_text
    
    return context_text


async def get_last_message_ts(in_memory=None) -> Optional[float]:
    """Get the timestamp of the last message."""
    if in_memory:
        try:
            messages = await in_memory.get_memory(
                exclude_mark=None,
                prepend_summary=False,
            )
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'timestamp'):
                    ts = last_msg.timestamp
                    if isinstance(ts, datetime):
                        return ts.timestamp()
                    return ts
        except Exception:
            pass

    try:
        from ....app.agent_context import get_current_agent_id
        from ....app.multi_agent_manager import MultiAgentManager

        multi_agent_manager = MultiAgentManager()
        agent_id = get_current_agent_id()
        workspace = await multi_agent_manager.get_agent(agent_id)
        chats = await workspace.chat_manager.list_chats()

        latest_update_ts = None
        for session in chats:
            updated_at_dt = ensure_tz_aware(session.updated_at)
            ts = updated_at_dt.timestamp()
            if latest_update_ts is None or ts > latest_update_ts:
                latest_update_ts = ts

        return latest_update_ts

    except Exception as e:
        logger.warning(
            "Could not read chats for fallback timestamp: %s", e
        )
        return None
