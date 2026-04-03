# -*- coding: utf-8 -*-
"""
Utility functions for proactive messaging features.
Kept self-contained within the CoPaw agents/memory directory.
"""

from typing import List,  Optional
from datetime import  timezone
import json
from reme.memory.file_based import ReMeInMemoryMemory


def ensure_tz_aware(dt):
    """Ensure datetime is timezone-aware, convert naive datetime to UTC."""
    if dt.tzinfo is None:
        # Assume naive datetime is in UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt



async def build_proactive_memory_context(
    agent_workspace_path: Optional[str] = None,
    max_session_messages: int = 30,
    max_session_chars: int = 12500,
    workspace = None
) -> str:
    """Build a combined memory context for proactive agent from active agent workspace.

    Args:
        agent_workspace_path: Path to agent workspace - must be provided, no fallback to WORKING_DIR
        max_session_messages: Maximum number of recent messages to include from sessions (default 30)
        max_session_chars: Maximum character length for the session context (default 12500)
        workspace: Workspace instance to access chat manager and session memory
    """
    import logging

    # agent_workspace_path must be provided - no fallback to WORKING_DIR
    if agent_workspace_path is None:
        raise ValueError("agent_workspace_path must be provided and cannot be None")

    combined_context = "[SESSION CONTEXT]\n"

    # Get workspace reference if not provided
    if workspace is None:
        from ...app.agent_context import get_current_agent_id
        from ...app.multi_agent_manager import MultiAgentManager
        active_agent_id = get_current_agent_id()
        multi_agent_manager = MultiAgentManager()
        workspace = await multi_agent_manager.get_agent(active_agent_id)

    # Read from active agent workspace - sessions and chats
    sessions_to_read = await _read_chat_sessions_metadata(workspace)

    if not sessions_to_read:
        return ""

    # Filter to recent sessions
    filtered_sessions = _filter_recent_sessions(sessions_to_read)
    print("#############", filtered_sessions)

    # Collect messages from session memory
    all_messages = await _collect_session_messages(filtered_sessions, workspace)

    # Format messages into context string
    session_context = ""
    if all_messages:
        session_context = _format_session_messages(all_messages, max_session_messages, max_session_chars)

    if session_context:
        combined_context += session_context + "\n\n"

    return combined_context


async def _process_session_memory(session_id: str, user_id: str, workspace) -> List[dict]:
    """Process a session's memory by loading it through InMemoryMemory and return a list of messages.

    Args:
        session_id: Session identifier
        user_id: User identifier
        workspace: Workspace instance

    Returns:
        List of messages with timestamps
    """
    import logging
    from datetime import datetime
    from agentscope.memory import InMemoryMemory
    from ...app.runner.utils import agentscope_msg_to_message
    logger = logging.getLogger(__name__)

    try:
        # Get the session state dictionary
        state = await workspace.runner.session.get_session_state_dict(
            session_id,
            user_id,
        )

        if not state:
            return []

        # Extract memory data from the state
        memories_data = state.get("agent", {}).get("memory", [])

        # Create InMemoryMemory instance and load state
        memory = InMemoryMemory()
        memory.load_state_dict(memories_data)

        # Get memory messages
        messages = await memory.get_memory()
        

        # Convert to serializable format
        serializable_messages = agentscope_msg_to_message(messages)

        # Process messages and filter out system and internal messages
        processed_messages = []

        def process_message(msg, default_time):
            from agentscope_runtime.engine.schemas.agent_schemas import MessageType

            # Handle both dict and Message object types
            if hasattr(msg, 'role'):  # It's a Message object
                role = msg.role or ""

                # Get content from Message object
                content = msg.content or []

                # Determine message type from Message object
                msg_type = getattr(msg, 'type', '')
            else:  # It's a dict
                role = msg.get("role", "")

                # Check if content contains "thinking" or "tool_use" types
                content = msg.get("content", [])
                msg_type = msg.get("type", "")

            # Skip system messages
            if role == "system":
                return None

            # Skip messages based on type
            if msg_type in [MessageType.REASONING, MessageType.PLUGIN_CALL, MessageType.PLUGIN_CALL_OUTPUT]:
                return None

            # Check if content contains "thinking" or "tool_use" types (for dict format)
            if isinstance(content, list):
                should_skip = False
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type in ["thinking", "tool_use"]:
                            should_skip = True
                            break
                    elif hasattr(item, 'type'):  # Content object
                        item_type = getattr(item, 'type', '')
                        if item_type in ["thinking", "tool_use"]:
                            should_skip = True
                            break
                if should_skip:
                    return None

            # Add timestamp to message for sorting later
            # For Message objects, use timestamp attribute if available
            if hasattr(msg, 'timestamp') and msg.timestamp:
                timestamp = ensure_tz_aware(msg.timestamp)
            else:
                timestamp_str = getattr(msg, 'timestamp', '') or (msg.get("timestamp", "") if isinstance(msg, dict) else "")
                try:
                    if isinstance(timestamp_str, str) and timestamp_str:
                        # Parse timestamp - might be in ISO format
                        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00").replace("+00:00", ""))
                        timestamp = ensure_tz_aware(parsed_timestamp)
                    else:
                        # Use current time if no timestamp in message
                        timestamp = ensure_tz_aware(default_time)
                except:
                    timestamp = ensure_tz_aware(default_time)

            return {
                "message": msg,
                "timestamp": timestamp
            }

        # Process all messages
        for msg in serializable_messages:
            processed_msg = process_message(msg, datetime.now())
            if processed_msg:
                processed_messages.append(processed_msg)

        # Sort messages by timestamp (descending - most recent first)
        processed_messages.sort(key=lambda x: x["timestamp"], reverse=True)

        logger.info(f"Processed session memory for session {session_id} and user {user_id}, filtered out messages containing thinking/tool_use/system roles")
        return processed_messages

    except Exception as e:
        logger.warning(f"Could not read session memory for session {session_id} and user {user_id}: {e}")
        return []


def load_json_safely(json_input) -> Optional[dict]:
    """Safely parse JSON string, returning None if invalid."""
    import logging
    logger = logging.getLogger(__name__)


    # Try direct parsing first
    try:
        cleaned_str = json_input.strip()

        # Handle common code block markers
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:].split("```")[0].strip()
        elif cleaned_str.startswith("```"):
            cleaned_str = cleaned_str[3:].split("```")[0].strip()

        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        pass

    # If direct parsing fails, try to find JSON object within the string
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
                        json_str_extracted = json_input[start_idx:i+1]
                        return json.loads(json_str_extracted)
                    except json.JSONDecodeError:
                        continue
    except Exception:
        logger.debug("JSON extraction from braces failed")

    # Last resort: return None
    return None

def extract_content(content) -> str:
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


async def _read_chat_sessions_metadata(workspace) -> List[dict]:
    """Read chat sessions metadata from chat manager.

    Args:
        agent_workspace_path: Path to the agent workspace (for backward compatibility)
        workspace: Workspace instance to access chat manager

    Returns:
        List of session metadata dictionaries
    """
    import logging
    from datetime import datetime
    logger = logging.getLogger(__name__)

    sessions_to_read = []

    try:
        # Get all chats using chat_manager.list_chats()
        chats = await workspace.chat_manager.list_chats()
        print("##########Chats:", chats)

        # Extract session info from chats
        for chat in chats:
            user_id = chat.user_id
            session_id = chat.session_id

            # Get modification time from the chat's updated_at
            updated_at_dt = chat.updated_at
            updated_at_dt = ensure_tz_aware(updated_at_dt)

            sessions_to_read.append({
                'filename': f"{user_id.replace(':', '--')}_{session_id.replace(':', '--')}.json",  # Kept for backward compatibility
                'user_id': user_id,
                'session_id': session_id,
                'mod_time': updated_at_dt
            })

    except Exception as e:
        logger.warning(f"Could not read chats through chat_manager: {e}")

    return sessions_to_read


def _filter_recent_sessions(sessions_to_read: List[dict], days: int = 7) -> List[dict]:
    """Filter sessions to only include recent ones or return most recent ones if none are recent.

    Args:
        sessions_to_read: List of session metadata
        days: Number of days to look back (default 7)

    Returns:
        Filtered list of sessions
    """
    from datetime import datetime, timedelta
    import logging

    logger = logging.getLogger(__name__)

    # Filter sessions based on time (last 3 days)
    filtered_sessions = []
    ts_date = ensure_tz_aware(datetime.now()) - timedelta(days=days)

    for session_info in sessions_to_read:
        if session_info['mod_time'] >= ts_date:
            filtered_sessions.append(session_info)

    # If no recent sessions, get the 3 most recently modified
    if not filtered_sessions:
        sessions_to_read.sort(key=lambda x: x['mod_time'], reverse=True)
        filtered_sessions = sessions_to_read[:5]
    else:
        filtered_sessions.sort(key=lambda x: x['mod_time'], reverse=True)

    return filtered_sessions


async def _collect_session_messages( filtered_sessions: List[dict], workspace) -> List[dict]:
    """Collect all messages from session memory using InMemoryMemory.

    Args:
        filtered_sessions: List of filtered session metadata
        workspace: Workspace instance to access session memory

    Returns:
        List of messages with timestamps
    """
    import logging

    logger = logging.getLogger(__name__)

    all_messages = []  # Collect all messages to be sorted later

    # Access the session memories through the workspace's memory system
    for session_info in filtered_sessions:
        session_id = session_info['session_id']
        user_id = session_info['user_id']

        try:
            # Pass the modification time from chats.json - returns list of messages
            session_messages = await _process_session_memory(session_id, user_id, workspace)
            if session_messages:
                all_messages.extend(session_messages)  # Extend instead of append
        except Exception as e:
            logger.warning(f"Could not read session memory for session {session_id} and user {user_id}: {e}")

    return all_messages


def _format_session_messages(all_messages: List[dict], max_messages: int = 30, max_chars: int = 12500) -> str:
    """Format collected session messages into a context string.

    Args:
        all_messages: List of messages with timestamps
        max_messages: Maximum number of recent messages to include (default 30)
        max_chars: Maximum character length for the context (default 12500)

    Returns:
        Formatted context string
    """
    import logging

    logger = logging.getLogger(__name__)

    # Sort messages by timestamp (descending - most recent first)
    all_messages.sort(key=lambda x: x["timestamp"], reverse=True)


    # Take only the most recent messages (max_messages)
    recent_messages = all_messages[:max_messages]

    # Format messages into context string
    context_text = "\n"
  
    for msg_info in recent_messages[::-1]:
        msg = msg_info["message"]
        role = msg.role

        content_text = " ".join(
            item.text for item in msg.content
        ).strip()
        
        clean_text = content_text.replace('\n', ' ')
        context_text += f"[{role}]: {clean_text}\n"



    # Truncate to max_chars if needed
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]
    
    return context_text






async def get_last_message_ts(in_memory: Optional['ReMeInMemoryMemory'] = None):
    """
    Attempts to get the timestamp of the last message from in-memory storage.
    If that fails, it falls back to using chat manager to retrieve the timestamp.

    Args:
        in_memory: Optional in-memory instance to access session history

    Returns:
        The timestamp of the last message from session or None if not found
    """
    import logging
    logger = logging.getLogger(__name__)

    # First try getting the last message from in_memory
    if in_memory:
        try:
            messages = await in_memory.get_memory(
                exclude_mark=None,
                prepend_summary=False,
            )

            if messages:
                # Return the timestamp of the last message
                last_msg = messages[-1]
                if hasattr(last_msg, 'timestamp'):
                    return last_msg.timestamp
        except Exception:
            # If reading from in_memory fails, proceed to fallback method
            pass


    try:
        from ...app.agent_context import get_current_agent_id
        from ...app.multi_agent_manager import MultiAgentManager

        # Get current agent workspace to access chat manager
        multi_agent_manager = MultiAgentManager()
        agent_id = get_current_agent_id()
        workspace = await multi_agent_manager.get_agent(agent_id)

        # Get chats using chat manager
        chats = await workspace.chat_manager.list_chats()

        # Find the session with the most recent updated_at timestamp
        latest_update_ts = None
        for session in chats:
            # Convert the timestamp to datetime
            updated_at_dt = session.updated_at
            updated_at_dt = ensure_tz_aware(updated_at_dt)
            updated_at_timestamp = updated_at_dt.timestamp()

            if latest_update_ts is None or updated_at_timestamp > latest_update_ts:
                latest_update_ts = updated_at_timestamp

        return latest_update_ts

    except Exception as e:
        logger.warning(f"Could not read chats through chat manager for fallback: {e}")
        return None