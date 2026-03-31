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



def build_proactive_memory_context(
    agent_workspace_path: Optional[str] = None,
    max_session_messages: int = 30,
    max_session_chars: int = 12500
) -> str:
    """Build a combined memory context for proactive agent from active agent workspace.

    Args:
        agent_workspace_path: Path to agent workspace - must be provided, no fallback to WORKING_DIR
        max_session_messages: Maximum number of recent messages to include from sessions (default 30)
        max_session_chars: Maximum character length for the session context (default 12500)
    """
    import logging

    # agent_workspace_path must be provided - no fallback to WORKING_DIR
    if agent_workspace_path is None:
        raise ValueError("agent_workspace_path must be provided and cannot be None")

    combined_context = "[SESSION CONTEXT]\n"

    # Read from active agent workspace - sessions and chats
        # Read chat sessions metadata
    sessions_to_read = _read_chat_sessions_metadata(agent_workspace_path)

    if not sessions_to_read:
        return ""

    # Filter to recent sessions
    filtered_sessions = _filter_recent_sessions(sessions_to_read)

    # Collect messages from session files
    all_messages = _collect_session_messages(agent_workspace_path, filtered_sessions)

    # Format messages into context string
    session_context = ""
    if all_messages:
        session_context = _format_session_messages(all_messages, max_session_messages, max_session_chars)

    if session_context:
        combined_context += session_context + "\n\n"

    return combined_context


def _process_session_file(session_file, mod_time) -> List[dict]:
    """Process a single session file and return a list of messages.

    Args:
        session_file: Path to the session JSON file
        mod_time: Modification time of the file

    Returns:
        List of messages with timestamps
    """
    import logging
    from datetime import datetime, timezone
    import json
    logger = logging.getLogger(__name__)

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # Different session files might have different structures
        # Try common structures used in the application
        messages = []

        # Helper function to process a single message
        def process_message(msg, default_mod_time):
            if not msg or not isinstance(msg, dict):
                return None

            # Filter out messages with role "system" and types "thinking"/"tool_use"
            role = msg.get("role", "")

            # Skip system messages
            if role == "system":
                return None

            # Check if content contains "thinking" or "tool_use" types
            content = msg.get("content", [])
            if isinstance(content, list):
                should_skip = False
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type in ["thinking", "tool_use"]:
                            should_skip = True
                            break
                if should_skip:
                    return None

            # Add timestamp to message for sorting later
            timestamp_str = msg.get("timestamp", "")
            try:
                if isinstance(timestamp_str, str) and timestamp_str:
                    # Parse timestamp - might be in ISO format
                    parsed_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00").replace("+00:00", ""))
                    timestamp = ensure_tz_aware(parsed_timestamp)
                else:
                    # Use file modification time if no timestamp in message
                    timestamp = ensure_tz_aware(default_mod_time)
            except:
                timestamp = ensure_tz_aware(default_mod_time)

            return {
                "message": msg,
                "timestamp": timestamp
            }

        # Structure from the example: agent.memory.content as list of [user_msg, assistant_msg]
        content_list = session_data["agent"]["memory"].get("content", [])

        for item_pair in content_list:
            if isinstance(item_pair, list) and len(item_pair) >= 1:
                # First element might be user message
                user_msg = item_pair[0] if len(item_pair) > 0 and item_pair[0] else None
                # Second element might be assistant response
                assistant_resp = item_pair[1] if len(item_pair) > 1 and item_pair[1] else []

                # Process user message if exists and is valid
                processed_user_msg = process_message(user_msg, mod_time)
                if processed_user_msg:
                    messages.append(processed_user_msg)

                # Process assistant response if exists
                if isinstance(assistant_resp, list):
                    for resp_item in assistant_resp:
                        processed_resp = process_message(resp_item, mod_time)
                        if processed_resp:
                            messages.append(processed_resp)


        # Sort messages by timestamp (descending - most recent first)
        messages.sort(key=lambda x: x["timestamp"], reverse=True)

        logger.info(f"Processed session file: {session_file.name}, filtered out messages containing thinking/tool_use/system roles")

        return messages

    except Exception as e:
        logger.warning(f"Could not read session file {session_file}: {e}")
        return []


def load_json_safely(json_input) -> Optional[dict]:
    """Safely parse JSON string, returning None if invalid."""
    import logging
    logger = logging.getLogger(__name__)

    # Handle if input is already a dictionary
    if isinstance(json_input, dict):
        return json_input

    # Handle if input is not a string
    if not isinstance(json_input, str):
        json_input = str(json_input)

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


def _read_chat_sessions_metadata(agent_workspace_path: str) -> List[dict]:
    """Read chat sessions metadata from chats.json file.

    Args:
        agent_workspace_path: Path to the agent workspace

    Returns:
        List of session metadata dictionaries
    """
    import logging
    from datetime import datetime, timezone
    import json
    from pathlib import Path

    logger = logging.getLogger(__name__)

    chats_file_path = Path(agent_workspace_path) / "chats.json"
    sessions_to_read = []

    if not chats_file_path.exists():
        logger.warning(f"chats.json not found at {chats_file_path}")
        return []

    try:
        with open(chats_file_path, "r", encoding="utf-8") as f:
            chats_data = json.load(f)

        # Extract session info from chats.json
        if 'chats' in chats_data and chats_data['chats']:
            for chat in chats_data['chats']:
                user_id = chat.get('user_id', '')
                session_id = chat.get('session_id', '')

                # Format file name according to requirements (replace ':' with '--')
                formatted_user_id = user_id.replace(':', '--')
                formatted_session_id = session_id.replace(':', '--')
                filename = f"{formatted_user_id}_{formatted_session_id}.json"

                # Get modification time from chats.json
                updated_at_str = chat.get('updated_at')
                if updated_at_str:
                    try:
                        updated_at_dt = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                        updated_at_dt = ensure_tz_aware(updated_at_dt)
                    except ValueError:
                        updated_at_dt = ensure_tz_aware(datetime.fromtimestamp(chats_file_path.stat().st_mtime))
                else:
                    updated_at_dt = ensure_tz_aware(datetime.fromtimestamp(chats_file_path.stat().st_mtime))

                sessions_to_read.append({
                    'filename': filename,
                    'user_id': user_id,
                    'session_id': session_id,
                    'mod_time': updated_at_dt
                })

    except Exception as e:
        logger.warning(f"Could not read chats.json file: {e}")

    return sessions_to_read


def _filter_recent_sessions(sessions_to_read: List[dict], days: int = 3) -> List[dict]:
    """Filter sessions to only include recent ones or return most recent ones if none are recent.

    Args:
        sessions_to_read: List of session metadata
        days: Number of days to look back (default 3)

    Returns:
        Filtered list of sessions
    """
    from datetime import datetime, timedelta
    import logging

    logger = logging.getLogger(__name__)

    # Filter sessions based on time (last 3 days)
    filtered_sessions = []
    three_days_ago = ensure_tz_aware(datetime.now()) - timedelta(days=days)

    for session_info in sessions_to_read:
        if session_info['mod_time'] >= three_days_ago:
            filtered_sessions.append(session_info)

    # If no recent sessions, get the 3 most recently modified
    if not filtered_sessions:
        sessions_to_read.sort(key=lambda x: x['mod_time'], reverse=True)
        filtered_sessions = sessions_to_read[:3]
    else:
        filtered_sessions.sort(key=lambda x: x['mod_time'], reverse=True)

    return filtered_sessions


def _collect_session_messages(agent_workspace_path: str, filtered_sessions: List[dict]) -> List[dict]:
    """Collect all messages from session files.

    Args:
        agent_workspace_path: Path to the agent workspace
        filtered_sessions: List of filtered session metadata

    Returns:
        List of messages with timestamps
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    all_messages = []  # Collect all messages to be sorted later

    # Now read the corresponding session files from sessions directory
    sessions_dir = Path(agent_workspace_path) / "sessions"
    if sessions_dir.exists():
        for session_info in filtered_sessions:
            session_file_path = sessions_dir / session_info['filename']

            if session_file_path.exists():
                try:
                    # Pass the modification time from chats.json - returns list of messages
                    session_messages = _process_session_file(session_file_path, session_info['mod_time'])
                    if session_messages:
                        all_messages.extend(session_messages)  # Extend instead of append
                except Exception as e:
                    logger.warning(f"Could not read session file {session_file_path}: {e}")
            else:
                logger.warning(f"Session file does not exist: {session_file_path}")

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
    context_text = "[IN-MEMORY SESSION HISTORY]\n"
    for msg_info in recent_messages:
        msg_data = msg_info["message"]
        role = msg_data.get("role", "unknown")
        content = msg_data.get("content", "")

        # Format content appropriately
        if isinstance(content, list):
            content_text = " ".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
        else:
            content_text = str(content)

        content_text = content_text.replace("\n", " ")
        context_text += f"[{role}]: {content_text}\n"

    context_text += "\n"

    # Truncate to max_chars if needed
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    return context_text






def get_last_message_ts(in_memory: Optional['ReMeInMemoryMemory'] = None):
    """
    Attempts to get the timestamp of the last message from in-memory storage.
    If that fails, it falls back to using build_proactive_memory_context to retrieve the timestamp.

    Args:
        session_id: The session identifier
        in_memory: Optional in-memory instance to access session history

    Returns:
        The timestamp of the last message from session or None if not found
    """
    import asyncio
    import logging
    logger = logging.getLogger(__name__)

    async def _async_get_last_message_ts():
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
            from datetime import datetime
            import json
            from pathlib import Path

            # Get current agent workspace to locate chats.json
            multi_agent_manager = MultiAgentManager()
            agent_id = get_current_agent_id()
            workspace = await multi_agent_manager.get_agent(agent_id)

            # Look for chats.json in the workspace directory
            chats_file_path = Path(workspace.workspace_dir) / "chats.json"

            if chats_file_path.exists():
                with open(chats_file_path, 'r', encoding='utf-8') as f:
                    chats_data = json.load(f)

                # Find the session with the most recent updated_at timestamp
                latest_update_ts = None
                if 'chats' in chats_data and chats_data['chats']:
                    for session in chats_data['chats']:
                        updated_at_str = session.get('updated_at')
                        if updated_at_str:
                            # Convert the timestamp string to datetime
                            updated_at_dt = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                            # Ensure both are timezone-aware for comparison
                            updated_at_dt = ensure_tz_aware(updated_at_dt)
                            updated_at_timestamp = updated_at_dt.timestamp()

                            if latest_update_ts is None or updated_at_timestamp > latest_update_ts:
                                latest_update_ts = updated_at_timestamp

                return latest_update_ts

        except Exception as e:
            logger.warning(f"Could not read chats.json for fallback: {e}")
            return None

        return None

    # Since this function is intended to be synchronous but uses async operations internally,
    # we need to run the async function using asyncio.run
    # However, this can cause issues if there's already a running event loop
    try:
        # Check if there's an existing event loop
        asyncio.get_running_loop()
        # If we're in an async context, return a coroutine that the caller can await
        import functools
        @functools.wraps(_async_get_last_message_ts)
        async def wrapper():
            return await _async_get_last_message_ts()
        return wrapper()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(_async_get_last_message_ts())