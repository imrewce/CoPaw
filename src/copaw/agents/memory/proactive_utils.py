# -*- coding: utf-8 -*-
"""
Utility functions for proactive messaging features.
Kept self-contained within the CoPaw agents/memory directory.
"""

import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import json
import os


async def get_all_session_ids_from_json_files(session_dir: str = None) -> List[str]:
    """
    Retrieve all session IDs by scanning JSON session files in the session directory.

    Args:
        session_dir: Directory containing session JSON files. If None, uses default.

    Returns:
        List of session IDs found in the session directory.
    """
    from ...constant import WORKING_DIR
    if session_dir is None:
        session_dir = str(WORKING_DIR / "sessions")

    session_path = Path(session_dir)
    if not session_path.exists():
        return []

    session_ids = []
    for file_path in session_path.glob("*.json"):
        filename = file_path.stem
        # Use the complete filename as session_id (e.g., "default_1773649538196")
        session_ids.append(filename)

    return session_ids


async def get_all_session_ids_from_chats_file(chats_file_path: str = None) -> List[str]:
    """
    Retrieve all session IDs from the chats.json file.

    Args:
        chats_file_path: Path to chats.json file. If None, uses default.

    Returns:
        List of session IDs found in chats file.
    """
    from ...constant import WORKING_DIR
    if chats_file_path is None:
        chats_file_path = WORKING_DIR / "chats.json"

    chats_path = Path(chats_file_path)
    if not chats_path.exists():
        return []

    import json
    # Primary attempt: use ChatsFile model
    from ...app.runner.models import ChatsFile
    with open(chats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        chats_file = ChatsFile.model_validate(data)
        return [chat.session_id for chat in chats_file.chats]




async def get_session_activity_times(session_dir: str = None) -> Dict[str, float]:
    """
    Get the last activity time for each session based on file modification times.

    Args:
        session_dir: Directory containing session JSON files. If None, uses default.

    Returns:
        Dictionary mapping session_id to last activity timestamp.
    """
    from ...constant import WORKING_DIR
    if session_dir is None:
        session_dir = str(WORKING_DIR / "sessions")

    session_path = Path(session_dir)
    if not session_path.exists():
        return {}

    session_activities = {}

    for file_path in session_path.glob("*.json"):
        filename = file_path.stem
        # Use the complete filename as session_id
        # Get file modification time (last activity time)
        mtime = file_path.stat().st_mtime
        session_activities[filename] = mtime

    return session_activities


async def find_most_recent_active_session() -> Tuple[Optional[str], Optional[datetime]]:
    """
    Find the session that had the most recent activity.

    Returns:
        Tuple of (most_recent_session_id, last_activity_datetime) or (None, None) if no sessions found.
    """
    try:
        # Import the session manager to access the active chats registry
        from ...app.runner.repo.base import RepoManager

        # Get all chats to find the most recently active one
        chats_file = RepoManager.get_instance().get_chats()

        if not chats_file.chats:
            print("No sessions found in chats file.")
            return None, None

        # Find the most recently active chat based on timestamp
        most_recent_chat = None
        most_recent_time = 0

        for chat in chats_file.chats:
            # Check the last updated time of each chat
            if hasattr(chat, 'update_time') and chat.update_time:
                update_time = chat.update_time
            elif hasattr(chat, 'create_time') and chat.create_time:
                update_time = chat.create_time
            else:
                # If no explicit time, try to extract from the session file
                try:
                    # Construct the expected session file path
                    # Use session_id to find corresponding file
                    session_file_pattern = str(Path(".") / "sessions" / f"{chat.session_id}.json")
                    session_file_path = Path(session_file_pattern)

                    # Try common patterns for session files
                    possible_paths = [
                        Path(f"../sessions/{chat.session_id}.json"),
                        Path(f"sessions/{chat.session_id}.json"),
                        Path(f"../sessions/{chat.session_id}_*.json"),  # wildcard pattern
                        Path(f"sessions/{chat.session_id}_*.json")
                    ]

                    session_file_path = None
                    for path_pattern in possible_paths:
                        if "*" in str(path_pattern):
                            # Handle wildcards by finding matching files
                            matching_files = list(Path(".").parent.glob(str(path_pattern))) or list(Path(".").glob(str(path_pattern)))
                            if matching_files:
                                # Pick the most recently modified file
                                session_file_path = max(matching_files, key=lambda x: x.stat().st_mtime)
                                break
                        elif path_pattern.exists():
                            session_file_path = path_pattern
                            break

                    if session_file_path and session_file_path.exists():
                        update_time = session_file_path.stat().st_mtime
                    else:
                        # Use current time as fallback
                        update_time = time.time()
                except:
                    update_time = time.time()

            if update_time and update_time > most_recent_time:
                most_recent_time = update_time
                most_recent_chat = chat

        if most_recent_chat:
            # Return the complete session ID that matches the filename (without .json)
            # Based on the fix suggestion: construct filename as {user_id}_{session_id}
            full_session_id = build_full_session_id(most_recent_chat)

            last_activity_datetime = datetime.fromtimestamp(most_recent_time)
            return full_session_id, last_activity_datetime

    except ImportError:
        print("SessionManager or RepoManager not available, using fallback")
    except Exception as e:
        print(f"Error accessing session manager to find most recent active session: {e}")
        # Fallback to original implementation
        pass

    # Original implementation as fallback
    # Get session IDs from multiple sources
    session_ids_from_files = await get_all_session_ids_from_json_files()

    # Attempt to get session IDs from chats file
    session_ids_from_chats = []
    try:
        session_ids_from_chats = await get_all_session_ids_from_chats_file()
    except Exception:
        # If primary method fails, try manual parsing
        from ...constant import WORKING_DIR
        chats_file_path = WORKING_DIR / "chats.json"
        chats_path = Path(chats_file_path)

        if chats_path.exists():
            try:
                import json
                with open(chats_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chats' in data:
                        for chat in data['chats']:
                            session_id = chat.get('session_id', '')
                            user_id = chat.get('user_id', 'default')

                            # Build proper session ID based on user_id and session_id
                            if user_id == "default":
                                full_session_id = f"default_{session_id}"
                            else:
                                full_session_id = f"{user_id}_{session_id}"

                            session_ids_from_chats.append(full_session_id)
            except Exception:
                pass

    # Combine all session IDs
    all_session_ids = set(session_ids_from_files + session_ids_from_chats )

    if not all_session_ids:
        print("No sessions found in any of the storage locations.")
        return None, None

    # Get activity times for sessions that have associated files
    session_activities = await get_session_activity_times()

    # For sessions without files, we can estimate activity from push store
    push_timestamps = {}

    # Get more detailed information from push store to determine activity times
    from ...app.console_push_store import get_recent
    all_messages = await get_recent(max_age_seconds=86400)  # Last 24 hours

    for msg in all_messages:
        session_id = msg.get('session_id')
        timestamp = msg.get('ts', time.time())  # Use timestamp from message or current time
        if session_id:
            if session_id not in push_timestamps or timestamp > push_timestamps[session_id]:
                push_timestamps[session_id] = timestamp

    # Combine all timestamps
    combined_activities = {}

    # Add file-based activities
    for session_id, activity_time in session_activities.items():
        combined_activities[session_id] = activity_time

    # Add push store activities for sessions not found in files
    for session_id, activity_time in push_timestamps.items():
        if session_id not in combined_activities or activity_time > combined_activities[session_id]:
            combined_activities[session_id] = activity_time

    # If we still don't have activity times for some sessions, assign current time
    for session_id in all_session_ids:
        if session_id not in combined_activities:
            combined_activities[session_id] = time.time()

    if not combined_activities:
        print("No activity records found for any session.")
        return None, None

    # Find the session with the most recent activity
    most_recent_session = max(combined_activities, key=combined_activities.get)
    last_activity_time = combined_activities[most_recent_session]
    last_activity_datetime = datetime.fromtimestamp(last_activity_time)

    return most_recent_session, last_activity_datetime


def get_recent_memory_file_paths(working_dir: str, num_days: int = 2) -> List[str]:
    """Get paths to the most recent memory files from the working directory."""
    memory_dir = os.path.join(working_dir, "memory")
    if not os.path.exists(memory_dir):
        return []

    memory_files = []
    for filename in os.listdir(memory_dir):
        if filename.endswith('.md'):
            try:
                date_part = filename.split('.')[0]  # Remove .md extension
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                file_path = os.path.join(memory_dir, filename)
                memory_files.append((file_date, file_path))
            except ValueError:
                # Skip files that don't match the date pattern
                continue

    # Sort by date descending (most recent first)
    memory_files.sort(key=lambda x: x[0], reverse=True)

    # Return only the file paths of the most recent N days
    return [file_path for _, file_path in memory_files[:num_days]]


def build_full_session_id(chat) -> str:
    """Build full session ID from chat object (matches filename without .json)."""
    session_id = getattr(chat, 'session_id', '')
    user_id = getattr(chat, 'user_id', 'default')
    print("#############################################")
    print(user_id)
    print("#############################################")

    if user_id == "default":
        return f"default_{session_id}"
    else:
        return f"{user_id}_{session_id}"


def build_proactive_memory_context(
    session_context: str,
    file_memories: List[str],
    max_chars: int = 100000
) -> str:
    """Build a combined memory context for proactive agent."""
    combined_context = "[SESSION CONTEXT]\n"
    combined_context += session_context + "\n\n"

    # Add file memories in chronological order (most recent first)
    for i, file_memory in enumerate(file_memories):
        if i == 0:
            combined_context += "[FILE MEMORY - most recent date]\n"
        elif i == 1:
            combined_context += "[FILE MEMORY - previous date]\n"
        else:
            combined_context += f"[FILE MEMORY - date {i+1}]\n"
        combined_context += file_memory + "\n\n"

    # Truncate if necessary, prioritizing session context
    if len(combined_context) > max_chars:
        session_section = "[SESSION CONTEXT]\n" + session_context + "\n\n"
        if len(session_section) > max_chars:
            return session_section[:max_chars]

        remaining_chars = max_chars - len(session_section)
        file_memories_section = combined_context[len(session_section):]
        if len(file_memories_section) > remaining_chars:
            file_memories_section = file_memories_section[:remaining_chars]
        combined_context = session_section + file_memories_section

    return combined_context



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