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
        return [f"{chat.user_id}_{chat.session_id}" for chat in chats_file.chats]




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

    if not session_ids_from_chats:
        print("No sessions found in any of the storage locations.")
        return None, None

    # Get activity times for sessions that have associated files
    session_activities = await get_session_activity_times()


    combined_activities = {}

    # Add file-based activities
    for session_id, activity_time in session_activities.items():
        combined_activities[session_id] = activity_time

    # If we still don't have activity times for some sessions, assign current time
    for session_id in session_ids_from_chats:
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