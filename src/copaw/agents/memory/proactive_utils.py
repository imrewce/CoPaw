# -*- coding: utf-8 -*-
"""Utility functions for proactive conversation feature."""

import os
from datetime import datetime, timedelta
from typing import List, Optional
import json


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


def calculate_elapsed_minutes(timestamp) -> float:
    """Calculate minutes elapsed since the given timestamp."""
    if timestamp is None:
        return float('inf')  # Represent no timestamp as infinite elapsed time
    return (datetime.now() - timestamp).total_seconds() / 60.0


def is_valid_idle_minutes(minutes: str) -> bool:
    """Validate if the provided string represents valid idle minutes."""
    try:
        mins = int(minutes)
        return mins > 0
    except ValueError:
        return False


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