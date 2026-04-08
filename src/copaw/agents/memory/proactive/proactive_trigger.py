# -*- coding: utf-8 -*-
"""Trigger logic for proactive conversation feature."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any, TYPE_CHECKING

from agentscope.message import Msg

from ....app.agent_context import get_current_agent_id
from ....app.multi_agent_manager import MultiAgentManager
from .proactive_types import ProactiveConfig
from .proactive_responder import generate_proactive_response
from .proactive_utils import get_last_message_ts

if TYPE_CHECKING:
    from ..reme_light_memory_manager import ReMeLightMemoryManager
    from reme.memory.file_based import ReMeInMemoryMemory

logger = logging.getLogger(__name__)


def ensure_tz_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware, convert naive datetime to UTC."""
    if dt.tzinfo is None:
        # Assume naive datetime is in UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Global storage for proactive configurations per session
# In a real implementation, this would be tied to the session lifecycle
proactive_configs: Dict[str, ProactiveConfig] = {}
proactive_tasks: Dict[str, asyncio.Task] = {}  # Track active trigger loops (should NOT be cancelled on user interaction)
proactive_responder_tasks: Dict[str, asyncio.Task] = {}  # Track active responder tasks (should be cancelled on user interaction)

# Store session references globally so they can be accessed during proactive trigger
session_references: Dict[str, Dict[str, Any]] = {}


def enable_proactive_for_session(
    session_id: str,
    idle_minutes: int = 30,
    memory_manager: Optional['ReMeLightMemoryManager'] = None,
    in_memory: Optional['ReMeInMemoryMemory'] = None,
    channel_manager: Optional[Any] = None,
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,

) -> str:
    """Enable proactive for the given session and start monitoring.

    Args:
        session_id: Unique identifier for the session
        idle_minutes: Number of minutes of idle time before proactive trigger
        memory_manager: Optional memory manager instance to access real memory
        in_memory: Optional in-memory instance to access session history
        channel_manager: Optional channel manager to send proactive messages
        channel_id: Optional channel ID for the session
        user_id: Optional user ID for the session

    Returns:
        Status message confirming activation
    """
    global proactive_configs

    config = ProactiveConfig(
        enabled=True,
        idle_minutes=idle_minutes,
        last_user_interaction=datetime.now(timezone.utc),  # This will be updated when user interacts
        mode_enabled_time=datetime.now(timezone.utc)  # Record when the proactive mode was enabled
    )
    proactive_configs[session_id] = config

    # Store session references for real memory access and message sending

    if memory_manager:
        session_references['memory_manager'] = memory_manager
    if in_memory:
        session_references['in_memory'] = in_memory
    if channel_manager:
        session_references['channel_manager'] = channel_manager
    if channel_id:
        session_references['channel_id'] = channel_id
    if user_id:
        session_references['user_id'] = user_id

    # Store the session references globally
    session_references[session_id] = session_id

    # Start the proactive trigger loop if not already running for this session
    if session_id not in proactive_tasks or proactive_tasks[session_id].done():
        task = asyncio.create_task(_run_trigger_loop(session_id))
        proactive_tasks[session_id] = task

    return f"Proactive mode enabled with {idle_minutes} minute idle threshold."


async def _run_trigger_loop(session_id: str) -> None:
    """Internal function to run the trigger loop."""
    try:
        await proactive_trigger_loop(session_id)
    except Exception as e:
        logger.error(f"Error in proactive trigger loop for session {session_id}: {e}")


class ActiveAgentMonitor:
    """Monitor the active agent to detect if it is busy processing user requests."""

    async def is_agent_busy(self, agent_id: str = None) -> bool:
        """Check if the agent is currently busy processing tasks."""
        # Use the provided agent_id or get the current active agent ID
        if agent_id is None:
            agent_id = get_current_agent_id()

        try:
            # Get the multi-agent manager by creating a new instance
            multi_agent_manager = MultiAgentManager()
            workspace = await multi_agent_manager.get_agent(agent_id)

            # Check if the workspace has a task tracker and if it has active tasks
            if hasattr(workspace, 'task_tracker') and workspace.task_tracker:
                active_tasks = await workspace.task_tracker.list_active_tasks()
                return len(active_tasks) > 0

            return False
        except Exception as e:
            logger.error(f"Error checking if agent {agent_id} is busy: {e}")
            return False

# Global instance of the ActiveAgentMonitor
active_agent_monitor = ActiveAgentMonitor()


async def proactive_trigger_loop(session_id: str) -> None:
    """Background loop that polls every 30s to detect idle periods.

    Args:
        session_id: Unique identifier for the session
    """
    global proactive_configs

    # Track when we last tried to trigger proactive for this session
    last_trigger_attempt: Optional[datetime] = None

    while True:
        try:
            await asyncio.sleep(30)  # Sleep for 30 seconds

            if session_id not in proactive_configs:
                continue

            config = proactive_configs[session_id]

            if not config.enabled:
                continue

            # Check if the active agent is busy - if so, skip triggering proactive
            if await active_agent_monitor.is_agent_busy():
                continue

            # Use the actual session's last user message time instead of config
            actual_last_user_time = await get_last_message_ts()

            # If we can't get the actual time from session, fall back to config
            if actual_last_user_time is not None:
                # Fix: Always use UTC when converting timestamp to ensure consistent timezone handling
                last_interaction_dt = datetime.fromtimestamp(actual_last_user_time, tz=timezone.utc)
            else:
                last_interaction_dt = config.last_user_interaction

            if last_interaction_dt is not None:
                # Ensure both datetimes are timezone-aware for comparison
                last_interaction_tz_aware = ensure_tz_aware(last_interaction_dt)
                current_time = datetime.now(timezone.utc)
                elapsed_minutes = (current_time - last_interaction_tz_aware).total_seconds() / 60.0

                # Additional check: ensure that the time since proactive mode was enabled
                # is at least the configured idle time for the first trigger
                if config.mode_enabled_time:
                    # Ensure both datetimes are timezone-aware for comparison
                    mode_enabled_time_tz_aware = ensure_tz_aware(config.mode_enabled_time)
                    current_time = datetime.now(timezone.utc)
                    time_since_mode_enabled = (current_time - mode_enabled_time_tz_aware).total_seconds() / 60.0
                    should_trigger = elapsed_minutes >= config.idle_minutes and time_since_mode_enabled >= config.idle_minutes
                else:
                    # Fallback to original logic if mode_enabled_time is not set
                    should_trigger = elapsed_minutes >= config.idle_minutes

                if should_trigger:
                    # Check multiple conditions to prevent duplicate triggers:
                    # 1. No proactive task currently running
                    # 2. Sufficient cooldown from last trigger attempt (to prevent rapid retries)
                    # 3. Last message in session is not already a proactive message (unless the last interaction was before proactive mode started)
                    # 4. Sufficient cooldown since last proactive was sent (to prevent rapid-fire triggering)


                    if (config.running_task_id is None and  # No proactive task currently running
                        (last_trigger_attempt is None or
                         (datetime.now(timezone.utc) - ensure_tz_aware(last_trigger_attempt)).total_seconds() > 120)):  # At least 2 min cooldown from last attempt

                        # Check if the last user interaction was before the proactive mode was enabled
                        # If so, skip the proactive message check to prevent blocking new triggers
                        last_interaction_was_before_mode_enabled = (
                            last_interaction_tz_aware <= mode_enabled_time_tz_aware
                        )


                        # Double-check: is the last message already proactive?
                        # Skip this check if the last interaction was before proactive mode was enabled
                        if last_interaction_was_before_mode_enabled or not await is_last_message_proactive():
                            logger.info("Triggering proactive response now")
                            # Mark that we're attempting to trigger proactive response
                            last_trigger_attempt = datetime.now(timezone.utc)

                            # Trigger proactive response
                            config.running_task_id = f"proactive_{datetime.now(timezone.utc).timestamp()}"

                            # Get session references passed through enable_proactive_for_session function
                            # We need to access them from a different mechanism since they're not directly available here
                            # For now, we'll need to fetch them again from wherever they're stored


                            try:
                                # Create and track the responder task separately from the trigger loop
                                responder_task = asyncio.create_task(
                                    generate_proactive_response(
                                        session_id,
                                        in_memory=session_references.get('in_memory')
                                    )
                                )
                                proactive_responder_tasks[session_id] = responder_task

                                proactive_msg = await responder_task

                                # Log the proactive message if it was generated
                                if proactive_msg:
                                    logger.info(f"Proactive message generated for session {session_id}: {str(proactive_msg)[:100]}...")

                                # Clean up the responder task reference after completion
                                if session_id in proactive_responder_tasks:
                                    del proactive_responder_tasks[session_id]


                            except Exception as e:
                                logger.error(f"Error in proactive responder for session {session_id}: {e}")
                                # Clean up the responder task reference on error
                                if session_id in proactive_responder_tasks:
                                    del proactive_responder_tasks[session_id]
                            finally:
                                # Clear the running task ID
                                if session_id in proactive_configs:
                                    proactive_configs[session_id].running_task_id = None
                                    # Update last proactive sent time to prevent immediate re-triggering
                        else:
                            # Update the trigger attempt time even if we skip triggering due to proactive message check
                            last_trigger_attempt = datetime.now(timezone.utc)
        except asyncio.CancelledError:
            logger.info(f"Proactive trigger loop cancelled for session {session_id}")
            break
        except Exception as e:
            logger.error(f"Error in proactive trigger loop: {e}")
            # Continue loop despite errors to maintain resilience


async def is_last_message_proactive() -> bool:
    """Check if the last message in session was a proactive message.
    Checks if the last message starts with [PROACTIVE].

    Returns:
        True if last message was proactive, False otherwise
    """
    from agentscope.memory import InMemoryMemory
    from ....app.runner.utils import agentscope_msg_to_message

    # Use the function imported at the top of the file
    active_agent_id = get_current_agent_id()
    multi_agent_manager = MultiAgentManager()
    workspace = await multi_agent_manager.get_agent(active_agent_id)

    try:
        # Get chats using chat manager instead of reading chats.json directly
        chats = await workspace.chat_manager.list_chats()

        # Find the session with the most recent updated_at timestamp
        latest_updated_session = None
        latest_update_time = None
        for session in chats:
            updated_at_dt = session.updated_at

            # Ensure consistent timezone-aware comparison
            if latest_update_time is None or ensure_tz_aware(updated_at_dt) > ensure_tz_aware(latest_update_time):
                latest_update_time = updated_at_dt
                latest_updated_session = session

        if latest_updated_session:
            # Now we have the latest updated session, we can check the session's messages
            # Load the session's memory using InMemoryMemory
            session_id = latest_updated_session.session_id
            user_id = latest_updated_session.user_id

            if session_id:
                # Get the session state dictionary
                state = await workspace.runner.session.get_session_state_dict(
                    session_id,
                    user_id,
                )

                if not state:
                    return False

                # Extract memory data from the state
                memories_data = state.get("agent", {}).get("memory", [])

                # Create InMemoryMemory instance and load state
                memory = InMemoryMemory()
                memory.load_state_dict(memories_data)

                # Get memory messages
                messages = await memory.get_memory()

                # Convert to serializable format
                serializable_messages = agentscope_msg_to_message(messages)

                # Check if the latest message contains [PROACTIVE] markers
                if serializable_messages:
                    # Get the latest message content
                    latest_msg = serializable_messages[-1]  # Latest message is at the end

                    # Access content from the Message object directly
                    # The Message object has a contents property which is a list of content items
                    contents = getattr(latest_msg, 'contents', [])

                    # Process the contents
                    if contents and isinstance(contents, list):
                        for content_item in contents:
                            # Check if content_item has text attribute (for TextContent)
                            if hasattr(content_item, 'text'):
                                if "[PROACTIVE]" in content_item.text:
                                    logger.info("Last Proactive Message Unresponded")
                                    return True
                            # Check if content_item has data attribute (for DataContent)
                            elif hasattr(content_item, 'data'):
                                if isinstance(content_item.data, str) and "[PROACTIVE]" in content_item.data:
                                    logger.info("Last Proactive Message Unresponded")
                                    return True
                                elif isinstance(content_item.data, dict):
                                    # If data is a dict, check its string representation
                                    if "[PROACTIVE]" in str(content_item.data):
                                        logger.info("Last Proactive Message Unresponded")
                                        return True
                            # As a fallback, check if content_item is string-like
                            else:
                                if "[PROACTIVE]" in str(content_item):
                                    logger.info("Last Proactive Message Unresponded")
                                    return True
                    # If contents is empty or not a list, check the message's string representation
                    else:
                        if "[PROACTIVE]" in str(latest_msg):
                            logger.info("Last Proactive Message Unresponded")
                            return True

    except Exception as e:
        logger.warning(f"Could not check if last message was proactive: {e}")
        return False

    return False

        