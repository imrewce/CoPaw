# -*- coding: utf-8 -*-
"""Trigger logic for proactive conversation feature."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any, TYPE_CHECKING
from .proactive_types import ProactiveConfig
from .proactive_responder import generate_proactive_response
from agentscope.message import Msg

if TYPE_CHECKING:
    from ..memory import MemoryManager
    from reme.memory.file_based import ReMeInMemoryMemory

logger = logging.getLogger(__name__)

# Global storage for proactive configurations per session
# In a real implementation, this would be tied to the session lifecycle
proactive_configs: Dict[str, ProactiveConfig] = {}
proactive_tasks: Dict[str, asyncio.Task] = {}  # Track active trigger loops (should NOT be cancelled on user interaction)
proactive_responder_tasks: Dict[str, asyncio.Task] = {}  # Track active responder tasks (should be cancelled on user interaction)
proactive_sessions: Dict[str, Dict[str, Any]] = {}  # Store session references (memory_manager, in_memory, etc.)


def enable_proactive_for_session(
    session_id: str,
    idle_minutes: int = 30,
    memory_manager: Optional['MemoryManager'] = None,
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
        last_user_interaction=datetime.now(),  # This will be updated when user interacts
        mode_enabled_time=datetime.now()  # Record when the proactive mode was enabled
    )
    proactive_configs[session_id] = config

    # Store session references for real memory access and message sending
    session_refs = {}
    if memory_manager:
        session_refs['memory_manager'] = memory_manager
    if in_memory:
        session_refs['in_memory'] = in_memory
    if channel_manager:
        session_refs['channel_manager'] = channel_manager
    if channel_id:
        session_refs['channel_id'] = channel_id
    if user_id:
        session_refs['user_id'] = user_id

    if session_refs:
        proactive_sessions[session_id] = session_refs

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

            # Use the actual session's last user message time instead of config
            actual_last_user_time = await get_last_message_ts(session_id)

            # If we can't get the actual time from session, fall back to config
            if actual_last_user_time is not None:
                last_interaction_dt = datetime.fromtimestamp(actual_last_user_time)
            else:
                last_interaction_dt = config.last_user_interaction

            if last_interaction_dt is not None:
                elapsed_minutes = (datetime.now() - last_interaction_dt).total_seconds() / 60.0

                # Additional check: ensure that the time since proactive mode was enabled
                # is at least the configured idle time for the first trigger
                if config.mode_enabled_time:
                    time_since_mode_enabled = (datetime.now() - config.mode_enabled_time).total_seconds() / 60.0
                    should_trigger = elapsed_minutes >= config.idle_minutes and time_since_mode_enabled >= config.idle_minutes
                else:
                    # Fallback to original logic if mode_enabled_time is not set
                    should_trigger = elapsed_minutes >= config.idle_minutes

                if should_trigger:
                    # Check multiple conditions to prevent duplicate triggers:
                    # 1. No proactive task currently running
                    # 2. Sufficient cooldown from last trigger attempt (to prevent rapid retries)
                    # 3. Last message in session is not already a proactive message
                    # 4. Sufficient cooldown since last proactive was sent (to prevent rapid-fire triggering)
                    

                    if (config.running_task_id is None and  # No proactive task currently running
                        (last_trigger_attempt is None or
                         (datetime.now() - last_trigger_attempt).total_seconds() > 60)):  # At least 1 min cooldown from last attempt

                        # Double-check: is the last message already proactive?
                        if not await is_last_message_proactive(session_id):
                            # Mark that we're attempting to trigger proactive response
                            last_trigger_attempt = datetime.now()

                            # Trigger proactive response
                            config.running_task_id = f"proactive_{datetime.now().timestamp()}"
                            try:
                                # Pass the session references if available
                                session_refs = proactive_sessions.get(session_id, {})

                                # Get access to the global app state to create a proper chat
                                chat_manager = session_refs.get('chat_manager')
                                if not chat_manager and 'channel_manager' in session_refs:
                                    # Try to get the global multi-agent manager to access the chat manager
                                    try:
                                        from ...app._app import app
                                        if hasattr(app.state, 'multi_agent_manager'):
                                            multi_agent_manager = app.state.multi_agent_manager
                                            # Try to get the default agent and its chat manager
                                            if hasattr(multi_agent_manager, 'agents'):
                                                default_agent = multi_agent_manager.agents.get("default")
                                                if default_agent and hasattr(default_agent, 'chat_manager'):
                                                    chat_manager = default_agent.chat_manager
                                    except Exception:
                                        chat_manager = None

                                # Create or ensure proactive chat exists for the specific session
                                if chat_manager:
                                    try:
                                        # Use the original session_id to ensure consistency with user's chat
                                        proactive_chat_id = f"proactive_{session_id}"
                                        # Check if proactive chat already exists for this session
                                        proactive_chat = await chat_manager.get_chat(proactive_chat_id)
                                        if not proactive_chat:
                                            # Create proactive chat for this session
                                            from ...app.runner.models import ChatSpec
                                            proactive_chat = await chat_manager.create_chat(ChatSpec(
                                                id=proactive_chat_id,
                                                name="Proactive Message",
                                                session_id=session_id,  # Use the original session_id
                                                user_id=session_refs.get('user_id', 'system'),
                                                channel=session_refs.get('channel_id', 'console'),
                                                meta={"type": "proactive", "original_session": session_id}
                                            ))
                                    except Exception as e:
                                        logger.warning(f"Could not create proactive chat for session {session_id}: {e}")
                                        # If creating chat fails, continue anyway
                                        pass

                                # Create and track the responder task separately from the trigger loop
                                responder_task = asyncio.create_task(
                                    generate_proactive_response(
                                        session_id,
                                        memory_manager=session_refs.get('memory_manager'),
                                        in_memory=session_refs.get('in_memory')
                                    )
                                )
                                proactive_responder_tasks[session_id] = responder_task

                                proactive_msg = await responder_task

                                # Clean up the responder task reference after completion
                                if session_id in proactive_responder_tasks:
                                    del proactive_responder_tasks[session_id]

                                if proactive_msg:
                                    # Send the proactive message to the original session, so it appears in the chat list
                                    await send_proactive_message(session_id=session_id, message=proactive_msg)
                            except asyncio.CancelledError:
                                logger.info(f"Proactive responder task cancelled for session {session_id}")
                                # Clean up the responder task reference if it was cancelled
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
                            last_trigger_attempt = datetime.now()
        except asyncio.CancelledError:
            logger.info(f"Proactive trigger loop cancelled for session {session_id}")
            break
        except Exception as e:
            logger.error(f"Error in proactive trigger loop: {e}")
            # Continue loop despite errors to maintain resilience


async def is_last_message_proactive(session_id: str) -> bool:
    """Check if the last message in session was a proactive message.

    Args:
        session_id: Unique identifier for the session

    Returns:
        True if last message was proactive, False otherwise
    """
    # First check if we have access to real session memory
    session_refs = proactive_sessions.get(session_id, {})
    in_memory = session_refs.get('in_memory')

    if in_memory:
        # Access the actual session memory to check last message
        try:
            messages = await in_memory.get_memory(
                exclude_mark=None,
                prepend_summary=False,
            )
            if messages:
                last_msg = messages[-1]
                # Check if the last message has proactive metadata
                if hasattr(last_msg, 'metadata') and last_msg.metadata:
                    return last_msg.metadata.get('is_proactive', False)
        except Exception as e:
            logger.warning(f"Could not access session memory for proactive check: {e}")

    # Fallback to the original implementation
    return False


async def get_last_message_ts(session_id: str) -> Optional[float]:
    """Get the timestamp of the last message in session.

    Args:
        session_id: Unique identifier for the session

    Returns:
        Timestamp of last message or None if not available
    """
    # First check if we have access to real session memory
    session_refs = proactive_sessions.get(session_id, {})
    in_memory = session_refs.get('in_memory')

    if in_memory:
        try:
            # Access the actual session memory to find last message
            messages = await in_memory.get_memory(
                exclude_mark=None,
                prepend_summary=False,
            )

            # Look for the most recent message with timestamp
            for msg in reversed(messages):
                # Return timestamp if available
                if hasattr(msg, 'timestamp'):
                    from datetime import datetime
                    # Convert string timestamp to datetime if needed
                    if isinstance(msg.timestamp, str):
                        try:
                            # Handle various timestamp formats
                            dt_obj = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                            return dt_obj.timestamp()
                        except:
                            # If parsing fails, return None
                            return None
                    elif isinstance(msg.timestamp, (float, int)):
                        return msg.timestamp
                elif hasattr(msg, 'create_time'):  # Some messages might use create_time instead
                    if isinstance(msg.create_time, str):
                        try:
                            dt_obj = datetime.fromisoformat(msg.create_time.replace('Z', '+00:00'))
                            return dt_obj.timestamp()
                        except:
                            return None
                    elif isinstance(msg.create_time, (float, int)):
                        return msg.create_time
        except Exception as e:
            logger.warning(f"Could not access session memory for user message timestamp: {e}")

    # Fallback to checking session activity times from proactive_utils
    try:
        from .proactive_utils import get_session_activity_times
        session_activities = await get_session_activity_times()
        if session_id in session_activities:
            return session_activities[session_id]
    except ImportError:
        logger.warning("Could not import get_session_activity_times from proactive_utils")
    except Exception as e:
        logger.warning(f"Could not get session activity times: {e}")

    # Fallback to the original implementation
    global proactive_configs

    if session_id in proactive_configs and proactive_configs[session_id].last_user_interaction:
        return proactive_configs[session_id].last_user_interaction.timestamp()

    return None


def update_last_interaction_time(session_id: str) -> None:
    """Update the last interaction time for a session.

    Only cancels the active proactive responder task, NOT the trigger loop.
    The trigger loop should continue running to monitor for future idle periods.

    Args:
        session_id: Unique identifier for the session
    """
    global proactive_configs

    if session_id in proactive_configs:
        # Update the config's last interaction time
        proactive_configs[session_id].last_user_interaction = datetime.now()

        # If we have access to session memory, we could potentially update
        # session-specific information, but for now we just focus on the config

        # Only cancel the proactive responder task if there's one running,
        # but do NOT cancel the trigger loop
        if session_id in proactive_responder_tasks and not proactive_responder_tasks[session_id].done():
            logger.info(f"Cancelling proactive responder task for session {session_id} due to user interaction")
            proactive_responder_tasks[session_id].cancel()

            # Clean up the reference after cancelling
            del proactive_responder_tasks[session_id]
    else:
        # Initialize config if not exists, with default settings
        proactive_configs[session_id] = ProactiveConfig(
            enabled=False,
            idle_minutes=30,
            last_user_interaction=datetime.now()
        )


def ensure_valid_session_id(channel_id: str, sender_id: str, provided_session_id: Optional[str] = None) -> str:
    """Ensure we have a valid session_id, either provided or derived from system context.

    Args:
        channel_id: Channel identifier
        sender_id: Sender identifier
        provided_session_id: Session ID that may have been provided

    Returns:
        Valid session_id for use in the system
    """
    if provided_session_id:
        return provided_session_id

    # Generate a session ID based on channel and sender (following system convention)
    return f"{channel_id}:{sender_id}"


# Removed redundant create_session_id function - session_id should be provided by the system
# instead of being created by this module


async def send_proactive_message(session_id: str = "proactive_assistant", message: Msg = None) -> Optional[Msg]:
    """Send a proactive message to the session and return the message.

    Args:
        session_id: Unique identifier for the session (defaults to proactive_assistant)
        message: The proactive message to send

    Returns:
        The message that was sent (either original or a new one if conversion was needed)
    """
    # Helper function to safely convert content to string
    def safe_content_str(content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return " ".join(str(item) for item in content)
        else:
            return str(content)

    # If no session_id provided, default to proactive_assistant
    if not session_id:
        session_id = "proactive_assistant"

    # Get session references (memory_manager, in_memory, etc.) based on session_id
    session_refs = proactive_sessions.get(session_id, {})
    # in_memory = session_refs.get('in_memory')  # Not used in this function

    # Extract channel and user info from session_id if it follows the format "channel:user_id"
    # or from session_refs if available
    channel_id = session_refs.get('channel_id', None)
    user_id = session_refs.get('user_id', None)

    if not channel_id or not user_id:
        # Try to extract from session_id if it's in format "channel:user"
        if ':' in session_id:
            parts = session_id.split(':', 1)
            if len(parts) >= 2:
                channel_id = parts[0]
                user_id = parts[1]

        # If we still don't have channel_id or user_id, create default ones for proactive assistant
        if not channel_id:
            channel_id = "console"
        if not user_id:
            user_id = "proactive_system"


    # If message is provided and has content, use it
    if message and message.content:
        content_str = safe_content_str(message.content)
    else:
        # Create a default message if none provided
        content_str = "Proactive message content"
        message = Msg(
            name="Friday",
            role="assistant",
            content=content_str,
            metadata={"is_proactive": True, "timestamp": datetime.now()}
        )


    try:
        # Import the console push store to add the message
        from ...app.console_push_store import append as push_store_append

        if session_id and content_str.strip():
            await push_store_append(session_id, f"[Proactive] {content_str}")
            logger.info(f"Proactive message pushed to console store for session {session_id}")
        else:
            logger.warning(f"Session ID or content is empty, cannot send via console push store")

    except Exception as e:
        logger.error(f"Failed to send proactive message via console push store: {e}")
        # If console push store fails, we'll try the channel manager as fallback
        # Use channel manager as fallback if console push store fails
        channel_manager = session_refs.get('channel_manager', None)

        # Try to access channel manager from globals if available
        if not channel_manager:
            try:
                # Since we're outside the request context, we need to get the
                # global multi-agent manager to access channel managers
                # Look for it in a common application state location
                from ...app._app import app
                if hasattr(app.state, 'multi_agent_manager'):
                    multi_agent_manager = app.state.multi_agent_manager
                    # In this context we can't use async/await, so we'll try to access
                    # the default agent's channel manager if it exists
                    if hasattr(multi_agent_manager, 'agents'):
                        # Look for default agent in loaded agents
                        default_agent = multi_agent_manager.agents.get("default")
                        if default_agent and hasattr(default_agent, 'channel_manager'):
                            channel_manager = default_agent.channel_manager

            except Exception:
                channel_manager = None

        # If we have a channel manager, send the message through the appropriate channel
        if channel_manager and channel_id and user_id:
            try:
                # Convert the message to a format suitable for the channel manager
                await channel_manager.send_text(
                    channel=channel_id,
                    user_id=user_id,
                    session_id=session_id,
                    text=content_str,  # Already prepared above
                    meta={'is_proactive': True}
                )
                logger.info(f"Proactive message sent via channel manager to {channel_id}:{user_id}")
            except Exception as e:
                logger.error(f"Failed to send proactive message via channel manager: {e}")
                # Still log the message even if channel send fails
                logger.info(f"Proactive message content (fallback logging): {content_str}")

    # Return the message object that was processed
    return message

        



def get_proactive_config(session_id: str) -> Optional[ProactiveConfig]:
    """Get the proactive configuration for a session.

    Args:
        session_id: Unique identifier for the session

    Returns:
        ProactiveConfig if exists, None otherwise
    """
    return proactive_configs.get(session_id)



def reset_proactive_session(session_id: str) -> None:
    """Reset the proactive session, disabling proactive for this session.

    Args:
        session_id: Unique identifier for the session
    """
    if session_id in proactive_configs:
        del proactive_configs[session_id]

    if session_id in proactive_sessions:
        del proactive_sessions[session_id]

    # Only cancel the responder task if it exists
    if session_id in proactive_responder_tasks and not proactive_responder_tasks[session_id].done():
        proactive_responder_tasks[session_id].cancel()
        del proactive_responder_tasks[session_id]

    # Cancel the trigger loop task if it exists
    if session_id in proactive_tasks and not proactive_tasks[session_id].done():
        proactive_tasks[session_id].cancel()