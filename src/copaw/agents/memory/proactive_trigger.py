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
    user_id: Optional[str] = None
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
        last_user_interaction=datetime.now()
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
            actual_last_user_time = await get_last_user_message_ts(session_id)

            # If we can't get the actual time from session, fall back to config
            if actual_last_user_time is not None:
                last_interaction_dt = datetime.fromtimestamp(actual_last_user_time)
            else:
                last_interaction_dt = config.last_user_interaction

            if last_interaction_dt is not None:
                elapsed_minutes = (datetime.now() - last_interaction_dt).total_seconds() / 60.0
                if elapsed_minutes >= config.idle_minutes:
                    # Check if we've already attempted a proactive response in this idle period
                    # and ensure sufficient cooldown after failure
                    if (config.running_task_id is None and  # No proactive task currently running
                        (last_trigger_attempt is None or
                         (datetime.now() - last_trigger_attempt).total_seconds() > 60)):  # At least 1 min cooldown
                        # Check if last message was proactive to prevent duplicate triggers
                        if not await is_last_message_proactive(session_id):
                            # Mark that we're attempting to trigger proactive response
                            last_trigger_attempt = datetime.now()

                            # Trigger proactive response
                            config.running_task_id = f"proactive_{datetime.now().timestamp()}"
                            try:
                                # Pass the session references if available
                                session_refs = proactive_sessions.get(session_id, {})

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
                                    # Send the proactive message (this would need integration with actual agent)
                                    await send_proactive_message(session_id, proactive_msg)
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
                                    # Update last proactive sent time
                                    proactive_configs[session_id].last_proactive_sent = datetime.now()

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


async def get_last_user_message_ts(session_id: str) -> Optional[float]:
    """Get the timestamp of the last user message in session.

    Args:
        session_id: Unique identifier for the session

    Returns:
        Timestamp of last user message or None if not available
    """
    # First check if we have access to real session memory
    session_refs = proactive_sessions.get(session_id, {})
    in_memory = session_refs.get('in_memory')

    if in_memory:
        try:
            # Access the actual session memory to find last user message
            messages = await in_memory.get_memory(
                exclude_mark=None,
                prepend_summary=False,
            )

            # Look for the most recent user message (role='user')
            for msg in reversed(messages):
                if msg.role == 'user':
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
                        elif isinstance(msg.timestamp, float) or isinstance(msg.timestamp, int):
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
                        else:
                            return None
                    return None
        except Exception as e:
            logger.warning(f"Could not access session memory for user message timestamp: {e}")

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
        proactive_configs[session_id].last_user_interaction = datetime.now()

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


async def send_proactive_message(session_id: str, message: Msg) -> None:
    """Send a proactive message to the session.

    Args:
        session_id: Unique identifier for the session
        message: The proactive message to send
    """
    # Get session references to actually send the message
    session_refs = proactive_sessions.get(session_id, {})
    in_memory = session_refs.get('in_memory')

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

    # Try to access the channel manager from the application context if available
    # This is a best-effort approach to get the channel manager from globals
    channel_manager = session_refs.get('channel_manager', None)

    # Store the message in session memory if available
    message_stored_in_memory = False
    if in_memory:
        try:
            # Actually store the message in the session memory
            # Make sure the message has the proactive marker
            if not hasattr(message, 'metadata') or message.metadata is None:
                message.metadata = {}
            message.metadata['is_proactive'] = True

            # Add timestamp if not present
            if not hasattr(message, 'timestamp') or message.timestamp is None:
                from datetime import datetime
                message.timestamp = datetime.now().isoformat()

            # Store the message in session memory
            await in_memory.update_memory([message])
            message_stored_in_memory = True
            logger.info(f"Proactive message stored in memory for session {session_id}: {message.content[:100]}...")
        except Exception as e:
            logger.error(f"Failed to store proactive message in memory for session {session_id}: {e}")

    # If we have a channel manager, send the message through the appropriate channel
    if channel_manager and channel_id and user_id:
        try:
            await channel_manager.send_text(
                channel=channel_id,
                user_id=user_id,
                session_id=session_id,
                text=message.content,
                meta={'is_proactive': True}
            )
            logger.info(f"Proactive message sent via channel manager to {channel_id}:{user_id}")
        except Exception as e:
            logger.error(f"Failed to send proactive message via channel manager: {e}")
            # Still log the message even if channel send fails
            logger.info(f"Proactive message content (fallback logging): {message.content}")
    elif message_stored_in_memory:
        # If we couldn't send via channel but stored in memory, that's acceptable
        logger.info(f"Proactive message stored in memory (no channel manager available) for session {session_id}")
    else:
        # Neither memory storage nor channel send worked - fallback to logging
        logger.info(f"Sending proactive message to session {session_id}: {message.content}")


def get_proactive_config(session_id: str) -> Optional[ProactiveConfig]:
    """Get the proactive configuration for a session.

    Args:
        session_id: Unique identifier for the session

    Returns:
        ProactiveConfig if exists, None otherwise
    """
    return proactive_configs.get(session_id)


def get_proactive_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get the proactive session references for a session.

    Args:
        session_id: Unique identifier for the session

    Returns:
        Session references dict if exists, None otherwise
    """
    return proactive_sessions.get(session_id)


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