# -*- coding: utf-8 -*-
"""Prompt templates for proactive conversation feature."""

PROACTIVE_TASK_EXTRACTION_PROMPT = """\
You are given the user's recent session contexts and current screen content (if provided) in a reversed order.

Your job:
1. Find 1–3 likely high-level goals the user requests (prioritize those mentioned repeatedly or recently).
2. For each goal, create one **new, concrete query** that helps the user move forward—**not** a repeat of past commands or searches.

A good goal:
- Based only on user request or user messages in the provided memory.

A good query:
- Specific, actionable, and tool-friendly (e.g., a search or request).
- Addresses missing info the user likely needs *now*, and queries for the newest information.
- Avoids duplicating anything already in context, or similar to previous [PROACTIVE] labeled messages.

Output ONLY a JSON object with this structure:
{
  "tasks": [
    {
      "task": "short goal description",
      "query": "concrete next query",
      "why": "why this goal is likely and why this query helps"
    }
  ]
}

Rules:
- Return 1 to 3 tasks, ordered by priority (frequency + recency).
- Do not create query only according to the screen content, use session contexts as well.
- Do not use any tools in inference, answer directly based on context.
- At least return 1 task as long as the contexts are not empty.
- No extra text—only valid JSON.
"""


PROACTIVE_USER_FACING_MESSAGE_PROMPT = """
Based on the following gathered information, create a helpful proactive message for the user:

Gathered Information:
{gathered_info}

Remember to be helpful but not intrusive. Provide concise, actionable information that addresses the user's likely needs based on the context.

When crafting the message, consider using reference phrases like:
- "I noticed you've been focusing on [topic/task], here's some information that might help..."
- "I've observed your interest in [topic], so I've gathered the following..."
- "I see you're working on [task], would you like to know more about..."
- "I noticed you've been looking into [topic], here are some updates/resources..."
- "I've seen you're concerned with [issue], here's what I found regarding that..."

These phrases can help frame your message as being attentive to the user's repeated interests rather than intrusive.
Answer in {language} language.

IMPORTANT: Begin your response with the identifier "[PROACTIVE] " to indicate this is a proactive message.
"""