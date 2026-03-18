# -*- coding: utf-8 -*-
"""Prompt templates for proactive conversation feature."""

PROACTIVE_TASK_EXTRACTION_PROMPT = """
You are given memory from the user's current session and recent persistent memory.

Your task is to identify the user's most likely current larger goals, then derive the most useful next information query for each goal.

Follow this reasoning process internally:
1. Identify patterns in the user's memory history to determine tasks or topics the user has repeatedly engaged with or expressed concern about.
2. Infer the 1 to 3 more general ongoing tasks the user is most likely trying to complete recently, prioritizing those that appear multiple times in the session memory.
3. For each task, identify what information the user is most likely still missing or actively concerned about.
4. Convert that missing information need into one concrete query.
5. Check whether that query would substantially duplicate a command, search, or request already present in memory.
6. If it duplicates prior user activity, replace it with a different but still relevant query that addresses an adjacent unmet information need.

What counts as a task:
- A broader goal such as implementing a feature, debugging a problem, preparing a plan, evaluating options, or completing a deliverable.
- Focus on tasks that appear multiple times in the user's memory history or express consistent concerns.
- Prioritize user instructions and directives over specific execution steps or intermediate results.
- Ignore detailed task execution process memories, focusing instead on user intent and goals.

What counts as a good query:
- A concrete search topic, question, or instruction that could be executed by tools.
- It should help the user make progress on the larger task right now.
- It should add new value rather than repeat prior user commands or searches.

Return JSON only in this format:
{
  "tasks": [
    {
      "task": "short description of the larger user task",
      "query": "specific non-duplicative information query",
      "why": "brief explanation of why this is likely the user's current task and why this query is the most useful next information need"
    }
  ]
}

Rules:
- Return 1 to 3 tasks only.
- Order by priority, with emphasis on tasks that appear repeatedly in memory.
- Prefer the tasks most likely to matter now based on frequency and recency in memory.
- Queries must be concrete, searchable, and non-duplicative.
- Do not repeat or lightly paraphrase user commands already seen in memory.
- Base all inferences on the provided memory only.
- Do not include any text outside the JSON.
"""


PROACTIVE_USER_FACING_MESSAGE_PROMPT = """
Based on the following memory context and gathered information, create a helpful proactive message for the user:

Session Context:
{session_context}

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
"""