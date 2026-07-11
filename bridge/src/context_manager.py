"""
Context manager - smart conversation history management.

Instead of blindly truncating old messages, this module:
1. Uses accurate token counting
2. Preserves system prompt and recent context
3. Summarizes old tool results instead of dropping them entirely
4. Prioritizes keeping tool call/result pairs together
"""

import sys
import json
from typing import List, Dict, Any, Tuple, Optional, Callable
from .token_counter import count_tokens, count_message_tokens, count_messages_tokens, truncate_to_tokens


def summarize_tool_result(content: str, max_chars: int = 1000) -> str:
    """
    Intelligently summarize a tool result instead of brute truncation.

    Strategy:
    - If short enough, keep as-is
    - Try to extract key information (headers, summary lines, error messages)
    - Keep first and last portions (usually most important)
    """
    if len(content) <= max_chars:
        return content

    lines = content.split("\n")

    # If few lines but very long, truncate each line
    if len(lines) <= 5:
        half = max_chars // 2
        return content[:half] + "\n...(truncated)...\n" + content[-half:]

    # Keep first few lines (headers/summary) and last few lines (totals/errors)
    head_lines = min(5, len(lines) // 3)
    tail_lines = min(5, len(lines) // 3)
    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:])
    middle_count = len(lines) - head_lines - tail_lines

    result = f"{head}\n...({middle_count} lines omitted)...\n{tail}"

    # If still too long, fall back to simple truncation
    if len(result) > max_chars:
        result = content[:max_chars - 20] + "\n...(truncated)"

    return result


def truncate_conversation_history(
    messages: List[Dict[str, Any]],
    max_total_tokens: int = 8000,
    max_system_chars: int = 3000,
    max_tool_result_chars: int = 3000,
    summarizer: Optional[Callable[[List[Dict[str, Any]]], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Truncate conversation history to fit within token budget.

    Preserves:
    - System messages (truncated if too long)
    - Most recent conversation messages
    - Tool call/result pairs (kept together when possible)

    Drops:
    - Oldest conversation messages first
    - Redundant tool results (summarized instead)

    If summarizer is provided and 3+ messages are dropped, calls it to get
    a summary of the dropped content and injects it as a system message.
    Falls back to truncation notice if summarizer returns None.
    """
    total_tokens = count_messages_tokens(messages)

    if total_tokens <= max_total_tokens:
        # Still apply tool result truncation
        result = []
        for msg in messages:
            msg = dict(msg)  # shallow copy
            if msg.get("role") in ("tool", "user") and isinstance(msg.get("content"), str):
                if "[Tool Result:" in msg.get("content", "") and len(msg["content"]) > max_tool_result_chars:
                    # Summarize long tool results
                    content = msg["content"]
                    prefix_end = content.find("]\n")
                    if prefix_end > 0:
                        prefix = content[:prefix_end + 2]
                        body = content[prefix_end + 2:]
                        msg["content"] = prefix + summarize_tool_result(body, max_tool_result_chars)
                    else:
                        msg["content"] = summarize_tool_result(content, max_tool_result_chars)
            result.append(msg)
        return result

    # Separate system and conversation messages
    system_msgs = []
    conv_msgs = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(dict(msg))
        else:
            conv_msgs.append(dict(msg))

    # Truncate system messages
    for msg in system_msgs:
        content = msg.get("content", "")
        if len(content) > max_system_chars:
            msg["content"] = truncate_to_tokens(content, max_system_chars // 3)

    system_tokens = count_messages_tokens(system_msgs)
    available_tokens = max_total_tokens - system_tokens

    if available_tokens < 500:
        # System message is too large, force truncate
        for msg in system_msgs:
            content = msg.get("content", "")
            if len(content) > 2000:
                msg["content"] = content[:2000] + "\n...(system prompt truncated)"
        system_tokens = count_messages_tokens(system_msgs)
        available_tokens = max_total_tokens - system_tokens

    # Summarize tool results in conversation messages
    for msg in conv_msgs:
        if msg.get("role") in ("tool", "user") and isinstance(msg.get("content"), str):
            content = msg.get("content", "")
            if "[Tool Result:" in content and len(content) > max_tool_result_chars:
                prefix_end = content.find("]\n")
                if prefix_end > 0:
                    prefix = content[:prefix_end + 2]
                    body = content[prefix_end + 2:]
                    msg["content"] = prefix + summarize_tool_result(body, max_tool_result_chars)
                else:
                    msg["content"] = summarize_tool_result(content, max_tool_result_chars)

    # Keep most recent messages within budget
    kept_conv = []
    current_tokens = 0
    for msg in reversed(conv_msgs):
        msg_tokens = count_message_tokens(msg)
        if current_tokens + msg_tokens > available_tokens:
            break
        kept_conv.append(msg)
        current_tokens += msg_tokens

    kept_conv.reverse()

    if len(kept_conv) < len(conv_msgs):
        dropped_count = len(conv_msgs) - len(kept_conv)
        dropped_msgs = conv_msgs[:dropped_count]
        print(f"[DEBUG] History truncated: dropped {dropped_count} oldest messages, "
              f"{total_tokens} -> {system_tokens + current_tokens} tokens", file=sys.stderr)

        # Try summarization if callback provided and enough messages to justify
        summary = None
        if summarizer and dropped_count >= 3:
            try:
                print(f"[INFO] Summarizing {dropped_count} dropped messages...", file=sys.stderr)
                summary = summarizer(dropped_msgs)
                if summary:
                    print(f"[INFO] Summary generated ({len(summary)} chars)", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Summarization failed: {e}", file=sys.stderr)

        if summary:
            summary_msg = {
                "role": "system",
                "content": f"[Earlier conversation summary]: {summary}"
            }
            kept_conv.insert(0, summary_msg)
        else:
            # Fall back to simple truncation notice
            if kept_conv:
                first = kept_conv[0]
                if first.get("role") == "user" and isinstance(first.get("content"), str):
                    first["content"] = f"[Earlier {dropped_count} messages truncated]\n" + first["content"]

    return system_msgs + kept_conv
