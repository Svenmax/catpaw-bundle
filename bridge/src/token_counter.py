"""
Token counting utilities - uses tiktoken for accurate token estimation.

Falls back to character-based estimation if tiktoken is not available.
"""

import sys
from typing import List, Dict, Any

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
    print("[WARN] tiktoken not installed, using char-based estimation (1 token ≈ 3.5 chars)", file=sys.stderr)


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if _HAS_TIKTOKEN:
        return len(_ENCODER.encode(text))
    # Fallback: approximate 1 token ≈ 3.5 chars for mixed CJK/English
    return max(1, len(text) // 3)


def count_message_tokens(message: Dict[str, Any]) -> int:
    """Count tokens in a single message dict."""
    # Each message has ~4 tokens of overhead (role, delimiters)
    overhead = 4
    content = message.get("content", "")
    if isinstance(content, str):
        return count_tokens(content) + overhead
    elif isinstance(content, list):
        total = overhead
        for item in content:
            if isinstance(item, dict):
                total += count_tokens(str(item.get("text", "")))
            else:
                total += count_tokens(str(item))
        return total
    return overhead


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count total tokens across all messages."""
    return sum(count_message_tokens(m) for m in messages)


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "...(truncated)") -> str:
    """Truncate text to fit within max_tokens."""
    if _HAS_TIKTOKEN:
        tokens = _ENCODER.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = _ENCODER.decode(tokens[:max_tokens])
        return truncated + suffix
    else:
        max_chars = max_tokens * 3
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + suffix
