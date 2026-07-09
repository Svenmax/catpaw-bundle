"""
Smart tool filtering - dynamically select relevant tools based on user query,
instead of injecting all 127 tools every time.

This dramatically reduces the tool prompt size, freeing context for
system prompt and conversation history.
"""

import re
import sys
from typing import List, Dict, Set, Tuple

# Core tools that are always included (file/terminal/search operations)
ALWAYS_INCLUDE = {
    "terminal_exec", "terminal_background",
    "file_read", "file_write", "file_list", "file_search",
    "file_edit", "file_multi_edit", "file_delete", "file_move",
    "grep_search", "glob_file_search", "codebase_search",
    "web_search",
}

# Keyword-based tool categories
KEYWORD_CATEGORIES = {
    "browser": {
        "keywords": ["browser", "web", "page", "url", "click", "navigate",
                     "scroll", "screenshot", "网页", "浏览器", "打开网站"],
        "tool_prefixes": ["browser_"],
    },
    "feishu": {
        "keywords": ["feishu", "lark", "飞书", "document", "doc", "wiki",
                     "sheet", "表格", "消息", "message", "chat", "文档",
                     "上传", "upload", "分享", "share"],
        "tool_prefixes": ["feishu", "lark"],
    },
    "mcp": {
        "keywords": ["mcp", "awesun", "remote", "desktop", "control",
                     "远程", "控制", "连接"],
        "tool_prefixes": ["mcp_"],
    },
    "skill": {
        "keywords": ["skill", "技能", "create", "create_skill", "skill_manage"],
        "tool_prefixes": ["skill_"],
    },
    "clarify": {
        "keywords": ["clarify", "ask", "question", "prompt"],
        "tool_prefixes": ["clarify"],
    },
}


def extract_tool_info(tool: dict) -> Tuple[str, str, List[str], List[str]]:
    """Extract (name, description, param_names, required_params) from tool definition."""
    func = tool.get("function", {})
    name = func.get("name", "")
    desc = func.get("description", "")
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])
    param_names = list(props.keys())
    return name, desc, param_names, required


def filter_tools_by_query(
    tools: List[dict],
    user_messages: List[str],
    always_include: Set[str] = None,
) -> List[dict]:
    """
    Dynamically filter tools based on user's recent messages.

    Strategy:
    1. Always include core tools (terminal, file, search)
    2. Scan user messages for keywords -> include matching categories
    3. Include tools whose name or description contains query keywords
    4. Never exceed a reasonable limit

    Args:
        tools: Full list of tool definitions
        user_messages: Recent user messages to analyze
        always_include: Set of tool names to always include

    Returns:
        Filtered list of relevant tools
    """
    if always_include is None:
        always_include = ALWAYS_INCLUDE

    if len(tools) <= 20:
        # Few tools, just include all
        return tools

    # Combine recent user messages for keyword analysis
    combined_text = " ".join(user_messages[-5:]).lower()

    # Determine which categories to include
    active_prefixes: Set[str] = set()
    for category, info in KEYWORD_CATEGORIES.items():
        for kw in info["keywords"]:
            if kw.lower() in combined_text:
                active_prefixes.update(info["tool_prefixes"])
                break

    # Also extract significant words from user messages
    user_words = set(re.findall(r'\w+', combined_text))
    # Filter out common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "must", "can",
                  "this", "that", "these", "those", "i", "you", "he", "she",
                  "it", "we", "they", "what", "which", "who", "when", "where",
                  "why", "how", "all", "each", "every", "some", "any", "no",
                  "not", "as", "at", "by", "for", "with", "about", "against",
                  "between", "into", "through", "during", "before", "after",
                  "above", "below", "to", "from", "up", "down", "in", "out",
                  "on", "off", "over", "under", "again", "further", "then",
                  "once", "here", "there", "and", "but", "or", "nor", "so",
                  "if", "than", "too", "very", "just", "also", "only", "my",
                  "me", "myself", "our", "ours", "ourselves", "your", "yours",
                  "yourself", "yourselves", "him", "his", "her", "hers",
                  "its", "their", "theirs", "them", "themselves", "what's",
                  "给", "我", "的", "了", "在", "是", "有", "和", "就", "不",
                  "人", "都", "一", "上", "也", "很", "到", "说", "要", "去",
                  "你", "会", "看", "好", "自己", "这", "那", "它", "把",
                  "个", "来", "还", "能", "做", "想", "什么", "怎么"}
    user_words = user_words - stop_words

    # Filter tools
    selected: List[dict] = []
    selected_names: Set[str] = set()

    for tool in tools:
        if tool.get("type") != "function":
            continue

        name, desc, _, _ = extract_tool_info(tool)
        name_lower = name.lower()
        desc_lower = desc.lower()

        # Rule 1: Always include core tools
        if name in always_include:
            selected.append(tool)
            selected_names.add(name)
            continue

        # Rule 2: Include if tool prefix matches active category
        if any(name_lower.startswith(prefix) for prefix in active_prefixes):
            selected.append(tool)
            selected_names.add(name)
            continue

        # Rule 3: Include if tool name or description contains user keywords
        if user_words:
            for word in user_words:
                if len(word) >= 3 and (word in name_lower or word in desc_lower):
                    selected.append(tool)
                    selected_names.add(name)
                    break

    # Always ensure we have at least the core tools
    if len(selected) < 5:
        for tool in tools:
            name = tool.get("function", {}).get("name", "")
            if name in always_include and name not in selected_names:
                selected.append(tool)
                selected_names.add(name)

    # Sort by priority (core tools first)
    priority_order = list(always_include)
    def sort_key(t):
        name = t.get("function", {}).get("name", "")
        try:
            return priority_order.index(name)
        except ValueError:
            return len(priority_order)

    selected.sort(key=sort_key)

    # Log filtering result
    all_names = [t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"]
    sel_names = [t.get("function", {}).get("name", "") for t in selected]
    print(f"[DEBUG] Tool filtering: {len(tools)} -> {len(selected)} tools. "
          f"Active categories: {active_prefixes or 'none'}. "
          f"Selected: {sel_names[:15]}{'...' if len(sel_names) > 15 else ''}", file=sys.stderr)

    return selected
