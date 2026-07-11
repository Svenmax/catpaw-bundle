"""
Tool calling translation layer.

CatPaw API doesn't support OpenAI's native `tools` parameter.
This module implements tool calling via:
1. Prompt injection - inject tool definitions into system prompt
2. Response parsing - extract <tool_call> tags from model output
3. Fallback parsing - handle code blocks and function-call syntax
"""

import json
import re
import sys
import uuid
from typing import List, Dict, Any, Tuple, Optional


# ── Tool name mapping ────────────────────────────────────────────────
# Map tool names from consumer to CatPaw model names (forward)
# and from CatPaw model to consumer native tools (reverse)
TOOL_NAME_MAP = {
    "terminal_exec": "bash",
}

TOOL_NAME_MAP_REVERSE = {v: k for k, v in TOOL_NAME_MAP.items()}

# Map CatPaw model tool names -> Agent native tool names
# Agent defines these tools in the prompt but doesn't have
# native implementations for them. Map to Agent's actual tools.
CATPAW_TO_AGENT_TOOL_MAP = {
    "bash": "bash",
    "terminal_exec": "bash",
    "file_list": "bash",
    "file_read": "read",
    "file_write": "write",
    "file_edit": "edit",
    "file_search": "grep",
    "glob_file_search": "glob",
    "grep_search": "grep",
    "codebase_search": "grep",
    "web_search": "webfetch",
    "file_delete": "bash",
    "file_move": "bash",
    "file_multi_edit": "edit",
    "terminal_background": "bash",
}


def map_tool_name(name: str, reverse: bool = False) -> str:
    """Map tool name between consumer and environment names."""
    if reverse:
        # First check CatPaw -> Agent mapping
        mapped = CATPAW_TO_AGENT_TOOL_MAP.get(name)
        if mapped:
            return mapped
        # Then check standard reverse mapping
        return TOOL_NAME_MAP_REVERSE.get(name, name)
    mapping = TOOL_NAME_MAP
    return mapping.get(name, name)


def transform_tool_arguments(name: str, args: dict) -> dict:
    """Transform tool arguments between CatPaw format and Agent native format."""
    if name == "file_list":
        path = args.get("directory_path") or args.get("path") or "."
        return {"command": f"ls -la {_quote(path)}"}
    if name == "file_delete":
        path = args.get("target_file") or args.get("path") or ""
        return {"command": f"rm -rf {_quote(path)}"}
    if name == "file_move":
        src = args.get("source") or args.get("source_path") or ""
        dst = args.get("destination") or args.get("target_path") or ""
        return {"command": f"mv {_quote(src)} {_quote(dst)}"}
    if name == "terminal_background":
        cmd = args.get("command") or ""
        return {"command": cmd}
    if name == "file_read":
        path = args.get("target_file") or args.get("file_path") or ""
        return {"filePath": path}
    if name == "file_write":
        path = args.get("target_file") or args.get("file_path") or ""
        content = args.get("content") or args.get("file_content") or ""
        return {"filePath": path, "content": content}
    if name == "file_edit" or name == "file_multi_edit":
        path = args.get("target_file") or args.get("file_path") or ""
        old = args.get("old_string") or args.get("old") or ""
        new = args.get("new_string") or args.get("new") or ""
        return {"filePath": path, "oldString": old, "newString": new}
    if name == "file_search" or name == "grep_search" or name == "codebase_search":
        pattern = args.get("pattern") or args.get("query") or ""
        path = args.get("path") or args.get("directory") or ""
        result = {"pattern": pattern}
        if path:
            result["path"] = path
        return result
    if name == "glob_file_search":
        pattern = args.get("pattern") or args.get("glob") or ""
        return {"pattern": pattern}
    if name == "web_search":
        query = args.get("query") or ""
        return {"query": query}
    return args


def _quote(s: str) -> str:
    """Shell-quote a string argument."""
    if not s:
        return ""
    if "'" in s:
        s = s.replace("'", "'\\''")
    return f"'{s}'"


# ── Tool prompt construction ──────────────────────────────────────────

# Priority ordering: core tools first
TOOL_PRIORITY = {
    'terminal_exec': 0, 'terminal_background': 1,
    'file_read': 2, 'file_write': 3, 'file_list': 4, 'file_search': 5,
    'file_edit': 6, 'file_multi_edit': 7, 'file_delete': 8, 'file_move': 9,
    'grep_search': 10, 'glob_file_search': 11, 'codebase_search': 12,
    'web_search': 13,
}


def build_tool_system_prefix() -> str:
    """Build the tool-calling instruction prefix for system prompt."""
    return (
        "You are an AI assistant with tool-calling capabilities. You MUST actively use tools to help the user.\n"
        "When the user asks you to do something, DO NOT just describe what you would do - ACTUALLY CALL the tool.\n"
        "\n"
        "## Tool Call Format\n"
        "Output tool calls using this EXACT format (no code blocks, no markdown):\n"
        '<tool_call>{"name":"tool_name","arguments":{"arg":"value"}}</tool_call>\n'
        "\n"
        "## Examples\n"
        'User: check disk space\n'
        '<tool_call>{"name":"bash","arguments":{"command":"df -h"}}</tool_call>\n'
        'User: list files in /tmp\n'
        '<tool_call>{"name":"bash","arguments":{"command":"ls -la /tmp"}}</tool_call>\n'
        'User: read file /etc/hosts\n'
        '<tool_call>{"name":"file_read","arguments":{"target_file":"/etc/hosts"}}</tool_call>\n'
        "\n"
        "## Rules\n"
        "1. ALWAYS call a tool when the user asks you to DO something.\n"
        "2. Use bash for system commands (df, du, ls, cat, ps, etc.)\n"
        "3. Use file_read/file_write/file_list for file operations.\n"
        "4. After receiving tool results, analyze them and respond. Call more tools if needed.\n"
        "5. Do NOT ask for permission to use tools. Just use them.\n"
        "---\n"
    )


def build_tool_prompt(tools: List[dict], max_chars: int = 4000) -> str:
    """
    Build compact tool list for system prompt injection.
    Tools are sorted by priority and grouped by category.
    """
    def tool_priority(t):
        name = t.get("function", {}).get("name", "")
        return TOOL_PRIORITY.get(name, 100)

    sorted_tools = sorted(tools, key=tool_priority)

    # Group by prefix
    groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for tool in sorted_tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = map_tool_name(func.get("name", ""))
        desc = func.get("description", "")
        if len(desc) > 60:
            desc = desc[:60] + "..."
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        param_names = [f"*{p}" if p in required else p for p in props]
        params_line = ",".join(param_names) if param_names else ""

        # Categorize
        if name.startswith("mcp_"):
            prefix = "mcp"
        elif name.startswith("browser_"):
            prefix = "browser"
        elif name.startswith("file_"):
            prefix = "file"
        elif name.startswith("feishu") or name.startswith("lark"):
            prefix = "feishu"
        elif name.startswith("skill_"):
            prefix = "skill"
        elif "_" in name:
            prefix = name.split("_")[0]
        else:
            prefix = "other"

        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((name, params_line, desc))

    lines = ["", "# Available Tools:", ""]
    for group_name, tool_list in groups.items():
        lines.append(f"## {group_name}")
        for name, params, desc in tool_list:
            if desc:
                lines.append(f"  {name}({params}): {desc}")
            else:
                lines.append(f"  {name}({params})")
        lines.append("")

    result = "\n".join(lines)
    if len(result) > max_chars:
        print(f"[DEBUG] Tool prompt truncated: {len(result)} -> {max_chars} chars", file=sys.stderr)
        result = result[:max_chars - 30] + "\n...(more tools truncated)"
    else:
        print(f"[DEBUG] Tool prompt: {len(result)} chars, {sum(len(v) for v in groups.values())} tools", file=sys.stderr)
    return result


# ── Message conversion ────────────────────────────────────────────────

def convert_messages_with_tools(
    messages: List[dict],
    tools: List[dict],
    tool_choice: Optional[dict] = None,
    max_system_chars: int = 3000,
    max_tool_prompt_chars: int = 4000,
) -> List[dict]:
    """
    Convert OpenAI messages with tools to CatPaw-compatible format.
    1. Inject tool definitions into system prompt
    2. Convert tool-role messages to user messages
    3. Convert assistant tool_calls to text format
    """
    def _normalize_content(val) -> str:
        """Convert content (which may be str or list) to plain string."""
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            parts = []
            for chunk in val:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
            return "\n".join(parts)
        return str(val) if val else ""

    tool_prefix = build_tool_system_prefix()
    tool_suffix = build_tool_prompt(tools, max_tool_prompt_chars)
    converted = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            content = _normalize_content(msg.get("content", ""))
            if len(content) > max_system_chars:
                content = content[:max_system_chars] + "\n...(system prompt truncated)"
            content = tool_prefix + content + tool_suffix

            if tool_choice and tool_choice != "auto" and tool_choice != "none":
                if isinstance(tool_choice, dict):
                    forced_name = tool_choice.get("function", {}).get("name", "")
                    if forced_name:
                        content += f"\n\nYou MUST call the tool '{forced_name}' now."
                elif tool_choice == "required":
                    content += "\n\nYou MUST call one of the available tools now."

            converted.append({"role": "system", "content": content})

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = _normalize_content(msg.get("content", ""))
            converted.append({
                "role": "user",
                "content": f"[Tool Result: {tool_call_id}]\n{content}"
            })

        elif role == "assistant" and msg.get("tool_calls"):
            content = _normalize_content(msg.get("content", "") or "")
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = map_tool_name(func.get("name", ""))
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                call_block = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                content += f"\n<tool_call>\n{call_block}\n</tool_call>"
            converted.append({"role": "assistant", "content": content.strip()})

        else:
            msg = dict(msg)
            msg["content"] = _normalize_content(msg.get("content", ""))
            converted.append(msg)

    # If no system message, add one
    if not any(m.get("role") == "system" for m in converted):
        full_prompt = tool_prefix + tool_suffix
        converted.insert(0, {"role": "system", "content": full_prompt})

    return converted


# ── Response parsing ──────────────────────────────────────────────────

def _extract_balanced_json(text: str, start: int) -> Tuple[Optional[str], int]:
    """Extract a complete JSON object starting at position `start` using brace counting."""
    if start >= len(text) or text[start] != '{':
        return None, start
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i + 1
    return None, start


def parse_tool_calls_from_content(content: str) -> Tuple[str, List[dict]]:
    """
    Parse tool calls from model response content.
    Returns (remaining_content, tool_calls_list).

    Parsing strategies (in order):
    1. <tool_call> XML tags with balanced-brace JSON extraction
    2. Code blocks with function-call syntax (func_name(key='value'))
    3. Code blocks with shell commands (converted to terminal_exec)
    """
    if not content:
        return "", []

    tool_calls = []

    # Method 1: <tool_call> tags with brace counting
    tag_positions = []
    pos = 0
    while True:
        idx = content.find('<tool_call>', pos)
        if idx == -1:
            break
        tag_positions.append(idx)
        pos = idx + len('<tool_call>')

    if tag_positions:
        for idx in tag_positions:
            json_start = idx + len('<tool_call>')
            while json_start < len(content) and content[json_start] in ' \t\n\r':
                json_start += 1
            json_str, _ = _extract_balanced_json(content, json_start)
            if json_str:
                try:
                    call = json.loads(json_str)
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": map_tool_name(name, reverse=True),
                            "arguments": json.dumps(transform_tool_arguments(name, args), ensure_ascii=False)
                        }
                    })
                except json.JSONDecodeError:
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
                    if name_match:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {"name": map_tool_name(name_match.group(1), reverse=True), "arguments": "{}"}
                        })

        remaining = re.sub(r'<tool_call>.*?(?:</tool_call>|(?=\n\n|\Z))', '', content, flags=re.DOTALL).strip()
        remaining = remaining.replace('</tool_call>', '').strip()
        if tool_calls:
            return remaining, tool_calls

    # Method 2: Code blocks (function calls or shell commands)
    code_block_pattern = re.compile(r'```(?:python|bash|shell|sh|javascript|json)?\s*\n(.*?)\n```', re.DOTALL)
    code_blocks = code_block_pattern.findall(content)
    if code_blocks:
        for block in code_blocks:
            func_pattern = re.compile(r"(\w+)\s*\(\s*(.*?)\s*\)", re.DOTALL)
            func_matches = list(func_pattern.finditer(block))

            if func_matches:
                for fm in func_matches:
                    name = fm.group(1)
                    args_str = fm.group(2).strip()
                    args = {}
                    if args_str:
                        kv_pattern = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,]+))")
                        for kv in kv_pattern.finditer(args_str):
                            key = kv.group(1)
                            val = kv.group(2) or kv.group(3) or kv.group(4)
                            if val is None:
                                continue
                            val = val.strip()
                            if val.lower() == 'true':
                                val = True
                            elif val.lower() == 'false':
                                val = False
                            else:
                                try:
                                    val = int(val)
                                except ValueError:
                                    try:
                                        val = float(val)
                                    except ValueError:
                                        pass
                            args[key] = val

                    if name not in ('print', 'process', 'len', 'str', 'int', 'float',
                                    'dict', 'list', 'range', 'open', 'type', 'import'):
                        mapped_name = map_tool_name(name, reverse=True)
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": mapped_name,
                                "arguments": json.dumps(transform_tool_arguments(name, args), ensure_ascii=False)
                            }
                        })
            else:
                # Shell command block
                cmd = block.strip()
                if cmd and not cmd.startswith('#'):
                    cmd = cmd.split('\n')[0].strip()
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": map_tool_name("terminal_exec", reverse=True),
                            "arguments": json.dumps(transform_tool_arguments("terminal_exec", {"command": cmd}), ensure_ascii=False)
                        }
                    })

        if tool_calls:
            remaining = code_block_pattern.sub("", content).strip()
            return remaining, tool_calls

    return content, []
