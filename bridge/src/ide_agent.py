"""IDE-agent helpers for CatPaw Bridge.

This module exposes the agent-style capabilities that CatPaw IDE contributes
through VS Code commands, but in a bridge-friendly HTTP shape.  It does not try
to emulate VS Code UI/webviews; it builds the same kinds of coding prompts from
workspace context and sends them through the existing CatPaw chat API client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class IdeAction:
    name: str
    title: str
    ide_command: str
    description: str


IDE_ACTIONS: Dict[str, IdeAction] = {
    "chat": IdeAction(
        "chat",
        "Agent Chat",
        "idekit.mcopilot.chat.selected",
        "General coding chat with optional selected code and file context.",
    ),
    "send_prompt": IdeAction(
        "send_prompt",
        "Send Prompt To Agent",
        "idekit.mcopilot.sendPrompt",
        "Send a raw prompt to the CatPaw agent.",
    ),
    "explain": IdeAction(
        "explain",
        "Explain Code",
        "idekit.mcopilot.explain.selected",
        "Explain selected code or files.",
    ),
    "bug": IdeAction(
        "bug",
        "Find Bugs",
        "idekit.mcopilot.bug.selected",
        "Inspect selected code or files for likely defects.",
    ),
    "test": IdeAction(
        "test",
        "Generate Tests",
        "idekit.mcopilot.test.selected",
        "Generate unit tests for selected code or files.",
    ),
    "comment": IdeAction(
        "comment",
        "Add Comments",
        "idekit.mcopilot.comment.selected",
        "Add concise comments to selected code.",
    ),
    "refactor": IdeAction(
        "refactor",
        "Refactor Code",
        "idekit.mcopilot.refactor.selected",
        "Suggest or produce a refactor for selected code.",
    ),
    "commit_message": IdeAction(
        "commit_message",
        "Generate Commit Message",
        "mcopilot.generateCommitMessage",
        "Generate a commit message from a diff or change summary.",
    ),
    "testagent_selected_file": IdeAction(
        "testagent_selected_file",
        "Generate Incremental Tests For Current File",
        "catpaw.workbench.testagent.selectedFile",
        "Generate incremental tests for one file.",
    ),
    "testagent_all_changes": IdeAction(
        "testagent_all_changes",
        "Generate Tests For All Changed Files",
        "catpaw.workbench.testagent.allFile",
        "Generate tests for all changed files from a diff.",
    ),
    "agent_review": IdeAction(
        "agent_review",
        "Agent Review",
        "catpaw.triggerAgentReview",
        "Review current code changes and report issues.",
    ),
    "agent_review_changes": IdeAction(
        "agent_review_changes",
        "Agent Review From Review Changes",
        "catpaw.triggerAgentReviewByReviewChanges",
        "Review a provided review-changes payload or diff.",
    ),
    "deploy_plan": IdeAction(
        "deploy_plan",
        "Build And Deploy Plan",
        "catpaw.deploy",
        "Create build and deploy steps from project context.",
    ),
}


ACTION_SYSTEM_PROMPTS: Dict[str, str] = {
    "chat": "You are CatPaw IDE's coding agent. Help with the user's coding task using the provided workspace context.",
    "send_prompt": "You are CatPaw IDE's coding agent. Follow the user's prompt exactly and use the provided context.",
    "explain": "Explain the provided code clearly. Cover purpose, control flow, data flow, edge cases, and important dependencies.",
    "bug": "Review the provided code for bugs. Prioritize concrete, reproducible issues. For each issue include severity, evidence, and a fix direction.",
    "test": "Generate practical unit tests for the provided code. Prefer existing style when context is available. Include test cases and rationale.",
    "comment": "Add concise useful comments to the provided code. Do not comment obvious statements. Return the revised code or targeted comment suggestions.",
    "refactor": "Refactor the provided code to improve maintainability without changing behavior. Explain risks and validation steps.",
    "commit_message": "Generate a high-quality commit message from the provided diff. Use a concise subject and a short body when useful.",
    "testagent_selected_file": "Act as CatPaw TestAgent. Generate incremental tests for the selected file based on code and change context.",
    "testagent_all_changes": "Act as CatPaw TestAgent. Generate or update tests for all changed files from the provided diff/context.",
    "agent_review": "Act as CatPaw Agent Review. Review the change set for correctness, regressions, security, tests, and maintainability. Return prioritized findings only.",
    "agent_review_changes": "Act as CatPaw Agent Review. Review the provided review changes/diff and return prioritized findings only.",
    "deploy_plan": "Act as CatPaw deploy assistant. Produce build, verification, deployment, rollback, and risk-check steps from the project context.",
}


def list_capabilities() -> List[Dict[str, str]]:
    """Return IDE-agent capabilities exposed by the bridge."""
    return [action.__dict__.copy() for action in IDE_ACTIONS.values()]


def normalize_action(action: Optional[str]) -> str:
    """Normalize action names and IDE command IDs to a bridge action."""
    if not action:
        return "chat"
    raw = str(action).strip()
    if raw in IDE_ACTIONS:
        return raw
    lowered = raw.lower()
    aliases = {
        "idekit.mcopilot.chat.selected": "chat",
        "idekit.mcopilot.chat.openfirsttab": "chat",
        "idekit.mcopilot.sendprompt": "send_prompt",
        "idekit.mcopilot.explain.selected": "explain",
        "idekit.mcopilot.bug.selected": "bug",
        "idekit.mcopilot.test.selected": "test",
        "idekit.mcopilot.comment.selected": "comment",
        "idekit.mcopilot.refactor.selected": "refactor",
        "mcopilot.generatecommitmessage": "commit_message",
        "catpaw.workbench.testagent.selectedfile": "testagent_selected_file",
        "catpaw.workbench.testagent.allfile": "testagent_all_changes",
        "catpaw.triggeragentreview": "agent_review",
        "catpaw.triggeragentreviewbyreviewchanges": "agent_review_changes",
        "catpaw.deploy": "deploy_plan",
        "review": "agent_review",
        "tests": "test",
        "unit_test": "test",
        "commit": "commit_message",
        "deploy": "deploy_plan",
    }
    return aliases.get(lowered, "chat")


def _append_section(parts: List[str], title: str, value: Any) -> None:
    if value is None or value == "":
        return
    parts.append(f"\n## {title}\n{value}")


def _format_files(files: Any) -> str:
    if not files:
        return ""
    if not isinstance(files, list):
        return str(files)
    chunks: List[str] = []
    for item in files:
        if isinstance(item, dict):
            path = item.get("path") or item.get("file_path") or item.get("name") or "<unknown>"
            content = item.get("content") or item.get("text") or ""
            language = item.get("language") or ""
            fence = language if language else ""
            chunks.append(f"### {path}\n```{fence}\n{content}\n```")
        else:
            chunks.append(str(item))
    return "\n\n".join(chunks)


def build_ide_messages(req: Dict[str, Any]) -> tuple[str, List[Dict[str, str]]]:
    """Build CatPaw chat messages from a bridge IDE-agent request."""
    action = normalize_action(req.get("action") or req.get("command"))
    prompt = req.get("prompt") or req.get("input") or req.get("query") or ""
    selection = req.get("selection") or req.get("selected_text")
    files = req.get("files")
    diff = req.get("diff")
    diagnostics = req.get("diagnostics")
    workspace = req.get("workspace") or req.get("workspace_path")
    extra_context = req.get("context")

    parts: List[str] = []
    if prompt:
        parts.append(str(prompt))
    _append_section(parts, "Workspace", workspace)
    _append_section(parts, "Selected Code", selection)
    _append_section(parts, "Files", _format_files(files))
    _append_section(parts, "Diff", diff)
    _append_section(parts, "Diagnostics", diagnostics)
    _append_section(parts, "Additional Context", extra_context)

    if not parts:
        parts.append("Run the requested IDE agent action with the available context.")

    system = ACTION_SYSTEM_PROMPTS.get(action, ACTION_SYSTEM_PROMPTS["chat"])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts).strip()},
    ]
    return action, messages
