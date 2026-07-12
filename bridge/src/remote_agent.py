"""Remote Agent compatibility helpers for CatPaw Bridge.

CatPaw IDE's Remote Agent UI is a webview shell around a podUrl returned by
/api/agent/conversation/detail.  This module exposes that shell behavior over
HTTP so Bridge users can open or poll Remote Agent conversations without VS Code.
"""

from __future__ import annotations

import http.client
import json
import ssl
import time
import urllib.parse
from typing import Any, Dict, Optional

from .catpaw_client import CatPawClient
from .crypto import decrypt_response_body
from .token_manager import TokenManager


class RemoteAgentError(RuntimeError):
    pass


def normalize_remote_agent_detail(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatPaw Remote Agent detail to the shape used by the IDE bridge."""
    if not isinstance(data, dict):
        return {}

    create_time = data.get("createTime")
    update_time = data.get("updateTime")
    duration = "unknown"
    if create_time and update_time:
        try:
            elapsed = int(update_time) / 1000 - int(create_time) / 1000
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            duration = f"{minutes}m{seconds}s"
        except Exception:
            duration = "unknown"

    return {
        "conversationId": data.get("conversationId"),
        "title": data.get("title") or "Remote Agent",
        "summary": data.get("summary"),
        "status": data.get("status") or "running",
        "duration": duration,
        "gitRepoUrl": data.get("gitRepoUrl"),
        "gitBaseBranch": data.get("gitBaseBranch"),
        "gitCheckoutBranch": data.get("gitCheckoutBranch"),
        "taskDescription": data.get("prompt"),
        "autoPullRequest": data.get("autoPullRequest"),
        "autoDeploy": data.get("autoDeploy"),
        "deployUrl": data.get("deployUrl"),
        "prUrl": data.get("prUrl"),
        "deployStatus": data.get("deployStatus"),
        "deployFailReason": data.get("deployFailReason"),
        "prStatus": data.get("prStatus"),
        "prFailReason": data.get("prFailReason"),
        "podReady": data.get("podReady"),
        "podUrl": data.get("podUrl"),
        "raw": data,
    }


def build_create_conversation_body(request: Dict[str, Any]) -> Dict[str, Any]:
    """Build the CatPaw payload for /api/agent/conversation/create."""
    prompt = request.get("prompt") or request.get("task") or request.get("taskDescription")
    git_repo_url = request.get("gitRepoUrl") or request.get("git_repo_url")
    if not prompt:
        raise RemoteAgentError("prompt is required")
    if not git_repo_url:
        raise RemoteAgentError("gitRepoUrl is required")

    body = {
        "modelType": request.get("modelType") or request.get("model") or "minimax-m2.7",
        "gitRepoUrl": git_repo_url,
        "gitBaseBranch": request.get("gitBaseBranch") or request.get("baseBranch") or request.get("git_base_branch") or "master",
        "gitCheckoutBranch": request.get("gitCheckoutBranch") or request.get("checkoutBranch") or request.get("git_checkout_branch") or "",
        "prompt": prompt,
        "mode": "REMOTE_AGENT",
        "autoDeploy": bool(request.get("autoDeploy") or request.get("auto_deploy") or False),
        "autoPullRequest": bool(request.get("autoPullRequest") or request.get("autoPR") or request.get("auto_pr") or False),
        "source": request.get("source") or "CatPaw",
        "appkeys": request.get("appkeys") or request.get("appKeys") or [],
        "imageUrls": request.get("imageUrls") or request.get("image_urls") or [],
        "contexts": request.get("contexts") or [],
        "editorContextStates": request.get("editorContextStates") or request.get("editor_context_states") or [],
        "mcpServers": request.get("mcpServers") or request.get("mcp_servers") or [],
    }
    return body


class RemoteAgentClient:
    """Client for CatPaw Remote Agent detail APIs."""

    def __init__(self, token_manager: TokenManager, host: str = "catpaw.meituan.com"):
        self.token_manager = token_manager
        self.host = host
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _request_json(self, method: str, path: str, request_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        token = self.token_manager.get_token()
        if not token:
            raise RemoteAgentError("CatPaw token not found. Make sure CatPaw IDE is logged in.")

        body = json.dumps(request_body or {}, ensure_ascii=False).encode("utf-8") if method != "GET" else None
        headers = self.token_manager.build_headers(token)
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        conn = http.client.HTTPSConnection(self.host, context=self._ssl_ctx)
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        resp_headers = dict(resp.getheaders())
        body = resp.read().decode("utf-8", errors="replace")
        conn.close()

        if resp.status == 401:
            self.token_manager.refresh_token()
            token = self.token_manager.get_token()
            if token:
                headers = self.token_manager.build_headers(token)
                if request_body is not None and method != "GET":
                    headers["Content-Type"] = "application/json; charset=utf-8"
                retry_body = json.dumps(request_body or {}, ensure_ascii=False).encode("utf-8") if method != "GET" else None
                conn = http.client.HTTPSConnection(self.host, context=self._ssl_ctx)
                conn.request(method, path, body=retry_body, headers=headers)
                resp = conn.getresponse()
                resp_headers = dict(resp.getheaders())
                body = resp.read().decode("utf-8", errors="replace")
                conn.close()

        if resp.status != 200:
            raise RemoteAgentError(f"CatPaw Remote Agent API returned HTTP {resp.status}: {body[:500]}")

        encrypted_key = resp_headers.get("encrypted-key")
        if encrypted_key:
            body = decrypt_response_body(body.strip('"'), encrypted_key)

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RemoteAgentError(f"CatPaw Remote Agent API returned invalid JSON: {body[:500]}") from exc

        if isinstance(result, dict) and result.get("code") not in (None, 0, 200):
            raise RemoteAgentError(f"CatPaw Remote Agent API error: {result.get('msg') or result.get('message') or result}")
        return result

    def get_detail(self, conversation_id: str) -> Dict[str, Any]:
        if not conversation_id:
            raise RemoteAgentError("conversationId is required")
        query = urllib.parse.urlencode({"conversationId": conversation_id})
        result = self._request_json("GET", f"/api/agent/conversation/detail?{query}")
        if not isinstance(result, dict):
            raise RemoteAgentError(f"CatPaw Remote Agent API returned unexpected payload: {result}")

        data = result.get("data", result)
        if not isinstance(data, dict):
            raise RemoteAgentError(f"CatPaw Remote Agent API returned no detail object: {data}")
        return normalize_remote_agent_detail(data)

    def create_conversation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        body = build_create_conversation_body(request)
        result = self._request_json("POST", "/api/agent/conversation/create", body)
        if not isinstance(result, dict):
            raise RemoteAgentError(f"CatPaw Remote Agent API returned unexpected payload: {result}")
        data = result.get("data", result)
        if not isinstance(data, dict):
            raise RemoteAgentError(f"CatPaw Remote Agent create returned no object: {data}")
        return data

    def connect_stream(self, conversation_id: str, message_index: int = 0):
        """Connect to the native CatPaw Agent SSE stream used by the IDE."""
        if not conversation_id:
            raise RemoteAgentError("conversationId is required")
        client = CatPawClient(self.host, "/api/agent/stream/connect", self.token_manager)
        return client.call_stream({
            "timestamp": int(time.time() * 1000),
            "conversationId": conversation_id,
            "messageIndex": max(0, int(message_index)),
        })

    def continue_conversation(self, conversation_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Continue an existing native Agent conversation with a new user input."""
        if not conversation_id:
            raise RemoteAgentError("conversationId is required")
        body = dict(request)
        body["conversationId"] = conversation_id
        return self._request_json("POST", "/api/agent/conversation/continue", body)

    def stop_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Request cancellation of a native Agent conversation."""
        if not conversation_id:
            raise RemoteAgentError("conversationId is required")
        return self._request_json("POST", "/api/agent/conversation/stop", {"conversationId": conversation_id})

    def wait_for_pod(self, conversation_id: str, timeout: float = 120.0, interval: float = 2.0) -> Dict[str, Any]:

        deadline = time.time() + timeout
        last = None
        while time.time() <= deadline:
            last = self.get_detail(conversation_id)
            if last.get("podReady") and last.get("podUrl"):
                return last
            time.sleep(max(0.2, interval))
        raise RemoteAgentError(f"Remote Agent pod not ready before timeout. Last detail: {last}")


def build_remote_agent_shell(detail: Dict[str, Any], refresh_seconds: int = 2) -> str:
    """Build an HTML shell that mirrors the VS Code webview container behavior."""
    pod_url = detail.get("podUrl") or ""
    ready = bool(detail.get("podReady") and pod_url)
    title = detail.get("title") or "Remote Agent"
    conversation_id = detail.get("conversationId") or ""
    status = detail.get("status") or "unknown"

    escaped_title = html_escape(title)
    escaped_status = html_escape(status)
    escaped_conversation_id = html_escape(conversation_id)
    escaped_pod_url = html_escape(pod_url)

    if ready:
        body = f'<iframe src="{escaped_pod_url}" allow="clipboard-read; clipboard-write; fullscreen" referrerpolicy="no-referrer"></iframe>'
    else:
        body = f"""
        <main class=\"loading\">
          <div class=\"spinner\"></div>
          <h1>{escaped_title}</h1>
          <p>Remote Agent pod is starting...</p>
          <p class=\"muted\">conversationId: {escaped_conversation_id}</p>
          <p class=\"muted\">status: {escaped_status}</p>
        </main>
        <script>setTimeout(function() {{ location.reload(); }}, {max(1, refresh_seconds) * 1000});</script>
        """

    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>{escaped_title}</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; background: #0f1115; color: #e6e8ee; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    iframe {{ border: 0; width: 100vw; height: 100vh; background: #fff; }}
    .loading {{ height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }}
    .spinner {{ width: 34px; height: 34px; border: 3px solid rgba(255,255,255,.2); border-top-color: #59a6ff; border-radius: 50%; animation: spin 1s linear infinite; }}
    .muted {{ color: #9aa4b2; margin: 0; }}
    h1 {{ margin: 8px 0 0; font-size: 20px; }}
    p {{ margin: 0; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def html_escape(value: Any) -> str:
    return ("" if value is None else str(value)).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
