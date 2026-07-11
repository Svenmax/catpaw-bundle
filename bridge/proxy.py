#!/usr/bin/env python3
"""
CatPaw Bridge - OpenAI-compatible Proxy Server

Bridges CatPaw IDE's private LLM API to a standard OpenAI-compatible endpoint,
enabling tools like Hermes Agent to use CatPaw's models (GLM, DeepSeek, Kimi, etc.)

Features:
  - RSA-OAEP + AES-128-ECB encryption (matching CatPaw plugin)
  - SSO token auto-refresh from CatPaw IDE's local database
  - Tool calling translation (prompt injection + response parsing)
  - Smart tool filtering based on user query
  - Token-based context management with smart summarization
  - SSE streaming support
  - YAML configuration

Usage:
  python proxy.py [--config config.yaml]
"""

import copy
import json
import os
import re
import sys
import time
import uuid
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.token_manager import TokenManager
from src.catpaw_client import CatPawClient
from src.tool_translator import (
    convert_messages_with_tools,
    parse_tool_calls_from_content,
)
from src.tool_filter import filter_tools_by_query
from src.context_manager import truncate_conversation_history
from src.token_counter import count_messages_tokens


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    # Class-level config (set by main)
    config: Config = None
    catpaw_client: CatPawClient = None
    token_manager: TokenManager = None

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", file=sys.stderr)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int, html: str):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_error(self, msg: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        error_data = {"error": {"message": msg, "type": "proxy_error"}}
        self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    # ── Routes ────────────────────────────────────────────────────────

    LOGIN_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CatPaw Bridge - 登录</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
  .container { background: #fff; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,.08); padding: 40px; width: 400px; max-width: 94vw; }
  h1 { text-align: center; font-size: 22px; color: #1a1a1a; margin-bottom: 8px; }
  .subtitle { text-align: center; font-size: 13px; color: #888; margin-bottom: 28px; }
  .tabs { display: flex; gap: 0; margin-bottom: 28px; background: #f0f0f0; border-radius: 10px; padding: 3px; }
  .tab { flex: 1; text-align: center; padding: 10px 0; font-size: 14px; cursor: pointer; border-radius: 8px; transition: all .2s; color: #666; }
  .tab.active { background: #fff; color: #1a1a1a; font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .qr-img { display: block; width: 280px; height: 280px; margin: 0 auto 20px; border: 1px solid #eee; border-radius: 12px; }
  .qr-placeholder { display: flex; align-items: center; justify-content: center; width: 280px; height: 280px; margin: 0 auto 20px; background: #fafafa; border: 1px dashed #ddd; border-radius: 12px; color: #999; font-size: 14px; }
  .status-bar { text-align: center; font-size: 14px; color: #666; margin-top: 12px; min-height: 22px; }
  .status-bar.loading { color: #1677ff; }
  .status-bar.success { color: #52c41a; }
  .status-bar.error { color: #ff4d4f; }
  .btn { display: block; width: 100%; padding: 12px; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; transition: all .2s; }
  .btn-primary { background: #1677ff; color: #fff; }
  .btn-primary:hover { background: #4096ff; }
  .btn-primary:disabled { background: #a0c4ff; cursor: not-allowed; }
  .btn-default { background: #f0f0f0; color: #333; }
  .btn-default:hover { background: #e0e0e0; }
  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; font-size: 13px; color: #555; margin-bottom: 6px; }
  .form-group input { width: 100%; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 15px; outline: none; transition: border .2s; }
  .form-group input:focus { border-color: #1677ff; }
  .form-group .send-code-row { display: flex; gap: 10px; }
  .form-group .send-code-row input { flex: 1; }
  .form-group .send-code-btn { white-space: nowrap; padding: 10px 16px; background: #f0f0f0; border: 1px solid #ddd; border-radius: 8px; cursor: pointer; font-size: 13px; color: #555; transition: all .2s; }
  .form-group .send-code-btn:hover { background: #e0e0e0; }
  .form-group .send-code-btn:disabled { color: #bbb; cursor: not-allowed; }
  .result-box { margin-top: 16px; padding: 12px 16px; border-radius: 10px; font-size: 13px; display: none; word-break: break-all; }
  .result-box.success { display: block; background: #f6ffed; border: 1px solid #b7eb8f; color: #389e0d; }
  .result-box.error { display: block; background: #fff2f0; border: 1px solid #ffccc7; color: #cf1322; }
</style>
</head>
<body>
<div class="container">
  <h1>CatPaw Bridge</h1>
  <p class="subtitle">登录后即可通过 OpenAI 兼容接口使用 CatPaw 模型</p>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('qrcode')">扫码登录</div>
    <div class="tab" onclick="switchTab('phone')">手机号登录</div>
  </div>

  <div id="tab-qrcode" class="tab-content active">
    <div id="qr-img-container">
      <div class="qr-placeholder" id="qr-placeholder">正在获取二维码...</div>
    </div>
    <div id="qr-status" class="status-bar loading">等待扫码...</div>
    <button class="btn btn-default" onclick="refreshQR()" style="margin-top:12px">刷新二维码</button>
  </div>

  <div id="tab-phone" class="tab-content">
    <div class="form-group">
      <label>手机号</label>
      <input type="tel" id="phone-input" placeholder="请输入手机号" maxlength="11">
    </div>
    <div class="form-group">
      <label>验证码</label>
      <div class="send-code-row">
        <input type="text" id="code-input" placeholder="请输入验证码" maxlength="6">
        <button class="send-code-btn" id="send-code-btn" onclick="sendCode()">获取验证码</button>
      </div>
    </div>
    <button class="btn btn-primary" onclick="phoneLogin()">登录</button>
    <div id="phone-status" class="status-bar" style="margin-top:12px"></div>
    <div id="phone-result" class="result-box"></div>
  </div>
</div>

<script>
let qrCode = null;
let pollTimer = null;
let countdownTimer = null;
let countdown = 0;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  if (name === 'qrcode') {
    document.querySelectorAll('.tab')[0].classList.add('active');
    document.getElementById('tab-qrcode').classList.add('active');
    if (!qrCode) refreshQR();
  } else {
    document.querySelectorAll('.tab')[1].classList.add('active');
    document.getElementById('tab-phone').classList.add('active');
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }
}

function refreshQR() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  document.getElementById('qr-placeholder').innerHTML = '正在获取二维码...';
  document.getElementById('qr-placeholder').className = 'qr-placeholder';
  document.getElementById('qr-status').textContent = '正在获取二维码...';
  document.getElementById('qr-status').className = 'status-bar loading';

  fetch('/login/qrcode')
    .then(r => r.json())
    .then(data => {
      qrCode = data.code;
      document.getElementById('qr-placeholder').innerHTML = '<img class="qr-img" src="' + data.qr_code_image_url + '" alt="QR Code">';
      document.getElementById('qr-placeholder').className = '';
      document.getElementById('qr-status').textContent = '请使用微信扫码登录';
      document.getElementById('qr-status').className = 'status-bar loading';
      startPolling();
    })
    .catch(e => {
      document.getElementById('qr-placeholder').innerHTML = '获取二维码失败';
      document.getElementById('qr-status').textContent = '获取二维码失败: ' + e.message;
      document.getElementById('qr-status').className = 'status-bar error';
    });
}

function startPolling() {
  if (!qrCode) return;
  pollTimer = setInterval(function() {
    fetch('/login/poll', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code: qrCode})
    })
    .then(r => r.json())
    .then(data => {
      if (data.status === 'ok') {
        clearInterval(pollTimer);
        pollTimer = null;
        document.getElementById('qr-status').textContent = '登录成功! Token: ' + data.token_prefix;
        document.getElementById('qr-status').className = 'status-bar success';
      }
    })
    .catch(function() {});
  }, 2000);
}

function sendCode() {
  var phone = document.getElementById('phone-input').value.trim();
  if (!phone || phone.length < 11) { setPhoneStatus('请输入正确的手机号', 'error'); return; }
  var btn = document.getElementById('send-code-btn');
  btn.disabled = true;
  setPhoneStatus('正在发送...', 'loading');
  document.getElementById('phone-result').className = 'result-box';

  fetch('/login/sendSms', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mobileNo: phone})
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'ok') {
      setPhoneStatus('验证码已发送', 'success');
      countdown = 60;
      btn.textContent = countdown + 's';
      if (countdownTimer) clearInterval(countdownTimer);
      countdownTimer = setInterval(function() {
        countdown--;
        if (countdown <= 0) {
          clearInterval(countdownTimer);
          countdownTimer = null;
          btn.textContent = '重新获取';
          btn.disabled = false;
        } else {
          btn.textContent = countdown + 's';
        }
      }, 1000);
    } else {
      btn.disabled = false;
      setPhoneStatus(data.error || '发送失败', 'error');
    }
  })
  .catch(function(e) {
    btn.disabled = false;
    setPhoneStatus('请求失败: ' + e.message, 'error');
  });
}

function phoneLogin() {
  var phone = document.getElementById('phone-input').value.trim();
  var code = document.getElementById('code-input').value.trim();
  if (!phone || !code) { setPhoneStatus('请填写手机号和验证码', 'error'); return; }
  setPhoneStatus('正在登录...', 'loading');
  document.getElementById('phone-result').className = 'result-box';

  fetch('/login/loginByPhone', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mobileNo: phone, verificationCode: code})
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'ok') {
      var box = document.getElementById('phone-result');
      box.className = 'result-box success';
      box.innerHTML = '登录成功! Token: ' + data.token_prefix + '<br>有效期: ' + data.expires + 's';
      setPhoneStatus('', '');
    } else {
      var box = document.getElementById('phone-result');
      box.className = 'result-box error';
      box.textContent = data.error || '登录失败';
      setPhoneStatus('', '');
    }
  })
  .catch(function(e) {
    setPhoneStatus('请求失败: ' + e.message, 'error');
  });
}

function setPhoneStatus(msg, type) {
  var el = document.getElementById('phone-status');
  el.textContent = msg;
  el.className = type ? 'status-bar ' + type : 'status-bar';
}
</script>
</body>
</html>"""

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self._send_html(200, self.LOGIN_PAGE_HTML)

        elif self.path == "/v1/models":
            models = [
                {"id": m, "object": "model", "created": 1700000000, "owned_by": "catpaw"}
                for m in self.config.models
            ]
            self._send_json(200, {"object": "list", "data": models})

        elif self.path == "/health":
            token = self.token_manager.get_token()
            self._send_json(200, {
                "status": "ok" if token else "no_token",
                "token_prefix": token[:20] + "..." if token else None,
                "models": self.config.models,
            })

        elif self.path == "/login/qrcode":
            from src.oauth_login import QRCodeOAuthLogin
            try:
                oauth = QRCodeOAuthLogin()
                qr = oauth.get_qrcode()
                self._send_json(200, {
                    "code": qr["code"],
                    "qr_code_image_url": qr["qrCodeImageUrl"],
                    "expire_time": qr["expireTime"],
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/token":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len)
            try:
                req_body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            access_token = req_body.get("accessToken") or req_body.get("access_token")
            refresh_token = req_body.get("refreshToken") or req_body.get("refresh_token")

            if not access_token:
                self._send_json(400, {"error": "accessToken is required"})
                return

            self.token_manager.set_token_from_external(access_token, refresh_token)
            self._send_json(200, {
                "status": "ok",
                "token_prefix": access_token[:20] + "...",
            })
            return

        if self.path == "/login/poll":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len)
            try:
                req_body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            code = req_body.get("code")
            if not code:
                self._send_json(400, {"error": "code is required"})
                return

            from src.oauth_login import QRCodeOAuthLogin
            try:
                oauth = QRCodeOAuthLogin()
                result = oauth.poll_access_token(code, timeout=1, interval=1)
                self.token_manager.set_token_from_external(result.access_token, result.refresh_token)
                self.token_manager.write_to_state_db(result.access_token, result.refresh_token)
                self._send_json(200, {
                    "status": "ok",
                    "token_prefix": result.access_token[:20] + "...",
                    "expires": result.expires,
                })
            except TimeoutError:
                self._send_json(200, {"status": "polling"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        if self.path == "/login/sendSms":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len)
            try:
                req_body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            mobile_no = req_body.get("mobileNo") or req_body.get("mobile_no")
            if not mobile_no:
                self._send_json(400, {"error": "mobileNo is required"})
                return

            from src.oauth_login import PhoneOAuthLogin
            try:
                phone_oauth = PhoneOAuthLogin()
                request_code = phone_oauth.send_code(mobile_no)
                self._send_json(200, {"status": "ok", "request_code": request_code})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        if self.path == "/login/loginByPhone":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len)
            try:
                req_body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            mobile_no = req_body.get("mobileNo") or req_body.get("mobile_no")
            verification_code = req_body.get("verificationCode") or req_body.get("verification_code") or req_body.get("code")
            if not mobile_no or not verification_code:
                self._send_json(400, {"error": "mobileNo and verificationCode are required"})
                return

            from src.oauth_login import PhoneOAuthLogin
            try:
                phone_oauth = PhoneOAuthLogin()
                result = phone_oauth.login(mobile_no, verification_code)
                self.token_manager.set_token_from_external(result.access_token, result.refresh_token)
                self.token_manager.write_to_state_db(result.access_token, result.refresh_token)
                self._send_json(200, {
                    "status": "ok",
                    "token_prefix": result.access_token[:20] + "...",
                    "expires": result.expires,
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len)
        try:
            req_body = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        messages = req_body.get("messages", [])
        model = req_body.get("model", "glm-5.2")
        stream = req_body.get("stream", False)
        tools = req_body.get("tools")
        tool_choice = req_body.get("tool_choice")

        if not messages:
            self._send_json(400, {"error": "messages is required"})
            return

        token = self.token_manager.get_token()
        if not token:
            self._send_json(401, {
                "error": "CatPaw token not found. Make sure CatPaw IDE is logged in."
            })
            return

        # ── Tool calling processing ──────────────────────────────────
        if tools:
            # Smart tool filtering: only inject relevant tools
            user_messages = [m.get("content", "") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
            filtered_tools = filter_tools_by_query(
                tools, user_messages,
                always_include=set(self.config.tools.always_include),
            )

            tool_names = [t.get("function", {}).get("name", "?") for t in filtered_tools if t.get("type") == "function"]
            print(f"[DEBUG] Tools: {len(tools)} received -> {len(filtered_tools)} selected: {tool_names[:10]}{'...' if len(tool_names) > 10 else ''}", file=sys.stderr)

            messages = convert_messages_with_tools(
                messages, filtered_tools, tool_choice,
                max_system_chars=self.config.context.max_system_prompt,
                max_tool_prompt_chars=self.config.context.max_tool_prompt,
            )

            # Context management
            messages = truncate_conversation_history(
                messages,
                max_total_tokens=self.config.context.max_total_tokens,
                max_system_chars=self.config.context.max_system_prompt,
                max_tool_result_chars=self.config.context.max_tool_result,
                summarizer=self._summarize_dropped,
            )

            total_tokens = count_messages_tokens(messages)
            print(f"[DEBUG] Final context: {len(messages)} msgs, ~{total_tokens} tokens", file=sys.stderr)
            for i, m in enumerate(messages):
                from src.token_counter import count_message_tokens
                print(f"[DEBUG]   msg[{i}] role={m.get('role','?')} ~{count_message_tokens(m)} tokens", file=sys.stderr)
        else:
            # No tools, just truncate history
            messages = truncate_conversation_history(
                messages,
                max_total_tokens=self.config.context.max_total_tokens,
                summarizer=self._summarize_dropped,
            )

        # ── Build API request ────────────────────────────────────────
        api_body = {"model": model, "messages": messages, "stream": stream}
        for k in ["temperature", "max_tokens", "top_p", "frequency_penalty",
                  "presence_penalty", "response_format"]:
            if k in req_body:
                api_body[k] = req_body[k]

        # ── Send request ─────────────────────────────────────────────
        try:
            if stream and not tools:
                self._handle_stream(api_body, model)
            elif stream and tools:
                self._handle_stream_with_tools(api_body, model)
            else:
                self._handle_non_stream(api_body, model, bool(tools))
        except Exception as e:
            import traceback
            print(f"[ERROR] Proxy error: {e}\n{traceback.format_exc()}", file=sys.stderr)
            try:
                self._send_json(500, {"error": {"message": str(e), "type": "proxy_error"}})
            except Exception:
                pass

    # ── Response handlers ────────────────────────────────────────────

    def _summarize_dropped(self, dropped: List[Dict]) -> Optional[str]:
        """Summarize dropped messages using a cheap CatPaw model."""
        sc = self.config.context.summarize
        if not sc.enabled:
            return None

        parts = []
        for m in dropped:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:2000]
            parts.append(f"[{role}]: {content}")
        summary_text = "\n\n".join(parts)

        req = {
            "model": sc.model,
            "messages": [
                {"role": "system", "content": "You are a conversation summarizer. Create a concise paragraph summarizing the key topics, outputs, decisions, and conclusions from this conversation segment."},
                {"role": "user", "content": f"Summarize this conversation segment:\n\n{summary_text}"}
            ],
            "max_tokens": sc.max_tokens,
            "stream": False
        }

        try:
            result = self.catpaw_client.call(req)
            if result.get("code") != 0:
                print(f"[WARN] Summarization API error: {result.get('msg', '?')}", file=sys.stderr)
                return None
            raw_data = result.get("data", result)
            content = raw_data.get("content", "") or ""
            if not content and raw_data.get("choices"):
                content = raw_data["choices"][0].get("message", {}).get("content", "") or ""
            return content.strip() if content else None
        except Exception as e:
            print(f"[WARN] Summarization call failed: {e}", file=sys.stderr)
            return None

    def _handle_non_stream(self, api_body: dict, model: str, has_tools: bool = False):
        """Handle non-streaming request."""
        result = self.catpaw_client.call(api_body)

        if result.get("code") != 0:
            import json as _json
            err_detail = _json.dumps(result, ensure_ascii=False)[:500]
            raise RuntimeError(f"CatPaw API error: {result.get('msg', 'unknown')}. Full response: {err_detail}")

        raw_data = result.get("data", result)
        raw_content = raw_data.get("content", "") or ""
        if not raw_content and raw_data.get("choices"):
            raw_content = raw_data["choices"][0].get("message", {}).get("content", "") or ""
        print(f"[DEBUG] Raw model response: {raw_content[:500]}", file=sys.stderr)

        openai_resp = self._to_openai_response(result, model, has_tools)
        print(f"[DEBUG] Parsed tool_calls: {len(openai_resp['choices'][0]['message'].get('tool_calls', []))}", file=sys.stderr)
        self._send_json(200, openai_resp)

    def _handle_stream_with_tools(self, api_body: dict, model: str):
        """Handle streaming request with tools: internal non-stream, external simulated stream."""
        api_body["stream"] = False
        result = self.catpaw_client.call(api_body)

        if result.get("code") != 0:
            import json as _json
            err_detail = _json.dumps(result, ensure_ascii=False)[:500]
            raise RuntimeError(f"CatPaw API error: {result.get('msg', 'unknown')}. Full response: {err_detail}")
        raw_data = result.get("data", result)
        raw_content = raw_data.get("content", "") or ""
        if not raw_content and raw_data.get("choices"):
            raw_content = raw_data["choices"][0].get("message", {}).get("content", "") or ""
        print(f"[DEBUG] Raw model response: {raw_content[:500]}", file=sys.stderr)
        catpaw_reasoning = raw_data.get("reasoning", "") or (raw_data.get("choices") and raw_data["choices"][0].get("message", {}).get("reasoning", "")) or ""
        if catpaw_reasoning:
            print(f"[DEBUG] Reasoning: {catpaw_reasoning[:300]}", file=sys.stderr)

        openai_resp = self._to_openai_response(result, model, True)
        print(f"[DEBUG] Parsed tool_calls: {len(openai_resp['choices'][0]['message'].get('tool_calls', []))}", file=sys.stderr)

        # Start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        choice = openai_resp["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]
        chunk_id = openai_resp["id"]
        created = openai_resp["created"]

        # Send initial role chunk
        role_chunk = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                      "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

        if message.get("tool_calls"):
            for i, tc in enumerate(message["tool_calls"]):
                chunk = {
                    "id": chunk_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": i, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    }]}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            chunk = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        else:
            reasoning = message.get("reasoning", "") or ""
            content = message.get("content", "") or ""

            if reasoning:
                chunk = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"reasoning": reasoning}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            chunk_size = 20
            for i in range(0, len(content), chunk_size):
                piece = content[i:i + chunk_size]
                chunk = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            chunk = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_stream(self, api_body: dict, model: str):
        """Handle pure streaming request (no tools)."""
        resp, conn = self.catpaw_client.call_stream(api_body)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        resp_headers = dict(resp.getheaders())
        resp_enc_key = resp_headers.get("encrypted-key")

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                elif line.startswith("data:"):
                    data_str = line[5:]
                else:
                    continue

                data_str = data_str.strip()
                if data_str == "[DONE]":
                    continue

                if resp_enc_key and not data_str.startswith("{"):
                    try:
                        decrypted = decrypt_response_body(data_str, resp_enc_key)
                        data = json.loads(decrypted)
                    except Exception:
                        continue
                else:
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                catpaw_data_check = data.get("data", data)
                if catpaw_data_check.get("content") == "[DONE]":
                    continue

                openai_chunk = self._convert_stream_chunk(data, model)
                if openai_chunk:
                    delta = openai_chunk["choices"][0]["delta"]
                    delta_content = delta.get("content", "")
                    if delta_content == "[DONE]":
                        continue
                    catpaw_reasoning = delta.pop("reasoning", None)
                    if catpaw_reasoning:
                        reason_chunk = copy.deepcopy(openai_chunk)
                        reason_chunk["choices"][0]["delta"] = {"reasoning": catpaw_reasoning}
                        self.wfile.write(f"data: {json.dumps(reason_chunk, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                    if delta.get("content") is not None:
                        self.wfile.write(f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:
            print(f"[ERROR] Stream error: {e}", file=sys.stderr)
            error_data = {"error": {"message": str(e), "type": "stream_error"}}
            self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            conn.close()

    def _convert_stream_chunk(self, data: dict, model: str) -> Optional[dict]:
        """Convert CatPaw streaming chunk to OpenAI format."""
        catpaw_data = data.get("data", data)
        content = ""
        reasoning = ""
        finish_reason = None

        choices = catpaw_data.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta")
            if delta and isinstance(delta, dict):
                content = (delta.get("content") or "")
                reasoning = (delta.get("reasoning") or delta.get("reasoning_content") or "")
            finish_reason = choice.get("finishReason") or choice.get("finish_reason")

        if not content and not finish_reason:
            top_content = catpaw_data.get("content") or ""
            if top_content:
                content = top_content

        if not content and not reasoning and not finish_reason:
            return None

        delta = {}
        if content:
            delta["content"] = content
        if reasoning:
            delta["reasoning"] = reasoning

        return {
            "id": catpaw_data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion.chunk",
            "created": catpaw_data.get("created", int(time.time())),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }

    def _to_openai_response(self, catpaw_resp: dict, model: str, has_tools: bool = False) -> dict:
        """Convert CatPaw response to standard OpenAI format."""
        data = catpaw_resp.get("data", catpaw_resp)
        choices = data.get("choices", [])
        content = data.get("content", "") or ""
        reasoning = (data.get("reasoning", "") or data.get("reasoning_content", "") or "")
        finish_reason = "stop"

        if choices:
            ch = choices[0]
            msg = ch.get("message", {})
            if not content:
                content = msg.get("content", "") or ""
            if not reasoning:
                reasoning = (msg.get("reasoning", "") or msg.get("reasoning_content", "") or "")
            finish_reason = ch.get("finishReason") or ch.get("finish_reason") or "stop"

        if has_tools:
            remaining_content, tool_calls = parse_tool_calls_from_content(content)
        else:
            remaining_content = content
            tool_calls = []

        message = {"role": "assistant", "content": remaining_content if remaining_content else None}
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        return {
            "id": data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion",
            "created": data.get("created", int(time.time())),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        }


def main():
    parser = argparse.ArgumentParser(description="CatPaw Bridge - OpenAI-compatible Proxy")
    parser.add_argument("--config", "-c", help="Path to config.yaml", default=None)
    parser.add_argument("--login", action="store_true", help="Start OAuth login flow and exit")
    parser.add_argument("--no-oauth", action="store_true", help="Skip OAuth login, fail if no token")
    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    print(f"[INFO] CatPaw Bridge starting on {config.server.host}:{config.server.port}", file=sys.stderr)
    print(f"[INFO] CatPaw API: https://{config.catpaw.api_host}{config.catpaw.api_path}", file=sys.stderr)
    print(f"[INFO] Available models: {', '.join(config.models)}", file=sys.stderr)

    # Initialize components
    token_manager = TokenManager(config.catpaw.state_db, config.catpaw.token_ttl)
    catpaw_client = CatPawClient(config.catpaw.api_host, config.catpaw.api_path, token_manager)

    if args.login:
        token = token_manager.login_oauth()
        if token:
            print(f"[INFO] Login successful: {token[:20]}...", file=sys.stderr)
        return

    token = token_manager.get_token()
    if token:
        print(f"[INFO] CatPaw token found: {token[:20]}...", file=sys.stderr)
        _start_heartbeat(token_manager)
    elif args.no_oauth:
        print("[WARN] CatPaw token not found. Make sure CatPaw IDE is logged in.", file=sys.stderr)
    else:
        print("[INFO] No token found, starting OAuth login...", file=sys.stderr)
        try:
            token = token_manager.login_oauth(timeout=300)
            if token:
                _start_heartbeat(token_manager)
        except Exception as e:
            print(f"[ERROR] OAuth login failed: {e}", file=sys.stderr)
            print("[WARN] Starting without token. Use --login to retry or POST /token to inject.", file=sys.stderr)

    # Set class-level config on handler
    ProxyHandler.config = config
    ProxyHandler.token_manager = token_manager
    ProxyHandler.catpaw_client = catpaw_client

    server = HTTPServer((config.server.host, config.server.port), ProxyHandler)
    print(f"[INFO] Proxy ready at http://{config.server.host}:{config.server.port}", file=sys.stderr)
    print(f"[INFO]    Login:   http://{config.server.host}:{config.server.port}/", file=sys.stderr)
    print(f"[INFO]    Models:  http://{config.server.host}:{config.server.port}/v1/models", file=sys.stderr)
    print(f"[INFO]    Chat:    http://{config.server.host}:{config.server.port}/v1/chat/completions", file=sys.stderr)
    print(f"[INFO]    Health:  http://{config.server.host}:{config.server.port}/health", file=sys.stderr)
    print(f"[INFO]    Token:   http://{config.server.host}:{config.server.port}/token (POST)", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...", file=sys.stderr)
        server.shutdown()


def _start_heartbeat(token_manager: TokenManager, interval: int = 3300):
    """Periodically refresh token from CatPaw server (every 55 minutes)."""
    def _heartbeat():
        while True:
            time.sleep(interval)
            try:
                token = token_manager.refresh_from_server()
                if token:
                    print(f"[HEARTBEAT] Token refreshed: {token[:20]}...", file=sys.stderr)
                else:
                    print("[HEARTBEAT] Token refresh failed, will retry", file=sys.stderr)
            except Exception as e:
                print(f"[HEARTBEAT] Error: {e}", file=sys.stderr)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    print(f"[INFO] Token heartbeat started (interval: {interval}s = {interval//60}min)", file=sys.stderr)


if __name__ == "__main__":
    main()
