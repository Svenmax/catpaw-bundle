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

import json
import os
import re
import sys
import time
import uuid
import argparse
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

    def _send_sse_error(self, msg: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        error_data = {"error": {"message": msg, "type": "proxy_error"}}
        self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    # ── Routes ────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/v1/models":
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

        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
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
                self._handle_non_stream(api_body, model)
        except Exception as e:
            print(f"[ERROR] Proxy error: {e}", file=sys.stderr)
            try:
                self._send_json(500, {"error": {"message": str(e), "type": "proxy_error"}})
            except Exception:
                pass

    # ── Response handlers ────────────────────────────────────────────

    def _handle_non_stream(self, api_body: dict, model: str):
        """Handle non-streaming request."""
        result = self.catpaw_client.call(api_body)

        if result.get("code") != 0:
            raise RuntimeError(f"CatPaw API error: {result.get('msg', 'unknown')}")

        raw_data = result.get("data", result)
        raw_content = raw_data.get("content", "")
        if not raw_content and raw_data.get("choices"):
            raw_content = raw_data["choices"][0].get("message", {}).get("content", "")
        print(f"[DEBUG] Raw model response: {raw_content[:500]}", file=sys.stderr)

        openai_resp = self._to_openai_response(result, model)
        print(f"[DEBUG] Parsed tool_calls: {len(openai_resp['choices'][0]['message'].get('tool_calls', []))}", file=sys.stderr)
        self._send_json(200, openai_resp)

    def _handle_stream_with_tools(self, api_body: dict, model: str):
        """Handle streaming request with tools: internal non-stream, external simulated stream."""
        api_body["stream"] = False
        result = self.catpaw_client.call(api_body)

        if result.get("code") != 0:
            raise RuntimeError(f"CatPaw API error: {result.get('msg', 'unknown')}")

        raw_data = result.get("data", result)
        raw_content = raw_data.get("content", "")
        if not raw_content and raw_data.get("choices"):
            raw_content = raw_data["choices"][0].get("message", {}).get("content", "")
        print(f"[DEBUG] Raw model response: {raw_content[:500]}", file=sys.stderr)

        openai_resp = self._to_openai_response(result, model)
        print(f"[DEBUG] Parsed tool_calls: {len(openai_resp['choices'][0]['message'].get('tool_calls', []))}", file=sys.stderr)

        # Start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        choice = openai_resp["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]
        chunk_id = openai_resp["id"]
        created = openai_resp["created"]

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
            content = message.get("content", "") or ""
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
        self.send_header("Connection", "keep-alive")
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
                    delta_content = openai_chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta_content == "[DONE]":
                        continue
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
        finish_reason = None

        choices = catpaw_data.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta")
            if delta and isinstance(delta, dict):
                content = delta.get("content", "")
            finish_reason = choice.get("finishReason") or choice.get("finish_reason")

        if not content and not finish_reason:
            top_content = catpaw_data.get("content", "")
            if top_content:
                content = top_content

        if not content and not finish_reason:
            return None

        return {
            "id": catpaw_data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion.chunk",
            "created": catpaw_data.get("created", int(time.time())),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }],
        }

    def _to_openai_response(self, catpaw_resp: dict, model: str) -> dict:
        """Convert CatPaw response to standard OpenAI format."""
        data = catpaw_resp.get("data", catpaw_resp)
        choices = data.get("choices", [])
        content = data.get("content", "")
        finish_reason = "stop"

        if choices:
            ch = choices[0]
            if not content:
                content = ch.get("message", {}).get("content", "")
            finish_reason = ch.get("finishReason") or ch.get("finish_reason") or "stop"

        remaining_content, tool_calls = parse_tool_calls_from_content(content)

        message = {"role": "assistant", "content": remaining_content if remaining_content else None}
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
    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    print(f"[INFO] CatPaw Bridge starting on {config.server.host}:{config.server.port}", file=sys.stderr)
    print(f"[INFO] CatPaw API: https://{config.catpaw.api_host}{config.catpaw.api_path}", file=sys.stderr)
    print(f"[INFO] Available models: {', '.join(config.models)}", file=sys.stderr)

    # Initialize components
    token_manager = TokenManager(config.catpaw.state_db, config.catpaw.token_ttl)
    catpaw_client = CatPawClient(config.catpaw.api_host, config.catpaw.api_path, token_manager)

    token = token_manager.get_token()
    if token:
        print(f"[INFO] ✅ CatPaw token found: {token[:20]}...", file=sys.stderr)
    else:
        print("[WARN] ⚠️  CatPaw token not found. Make sure CatPaw IDE is logged in.", file=sys.stderr)

    # Set class-level config on handler
    ProxyHandler.config = config
    ProxyHandler.token_manager = token_manager
    ProxyHandler.catpaw_client = catpaw_client

    server = HTTPServer((config.server.host, config.server.port), ProxyHandler)
    print(f"[INFO] 🚀 Proxy ready at http://{config.server.host}:{config.server.port}/v1", file=sys.stderr)
    print(f"[INFO]    Models:  http://{config.server.host}:{config.server.port}/v1/models", file=sys.stderr)
    print(f"[INFO]    Chat:    http://{config.server.host}:{config.server.port}/v1/chat/completions", file=sys.stderr)
    print(f"[INFO]    Health:  http://{config.server.host}:{config.server.port}/health", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
