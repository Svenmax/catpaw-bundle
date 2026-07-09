"""CatPaw API client - handles encrypted HTTP requests to CatPaw backend."""

import http.client
import json
import ssl
import sys
import time
from typing import Dict, Any, Tuple

from .crypto import encrypt_request_body, decrypt_response_body
from .token_manager import TokenManager

MAX_RETRIES = 3
RETRY_DELAY = 1


class CatPawClient:
    """Client for making encrypted requests to CatPaw API."""

    def __init__(self, host: str, path: str, token_manager: TokenManager):
        self.host = host
        self.path = path
        self.token_manager = token_manager
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _get_token(self) -> str:
        token = self.token_manager.get_token()
        if not token:
            raise RuntimeError("CatPaw token not found. Make sure CatPaw IDE is logged in.")
        return token

    def _do_call(self, request_body: dict) -> dict:
        """Send encrypted request and return decrypted JSON response (single attempt)."""
        token = self._get_token()
        plaintext = json.dumps(request_body, ensure_ascii=False)
        encrypted_body, encrypted_key = encrypt_request_body(plaintext)
        headers = self.token_manager.build_headers(token, encrypted_key)

        conn = http.client.HTTPSConnection(self.host, context=self._ssl_ctx)
        conn.request("POST", self.path, encrypted_body, headers)
        resp = conn.getresponse()
        resp_headers = dict(resp.getheaders())
        resp_body = resp.read().decode("utf-8")
        status = resp.status
        conn.close()

        if status == 401:
            raise AuthError(f"CatPaw API auth failed: {resp_body[:500]}")
        if status != 200:
            raise RuntimeError(f"CatPaw API returned HTTP {status}: {resp_body[:500]}")

        resp_enc_key = resp_headers.get("encrypted-key")
        if resp_enc_key:
            encrypted_data = resp_body.strip('"')
            decrypted = decrypt_response_body(encrypted_data, resp_enc_key)
            return json.loads(decrypted)
        else:
            return json.loads(resp_body)

    def call(self, request_body: dict) -> dict:
        """Send encrypted request with auto-retry on auth failure."""
        for attempt in range(MAX_RETRIES):
            try:
                return self._do_call(request_body)
            except AuthError:
                if attempt < MAX_RETRIES - 1:
                    print(f"[WARN] Auth failed, refreshing token and retrying ({attempt + 1}/{MAX_RETRIES})...", file=sys.stderr)
                    self.token_manager.refresh_token()
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"CatPaw API auth failed after {MAX_RETRIES} retries")

    def call_stream(self, request_body: dict) -> Tuple[http.client.HTTPResponse, http.client.HTTPSConnection]:
        """Send encrypted request for streaming response with auto-retry on auth failure."""
        for attempt in range(MAX_RETRIES):
            token = self._get_token()
            request_body["stream"] = True
            plaintext = json.dumps(request_body, ensure_ascii=False)
            encrypted_body, encrypted_key = encrypt_request_body(plaintext)
            headers = self.token_manager.build_headers(token, encrypted_key)
            headers["Accept"] = "text/event-stream"

            conn = http.client.HTTPSConnection(self.host, context=self._ssl_ctx)
            conn.request("POST", self.path, encrypted_body, headers)
            resp = conn.getresponse()

            if resp.status == 401:
                resp_body = resp.read().decode("utf-8")
                conn.close()
                if attempt < MAX_RETRIES - 1:
                    print(f"[WARN] Auth failed, refreshing token and retrying ({attempt + 1}/{MAX_RETRIES})...", file=sys.stderr)
                    self.token_manager.refresh_token()
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"CatPaw API auth failed after {MAX_RETRIES} retries")
            else:
                return resp, conn


class AuthError(Exception):
    """Raised when CatPaw API returns 401."""
    pass
