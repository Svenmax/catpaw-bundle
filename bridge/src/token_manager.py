"""Token management - reads SSO access token from CatPaw IDE's local database."""

import http.client
import json
import os
import sqlite3
import ssl
import sys
import time
import threading
from typing import Optional, Tuple


class TokenManager:
    """Manages CatPaw SSO access token with caching."""

    CACHE_FILE = "/tmp/catpaw-bridge-token.json"

    def __init__(self, state_db_path: str, ttl: int = 240):
        self.state_db = os.path.expanduser(state_db_path)
        self.ttl = ttl
        self._cache: dict = {"value": None, "ts": 0, "user_info": {}}

    def _read_cache_file(self) -> Optional[dict]:
        """Read token from persistent cache file."""
        try:
            if os.path.exists(self.CACHE_FILE):
                with open(self.CACHE_FILE, "r") as f:
                    data = json.load(f)
                ts = data.get("ts", 0)
                if time.time() - ts < data.get("ttl", 3600):
                    return data
                os.remove(self.CACHE_FILE)
        except Exception as e:
            print(f"[WARN] Cache file read failed: {e}", file=sys.stderr)
        return None

    def _write_cache_file(self, token: str, refresh_token: str = None, ttl: int = 3600):
        """Write token to persistent cache file."""
        try:
            data = {
                "accessToken": token,
                "ts": time.time(),
                "ttl": ttl,
            }
            if refresh_token:
                data["refreshToken"] = refresh_token
            with open(self.CACHE_FILE, "w") as f:
                json.dump(data, f)
            os.chmod(self.CACHE_FILE, 0o600)
        except Exception as e:
            print(f"[WARN] Cache file write failed: {e}", file=sys.stderr)

    def set_token_from_external(self, token: str, refresh_token: str = None) -> bool:
        """Accept token from external source (e.g. Bridge /token endpoint)."""
        self._cache["value"] = token
        self._cache["ts"] = time.time()
        self._write_cache_file(token, refresh_token)
        print(f"[INFO] Token set from external source: {token[:20]}...", file=sys.stderr)
        return True

    def _read_plugin_auth(self, cur) -> Optional[str]:
        """Read the auth token CatPaw's IDEKit plugin uses for API calls."""
        cur.execute("SELECT value FROM ItemTable WHERE key = 'mt-idekit.mt-idekit-code'")
        row = cur.fetchone()
        if not row:
            return None

        data = json.loads(row[0])
        token = data.get("accessTokenprod") or data.get("accessToken")
        user_info = data.get("userInfoprod") or data.get("userInfo") or {}
        if isinstance(user_info, str):
            try:
                user_info = json.loads(user_info)
            except json.JSONDecodeError:
                user_info = {}
        if isinstance(user_info, dict):
            self._cache["user_info"] = user_info
        return token

    def _read_auth_provider_token(self, cur) -> Optional[str]:
        """Fallback to VS Code authentication provider storage."""
        cur.execute(
            "SELECT value FROM ItemTable WHERE key = 'catpaw.mt-authentication'"
        )
        row = cur.fetchone()
        if not row:
            return None

        data = json.loads(row[0])
        mt_auth = json.loads(data["mt.auth"])
        sessions = mt_auth.get("sessions", [])
        if not sessions:
            print("[ERROR] No sessions in mt.auth", file=sys.stderr)
            return None
        return sessions[0].get("accessToken")

    def get_token(self) -> Optional[str]:
        """Get current SSO token: cache file → memory → DB → server refresh."""
        now = time.time()
        if self._cache["value"] and now - self._cache["ts"] < self.ttl:
            return self._cache["value"]

        # Try persistent cache file
        cached = self._read_cache_file()
        if cached and cached.get("accessToken"):
            self._cache["value"] = cached["accessToken"]
            self._cache["ts"] = cached.get("ts", 0)
            print(f"[INFO] Token loaded from cache file: {cached['accessToken'][:20]}...", file=sys.stderr)
            return cached["accessToken"]

        return self.refresh_token()

    def _read_refresh_token_and_access(self, cur) -> Tuple[Optional[str], Optional[str]]:
        """Read refreshToken and accessToken from auth extension keychain storage."""
        cur.execute("SELECT value FROM ItemTable WHERE key = 'catpaw.mt-authentication'")
        row = cur.fetchone()
        if not row:
            return None, None

        data = json.loads(row[0])
        mt_auth = json.loads(data.get("mt.auth", "{}"))
        refresh_token = mt_auth.get("refreshToken")
        sessions = mt_auth.get("sessions", [])
        access_token = sessions[0].get("accessToken") if sessions else None
        return refresh_token, access_token

    def _call_refresh_api(self, access_token: str, refresh_token: str) -> Optional[dict]:
        """Call CatPaw refreshToken API to get new access token."""
        tenant = os.environ.get("CATPAW_TENANT", "5282fa6645")
        body = json.dumps({"refreshToken": refresh_token})

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        conn = http.client.HTTPSConnection("catpaw.meituan.com", context=ctx)
        conn.request(
            "POST",
            "/api/login/refreshToken",
            body,
            {
                "Content-Type": "application/json",
                "Catpaw-Auth": access_token,
                "tenant": tenant,
            },
        )
        resp = conn.getresponse()
        resp_body = resp.read().decode("utf-8")
        conn.close()

        if resp.status != 200:
            print(f"[ERROR] refreshToken API returned HTTP {resp.status}: {resp_body[:500]}", file=sys.stderr)
            return None

        result = json.loads(resp_body)
        if result.get("code") != 0:
            print(f"[ERROR] refreshToken API error: code={result.get('code')} msg={result.get('msg', '')}", file=sys.stderr)
            return None

        data = result.get("data", result)
        print(f"[INFO] Token refreshed: expires={data.get('expires')}, refreshExpires={data.get('refreshExpires')}", file=sys.stderr)
        return data

    def refresh_from_server(self) -> Optional[str]:
        """Try to refresh access token by calling CatPaw refreshToken API."""
        try:
            conn = sqlite3.connect(self.state_db)
            cur = conn.cursor()
            refresh_token, access_token = self._read_refresh_token_and_access(cur)
            conn.close()

            if not refresh_token or not access_token:
                print("[ERROR] No refresh token or access token in keychain", file=sys.stderr)
                return None

            result = self._call_refresh_api(access_token, refresh_token)
            if not result:
                return None

            new_access_token = result.get("accessToken") or result.get("access_token")
            new_refresh_token = result.get("refreshToken") or result.get("refresh_token")

            if not new_access_token:
                print("[ERROR] refreshToken API returned no new access token", file=sys.stderr)
                return None

            self._cache["value"] = new_access_token
            self._cache["ts"] = time.time()
            self._cache["refreshed_from_server"] = True

            self._write_cache_file(new_access_token, new_refresh_token)

            if new_refresh_token:
                print(f"[INFO] Server returned new refreshToken as well", file=sys.stderr)

            return new_access_token

        except Exception as e:
            print(f"[ERROR] Server refresh failed: {e}", file=sys.stderr)
            return None

    def refresh_token(self) -> Optional[str]:
        """Force refresh token: DB → cache file → server refresh."""
        now = time.time()

        try:
            conn = sqlite3.connect(self.state_db)
            cur = conn.cursor()
            token = self._read_plugin_auth(cur) or self._read_auth_provider_token(cur)
            conn.close()

            if token:
                self._cache["value"] = token
                self._cache["ts"] = now
                self._write_cache_file(token)
                return token

            # Try cache file for refreshToken
            cached = self._read_cache_file()
            if cached and cached.get("refreshToken") and cached.get("accessToken"):
                print("[INFO] DB token not found, trying refresh with cached refreshToken...", file=sys.stderr)
                result = self._call_refresh_api(cached["accessToken"], cached["refreshToken"])
                if result:
                    new_token = result.get("accessToken") or result.get("access_token")
                    new_rt = result.get("refreshToken") or result.get("refresh_token")
                    if new_token:
                        self._cache["value"] = new_token
                        self._cache["ts"] = time.time()
                        self._write_cache_file(new_token, new_rt)
                        return new_token

            print("[INFO] DB token not found, trying server-side refresh...", file=sys.stderr)
            return self.refresh_from_server()

        except Exception as e:
            print(f"[ERROR] Failed to read token from DB: {e}", file=sys.stderr)
            print("[INFO] Trying server-side refresh...", file=sys.stderr)
            return self.refresh_from_server()

    def write_to_state_db(self, access_token: str, refresh_token: str = ""):
        """Write token directly to state.vscdb so Bridge reads it on next restart."""
        try:
            conn = sqlite3.connect(self.state_db)
            cur = conn.cursor()

            data = json.dumps({
                "sessions": [{
                    "id": "bridge-oauth",
                    "accessToken": access_token,
                    "account": {"label": "oauth", "id": "oauth", "userInfoId": ""},
                    "scopes": ["sankuai"],
                }],
                "refreshToken": refresh_token,
            })

            cur.execute(
                "SELECT value FROM ItemTable WHERE key = 'catpaw.mt-authentication'"
            )
            row = cur.fetchone()
            if row:
                existing = json.loads(row[0])
                existing["mt.auth"] = data
                cur.execute(
                    "UPDATE ItemTable SET value = ? WHERE key = 'catpaw.mt-authentication'",
                    (json.dumps(existing),),
                )
            else:
                cur.execute(
                    "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
                    ("catpaw.mt-authentication", json.dumps({"mt.auth": data})),
                )

            conn.commit()
            conn.close()
            print(f"[INFO] Token written to state.vscdb: {access_token[:20]}...", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to write token to state.vscdb: {e}", file=sys.stderr)

    def login_oauth(self, interactive: bool = True, timeout: int = 300) -> Optional[str]:
        """Perform QR Code OAuth login, get tokens, store them."""
        from .oauth_login import QRCodeOAuthLogin

        oauth = QRCodeOAuthLogin()
        if interactive:
            result = oauth.login_interactive(timeout)
        else:
            qr = oauth.get_qrcode()
            print(f"[LOGIN] QR Code URL: {qr['qrCodeImageUrl']}", file=sys.stderr)
            print(f"[LOGIN] Waiting for scan (timeout: {timeout}s)...", file=sys.stderr)
            result = oauth.poll_access_token(qr["code"], timeout)

        self._cache["value"] = result.access_token
        self._cache["ts"] = time.time()
        self._write_cache_file(result.access_token, result.refresh_token, result.expires)
        self.write_to_state_db(result.access_token, result.refresh_token)

        print(f"[INFO] OAuth login successful: {result.access_token[:20]}...", file=sys.stderr)
        return result.access_token

    def get_token_with_oauth(self, interactive: bool = True, timeout: int = 300) -> Optional[str]:
        """Get token with automatic OAuth fallback when no token is available."""
        token = self.get_token()
        if token:
            return token
        print("[INFO] No token found, starting OAuth login...", file=sys.stderr)
        return self.login_oauth(interactive, timeout)

    def build_headers(self, token: str, encrypted_key: str = None) -> dict:
        """Build request headers for CatPaw API."""
        user_info = self._cache.get("user_info") or {}
        mis_id = os.environ.get("CATPAW_MIS_ID", "") or user_info.get("misId", "")
        tenant = os.environ.get("CATPAW_TENANT", "5282fa6645")
        is_external = tenant == "5282fa6645"

        headers = {
            "Content-Type": "application/json",
            "Catpaw-Auth": token,
            "tenant": tenant,
            "client-type": "CatPaw IDE",
            "ide-version": "2026.6.0",
            "plugin-id": "mt-idekit.mt-idekit-code",
            "plugin-version": "2026.6.0",
        }

        if mis_id:
            headers["user-mis-id"] = mis_id

        if is_external:
            headers["platform"] = "linux-x64"
        else:
            passport_key = os.environ.get("CATPAW_PASSPORT_KEY", "1d47d6ff96")
            sso_key = os.environ.get("CATPAW_SSO_KEY", "f32a546874")
            primary_cookie_name = os.environ.get("CATPAW_PRIMARY_COOKIE", f"{passport_key}_ssoid")
            headers["Cookie"] = f"{primary_cookie_name}={token}; {sso_key}_ssoid={token}"
            headers["ide-type"] = "CatPaw IDE"
            headers["client-env"] = "LOCAL_IDE"
            headers["platform-info"] = "linux-x64"

        if encrypted_key:
            headers["encrypted-key"] = encrypted_key
        return headers
