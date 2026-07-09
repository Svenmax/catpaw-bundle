"""Token management - reads SSO access token from CatPaw IDE's local database."""

import json
import os
import sqlite3
import sys
import time
from typing import Optional


class TokenManager:
    """Manages CatPaw SSO access token with caching."""

    def __init__(self, state_db_path: str, ttl: int = 240):
        self.state_db = os.path.expanduser(state_db_path)
        self.ttl = ttl
        self._cache: dict = {"value": None, "ts": 0, "user_info": {}}

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
        """Get current SSO token, refreshing from DB if cache expired."""
        now = time.time()
        if self._cache["value"] and now - self._cache["ts"] < self.ttl:
            return self._cache["value"]
        return self.refresh_token()

    def refresh_token(self) -> Optional[str]:
        """Force refresh token from DB, bypassing cache."""
        now = time.time()

        try:
            conn = sqlite3.connect(self.state_db)
            cur = conn.cursor()
            token = self._read_plugin_auth(cur) or self._read_auth_provider_token(cur)
            conn.close()

            if not token:
                print("[ERROR] No CatPaw access token found in state.vscdb", file=sys.stderr)
                return None

            self._cache["value"] = token
            self._cache["ts"] = now
            return token

        except Exception as e:
            print(f"[ERROR] Failed to read token: {e}", file=sys.stderr)
            return None

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
