"""QR Code and Phone OAuth login for CatPaw - obtains tokens without IDE."""

import http.client
import json
import ssl
import time
import uuid
import urllib.request
from typing import Optional, Dict, Tuple

from .crypto import decrypt_response_body


LOGIN_HEADERS = {
    "client-type": "CatPaw IDE",
    "tenant": "5282fa6645",
    "platform": "linux-x64",
    "ide-version": "2026.6.0",
}


class QRCodeLoginResult:
    def __init__(self, access_token: str, refresh_token: str, expires: int, refresh_expires: int):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires = expires
        self.refresh_expires = refresh_expires


class _BaseLogin:
    """Shared HTTP client for CatPaw login APIs."""

    def __init__(self, base_url: str = "https://catpaw.meituan.com"):
        self.base_url = base_url.rstrip("/")
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _request(self, method: str, path: str, body: Optional[bytes] = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = dict(LOGIN_HEADERS)
        if body:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        resp = urllib.request.urlopen(req, context=self._ssl_ctx, timeout=30)
        return json.loads(resp.read())

    def _request_encrypted(self, method: str, path: str, body: Optional[bytes] = None) -> dict:
        """Send request and decrypt encrypted response (used by /api/login/mobile)."""
        host = self.base_url.replace("https://", "").split("/")[0]
        headers = dict(LOGIN_HEADERS)
        if body:
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPSConnection(host, context=self._ssl_ctx)
        conn.request(method, path, body, headers)
        resp = conn.getresponse()
        resp_headers = dict(resp.getheaders())
        resp_body = resp.read().decode("utf-8")
        conn.close()

        enc_key = resp_headers.get("encrypted-key")
        if enc_key:
            encrypted_data = resp_body.strip('"')
            decrypted = decrypt_response_body(encrypted_data, enc_key)
            return json.loads(decrypted)
        return json.loads(resp_body)


class QRCodeOAuthLogin(_BaseLogin):
    """CatPaw QR Code OAuth login flow.

    1. GET /api/login/qrcode → get QR code image URL + polling code
    2. User scans QR code with WeChat
    3. POST /api/login/accessToken {code} → poll until scanned → get tokens
    """

    def get_qrcode(self) -> Dict:
        """Fetch a QR code for WeChat scanning."""
        data = self._request("GET", "/api/login/qrcode")
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to get QR code: {data.get('msg', 'unknown')}")
        return data["data"]

    def poll_access_token(self, code: str, timeout: int = 300, interval: int = 3) -> QRCodeLoginResult:
        """Poll access token endpoint until user scans QR code."""
        body = json.dumps({"code": code}).encode()
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self._request("POST", "/api/login/accessToken", body)
            if data.get("code") != 0:
                raise RuntimeError(f"Access token poll failed: {data.get('msg', 'unknown')}")
            result = data["data"]
            if result.get("scanned") and result.get("accessToken"):
                return QRCodeLoginResult(
                    access_token=result["accessToken"],
                    refresh_token=result.get("refreshToken", ""),
                    expires=result.get("expires", 3600),
                    refresh_expires=result.get("refreshExpires", 86400),
                )
            time.sleep(interval)
        raise TimeoutError("QR code scan timed out")

    def login_interactive(self, timeout: int = 300) -> QRCodeLoginResult:
        """Full interactive login: displays QR code URL and waits for scan."""
        qr = self.get_qrcode()
        print(f"\n{'=' * 60}", flush=True)
        print(f"[LOGIN] Scan this QR code with WeChat to log in:", flush=True)
        print(f"[LOGIN] QR Code URL: {qr['qrCodeImageUrl']}", flush=True)
        print(f"[LOGIN] Expires at: {time.strftime('%H:%M:%S', time.localtime(qr['expireTime'] / 1000))}", flush=True)
        print(f"{'=' * 60}\n", flush=True)
        print(f"[LOGIN] Waiting for scan (timeout: {timeout}s)...", flush=True)
        return self.poll_access_token(qr["code"], timeout)


class PhoneOAuthLogin(_BaseLogin):
    """CatPaw phone verification code login flow.

    1. POST /api/login/sendSmsVerificationCode {mobileNo, uuid} → send SMS code
    2. POST /api/login/mobile {mobileNo, verificationCode, uuid} → verify and get tokens
    """

    def __init__(self, base_url: str = "https://catpaw.meituan.com"):
        super().__init__(base_url)
        self._session_uuid = str(uuid.uuid4())

    def send_code(self, mobile_no: str) -> str:
        """Send verification code to the given phone number. Returns requestCode."""
        body = json.dumps({
            "mobileNo": mobile_no,
            "uuid": self._session_uuid,
        }).encode()
        data = self._request("POST", "/api/login/sendSmsVerificationCode", body)
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to send SMS code: {data.get('msg', 'unknown')}")
        return data["data"]["requestCode"]

    def login(self, mobile_no: str, verification_code: str) -> QRCodeLoginResult:
        """Verify code and get tokens."""
        body = json.dumps({
            "mobileNo": mobile_no,
            "verificationCode": verification_code,
            "uuid": self._session_uuid,
        }).encode()
        data = self._request_encrypted("POST", "/api/login/mobile", body)
        if data.get("code") != 0:
            raise RuntimeError(f"Phone login failed: {data.get('msg', 'unknown')}")
        result = data["data"]
        return QRCodeLoginResult(
            access_token=result["accessToken"],
            refresh_token=result.get("refreshToken", ""),
            expires=result.get("expires", 3600),
            refresh_expires=result.get("refreshExpires", 86400),
        )