"""CatPaw API encryption/decryption module.

Implements RSA-OAEP + AES-128-ECB encryption scheme matching the CatPaw IDE plugin.
RSA keys are extracted from the plugin source code (XOR-obfuscated).
"""

import base64
import secrets
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Util.Padding import pad, unpad

# XOR key for decoding obfuscated RSA keys from CatPaw plugin source
XOR_KEY = "ThisIsMyXorKey"

# XOR-obfuscated RSA public key (for encrypting request AES key)
KEY1_ENCODED = (
    "eUVEXmQxCD4RIVIbMDsYISpTAjYUVHVCX2ZvNB0hKzojMgM7PwQDIw4QE1EeQwsyHDweLjMEJjgF"
    "UCg+ADoPOj8kMQo0PBUdGTgxF2Y8DxwEMiobI1wqIi0QBnMLDjc7ExQQWV49OAIEMG47GD4QKhIkU"
    "AYvQz0SLl4WGxwvOAoMOy0SAz9hNR4hKlI3OAAgegA7NWk3XX4cQHspYzYaWCw4Ky0LIgABFyceCg"
    "siOg4sHgZ6FC8dH1gwekIVNBwbNj4rODcOORAPBiIyEgslOFYpDgQKFS4kOQwiLj1BXBQEGlskGEU"
    "FFW4dPwYkFw0HXBcgNx1WNRpEAFc9NztcMHwFJhEeLEc/Vy0lJgUxezl1GBs2OQMODiYsWhcjJEcL"
    "GQ4Bc0o0NjEsO3FDOTINOgYfNjcfGBwmBB4sPg0/NSkRSBI8AgUnHSwLOlslAD03YVo4CjoYOg9pA"
    "DF+VTIWYlwYH0EIKGo+WQMiKzkRHwQNIgEuKFoEHA0WZywqNhMkdVYpNyIOFhQ+HhFcLhUMDw8ECC"
    "wMID05IEYlJT5MCAN4CBIwECk4Mgt5YFR1Ql8OKz10ODwxBToOWRMqK2ZIVHlF"
)

# XOR-obfuscated RSA private key (for decrypting response AES key)
KEY2_ENCODED = (
    "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"
    "tPQUhUeUVENgc3bSkKJiQKMTx0IywqZF5gVHU="
)


def _xor_decode(encoded: str) -> str:
    """Decode XOR-obfuscated string from CatPaw plugin source."""
    raw = base64.b64decode(encoded)
    result = bytearray()
    for i in range(len(raw)):
        result.append(raw[i] ^ ord(XOR_KEY[i % len(XOR_KEY)]))
    return result.decode("utf-8")


# RSA keys (decoded once at module load)
PUB_KEY = RSA.import_key(_xor_decode(KEY1_ENCODED))
PRIV_KEY = RSA.import_key(_xor_decode(KEY2_ENCODED))


def encrypt_request_body(plaintext: str) -> tuple[str, str]:
    """
    Encrypt request body using AES-128-ECB + RSA-OAEP.
    Returns (encrypted_body_base64, encrypted_key_base64).
    """
    aes_key = secrets.token_bytes(16)
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(plaintext.encode("utf-8"), AES.block_size))
    encrypted_b64 = base64.b64encode(encrypted).decode()

    # RSA encrypt the base64-encoded AES key
    aes_key_b64 = base64.b64encode(aes_key).decode()
    cipher_rsa = PKCS1_OAEP.new(PUB_KEY)
    encrypted_key = cipher_rsa.encrypt(aes_key_b64.encode("utf-8"))
    encrypted_key_b64 = base64.b64encode(encrypted_key).decode()

    return encrypted_b64, encrypted_key_b64


def decrypt_response_body(encrypted_b64: str, enc_key_b64: str) -> str:
    """
    Decrypt response body: RSA-OAEP decrypt AES key -> AES-128-ECB decrypt body.
    """
    cipher_rsa = PKCS1_OAEP.new(PRIV_KEY)
    rsa_decrypted = cipher_rsa.decrypt(base64.b64decode(enc_key_b64))
    aes_key = base64.b64decode(rsa_decrypted.decode("utf-8"))

    cipher = AES.new(aes_key, AES.MODE_ECB)
    raw = base64.b64decode(encrypted_b64)
    decrypted = cipher.decrypt(raw)
    return unpad(decrypted, AES.block_size).decode("utf-8")
