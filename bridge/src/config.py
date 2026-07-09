"""Configuration loader for CatPaw Bridge."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 4567


@dataclass
class CatPawConfig:
    api_host: str = "catpaw.meituan.com"
    api_path: str = "/api/gpt/chat/completions"
    state_db: str = "~/Library/Application Support/CatPawAI/User/globalStorage/state.vscdb"
    token_ttl: int = 240


@dataclass
class ContextConfig:
    max_total_tokens: int = 8000
    max_system_prompt: int = 3000
    max_tool_result: int = 3000
    max_tool_prompt: int = 4000


@dataclass
class ToolsConfig:
    always_include: List[str] = field(default_factory=list)
    keyword_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "/tmp/catpaw-proxy.log"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    catpaw: CatPawConfig = field(default_factory=CatPawConfig)
    models: List[str] = field(default_factory=lambda: [
        "glm-5.2", "glm-5.1", "glm-5v-turbo",
        "deepseek-v3.2", "kimi-k2.6", "kimi-k2.5",
        "minimax-m2.7", "longcat-2.0", "longcat-flash",
    ])
    context: ContextConfig = field(default_factory=ContextConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str = None) -> "Config":
        """Load configuration from YAML file."""
        if path is None:
            # Search in order: env var, current dir, home dir
            candidates = [
                os.environ.get("CATPAW_BRIDGE_CONFIG"),
                "config.yaml",
                os.path.expanduser("~/.catpaw-bridge/config.yaml"),
                os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    path = c
                    break

        config = cls()

        if path and os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            if "server" in data:
                config.server = ServerConfig(**data["server"])
            if "catpaw" in data:
                cp = data["catpaw"]
                cp["state_db"] = os.path.expanduser(cp.get("state_db", config.catpaw.state_db))
                config.catpaw = CatPawConfig(**cp)
            if "models" in data:
                config.models = data["models"]
            if "context" in data:
                config.context = ContextConfig(**data["context"])
            if "tools" in data:
                t = data["tools"]
                config.tools = ToolsConfig(
                    always_include=t.get("always_include", []),
                    keyword_groups=t.get("keyword_groups", {}),
                )
            if "logging" in data:
                config.logging = LoggingConfig(**data["logging"])

        return config
