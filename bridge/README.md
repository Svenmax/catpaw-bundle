# CatPaw Bridge - OpenAI Compatible Proxy

将 [CatPaw IDE](https://catpaw.meituan.com) 的私有 LLM API 桥接为标准 OpenAI 兼容端点，供 [Hermes Agent](https://github.com/) 或其他支持 OpenAI API 的工具调用。

## ✨ 特性

- **OpenAI 兼容**：对外暴露标准 `/v1/chat/completions` 和 `/v1/models` 端点
- **自动加解密**：RSA-OAEP + AES-128-ECB，与 CatPaw 插件一致
- **SSO 认证**：自动从 CatPaw IDE 的本地数据库读取 access token
- **智能工具过滤**：根据用户问题动态选择相关工具，不再注入全部 127 个工具
- **Tool Calling 翻译层**：CatPaw API 不支持原生 `tools` 参数，通过 prompt 注入 + 响应解析实现
- **Token 精确计数**：使用 `tiktoken` 进行准确的 token 计数（非字符估算）
- **智能上下文管理**：工具结果摘要而非粗暴截断，自动保留最近对话
- **流式响应**：支持 SSE 流式输出
- **YAML 配置**：所有参数通过 `config.yaml` 配置，不再硬编码
- **模块化架构**：各功能模块独立，便于维护和扩展

## 项目结构

```
catpaw-bridge/
├── proxy.py                 # 主入口 - HTTP 服务器
├── config.yaml              # 配置文件
├── requirements.txt         # Python 依赖
├── src/
│   ├── __init__.py
│   ├── config.py            # 配置加载器
│   ├── crypto.py            # RSA/AES 加解密模块
│   ├── token_manager.py     # SSO token 管理
│   ├── catpaw_client.py     # CatPaw API 客户端
│   ├── tool_filter.py       # 智能工具过滤
│   ├── tool_translator.py   # Tool calling 翻译层
│   ├── token_counter.py     # Token 计数 (tiktoken)
│   └── context_manager.py   # 上下文管理 + 智能截断
└── README.md
```

## 支持的模型

| 模型 | 说明 |
|------|------|
| `glm-5.2` | 智谱 GLM-5.2 |
| `glm-5.1` | 智谱 GLM-5.1 |
| `deepseek-v3.2` | DeepSeek V3.2 |
| `kimi-k2.6` | Kimi K2.6 |
| `kimi-k2.5` | Kimi K2.5 |
| `minimax-m2.7` | MiniMax M2.7 |
| `longcat-2.0` | LongCat 2.0 |
| `longcat-flash` | LongCat Flash |

## 架构原理

```
Hermes Agent                CatPaw Bridge                       CatPaw API
    │                            │                                  │
    │ POST /v1/chat/completions  │                                  │
    │ (OpenAI format + tools)    │                                  │
    │ ─────────────────────────> │                                  │
    │                            │ 1. Smart filter: 127 tools → 15  │
    │                            │ 2. Inject tools to system prompt │
    │                            │ 3. Token-based context truncate  │
    │                            │ 4. AES+RSA encrypt request       │
    │                            │ ───────────────────────────────> │
    │                            │                                  │
    │                            │ 5. RSA+AES decrypt response      │
    │                            │ 6. Parse <tool_call> tags        │
    │                            │ 7. Convert to OpenAI tool_calls  │
    │                            │ <─────────────────────────────── │
    │ (标准 OpenAI 响应)          │                                  │
    │ <───────────────────────── │                                  │
```

### 智能工具过滤

当 Hermes 发送 127 个工具时，不再全部注入 system prompt（会撑爆上下文），而是：

1. **始终包含核心工具**：`terminal_exec`、`file_read`、`file_write` 等
2. **关键词匹配**：扫描用户最近 5 条消息，匹配工具类别
   - 用户问"硬盘" → 包含 `terminal_exec`、`file_list`
   - 用户问"飞书文档" → 包含 `feishu_*` 系列工具
   - 用户问"浏览器" → 包含 `browser_*` 系列工具
3. **结果**：从 127 个工具（15000+ 字符）减少到 10-20 个（2000-3000 字符）

### 上下文管理

- **Token 精确计数**：使用 `tiktoken`（GPT-4 同款分词器）准确计算 token 数
- **智能摘要**：工具结果不再粗暴截断，而是保留首尾关键行 + 中间省略
- **历史截断**：保留 system prompt + 最近对话，自动截断旧消息
- **工具结果限制**：单个工具结果最多 3000 字符

## 快速开始

### 前置条件

- 已安装并登录 [CatPaw IDE](https://catpaw.meituan.com)
- Python 3.10+
- 网络可访问 `catpaw.meituan.com`（公网可达，不需要 VPN）

### 安装

```bash
git clone https://github.com/fifasheng-tech/catpaw-bridge.git
cd catpaw-bridge
pip install -r requirements.txt
```

### 配置

编辑 `config.yaml`，或设置环境变量：

```bash
# 可选：从 CatPaw IDE 插件源码中获取这些值
export CATPAW_MIS_ID="your_mis_id"
export CATPAW_TENANT="your_tenant_id"
export CATPAW_PASSPORT_KEY="your_passport_key"
export CATPAW_SSO_KEY="your_sso_key"
```

### 启动

```bash
python proxy.py
# 或指定配置文件
python proxy.py --config /path/to/config.yaml
```

### 验证

```bash
# 健康检查
curl http://127.0.0.1:4567/health

# 查看可用模型
curl http://127.0.0.1:4567/v1/models

# 发送聊天请求
curl http://127.0.0.1:4567/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 请求简短的可见思路，并用 OpenAI-compatible reasoning_content 流式字段返回。
# thinking 不是模型内部隐藏推理开关；它要求模型生成简短、可见的理由说明。
curl -N http://127.0.0.1:4567/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "stream": true,
    "thinking": true,
    "messages": [{"role": "user", "content": "计算 27 乘以 43"}]
  }'

# 带工具调用
curl http://127.0.0.1:4567/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "查看磁盘空间"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "terminal_exec",
        "description": "Execute a terminal command",
        "parameters": {
          "type": "object",
          "properties": {"command": {"type": "string"}},
          "required": ["command"]
        }
      }
    }]
  }'
```

### IDE Agent 兼容接口

Bridge 额外暴露了 CatPaw IDE 中 agent 相关命令的 HTTP 入口，不依赖 VS Code webview：

```bash
# 查看支持的 IDE Agent 能力
curl http://127.0.0.1:4567/v1/ide/capabilities

# 调用“解释代码”能力
curl http://127.0.0.1:4567/v1/ide/agent \
  -H "Content-Type: application/json" \
  -d '{
    "action": "explain",
    "model": "glm-5.2",
    "selection": "function add(a, b) { return a + b }"
  }'
```

`action` 支持：`chat`、`send_prompt`、`explain`、`bug`、`test`、`comment`、`refactor`、`commit_message`、`testagent_selected_file`、`testagent_all_changes`、`agent_review`、`agent_review_changes`、`deploy_plan`。也可以直接传 CatPaw IDE command ID，例如 `idekit.mcopilot.explain.selected` 或 `catpaw.triggerAgentReview`。

请求可携带：`prompt`、`selection`/`selected_text`、`files`、`diff`、`diagnostics`、`workspace`、`context`。Bridge 会把这些上下文整理成 CatPaw Agent 风格的消息，再走现有加密 API、token 和模型配置。

### Remote Agent 兼容接口

CatPaw IDE 的 Remote Agent 主要是一个 webview 壳：先通过 `conversationId` 查询远程 Pod 信息，再把返回的 `podUrl` 嵌入界面。Bridge 提供同等能力。当前 CatPaw IDE 仅支持 `ssh://git@git.sankuai.com/...` 的内部仓库作为 Remote Agent 工作区；GitHub 和其他外部 URL 会在 Bridge 本地以明确错误拒绝，不再转发成上游的“系统服务异常”。

```bash
# 创建 Remote Agent 任务
curl http://127.0.0.1:4567/v1/remote-agent/create \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "修复测试失败并提交总结",
    "gitRepoUrl": "ssh://git@git.sankuai.com/group/repo.git",
    "gitBaseBranch": "main",
    "gitCheckoutBranch": "agent/fix-tests",
    "modelType": "minimax-m2.7",
    "autoPullRequest": false,
    "autoDeploy": false
  }'

# 查询 Remote Agent 详情
curl 'http://127.0.0.1:4567/v1/remote-agent/detail?conversationId=xxx'

# 等待 Pod ready，并返回 podUrl
curl 'http://127.0.0.1:4567/v1/remote-agent/wait?conversationId=xxx&timeout=120'

# 返回可直接打开的 iframe 容器页
open 'http://127.0.0.1:4567/v1/remote-agent/open?conversationId=xxx'

# 只取 podUrl，或 redirect=1 直接跳转
curl 'http://127.0.0.1:4567/v1/remote-agent/pod?conversationId=xxx'
```

Remote Agent 的思考过程渲染发生在 `podUrl` 指向的远程 webapp 内；Bridge 复刻的是 VS Code 侧查询 detail、等待 pod、打开容器页的能力。

### Native Agent Long Tasks

`/v1/chat/completions` remains a one-shot chat API. For IDE-style long tasks, use the native Agent conversation endpoints. They preserve a CatPaw `conversationId` and use the same upstream lifecycle as the IDE: create, connect to its event stream, continue, and cancel. The task repository must be an internal `ssh://git@git.sankuai.com/...` URL that the logged-in CatPaw account can access.

```bash
# 1. Create a task. The response has id and stream_url.
curl http://127.0.0.1:4567/v1/agent/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Inspect the test failures, fix them, and report verification.",
    "gitRepoUrl": "ssh://git@git.sankuai.com/group/repo.git",
    "gitBaseBranch": "main",
    "gitCheckoutBranch": "agent/fix-tests",
    "modelType": "minimax-m2.7"
  }'

# 2. Subscribe to native Agent progress. Standard chunks carry content or
# reasoning_content; event: catpaw.agent preserves each raw CatPaw event.
curl -N 'http://127.0.0.1:4567/v1/agent/conversations/CONVERSATION_ID/stream?messageIndex=0'

# 3. Send a follow-up or supply fields accepted by CatPaw's native continue API.
curl http://127.0.0.1:4567/v1/agent/conversations/CONVERSATION_ID/continue \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Run the focused tests and continue."}'

# 4. Request cancellation.
curl -X POST http://127.0.0.1:4567/v1/agent/conversations/CONVERSATION_ID/cancel
```

The bridge does not execute agent-supplied commands locally. Native Agent/Pod tool execution stays in CatPaw's task environment. `reasoning_content` is mapped only from visible plan, thinking, or tool events and is not hidden model chain-of-thought.

Clients restricted to `/v1/chat/completions` can opt into the same long-task lifecycle with the nonstandard `catpaw_agent` object. `stream: true` is required. The initial call creates a native Agent conversation; follow-ups reuse its `conversationId`. Read the initial `event: catpaw.agent` event to capture the conversation ID for the next request.

```bash
# Initial task through the standard OpenAI chat URL.
curl -N http://127.0.0.1:4567/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "stream": true,
    "messages": [{"role":"user","content":"Fix the failing tests and report verification."}],
    "catpaw_agent": {
      "gitRepoUrl": "ssh://git@git.sankuai.com/group/repo.git",
      "gitBaseBranch": "main",
      "gitCheckoutBranch": "agent/fix-tests"
    }
  }'

# Continue that long task through chat/completions.
curl -N http://127.0.0.1:4567/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "stream": true,
    "messages": [{"role":"user","content":"Now run the complete suite."}],
    "catpaw_agent": {"conversationId":"CONVERSATION_ID"}
  }'
```

This is an extension understood by CatPaw Bridge, not a field standardized by OpenAI. A client that cannot add custom JSON fields cannot request a native long task through chat completions alone; it must use the dedicated conversation endpoints or an adapter that adds `catpaw_agent`.

### 配置 Hermes Agent

在 Hermes 的 `config.yaml` 中添加：

```yaml
providers:
  catpaw:
    type: openai
    base_url: http://127.0.0.1:4567/v1
    api_key: any-string
    models:
      glm-5.2:
        context_length: 262144
```

### Launchd 自启服务（macOS）

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.catpaw.bridge-proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python3</string>
        <string>/path/to/catpaw-bridge/proxy.py</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardErrorPath</key><string>/tmp/catpaw-proxy.log</string>
    <key>StandardOutPath</key><string>/tmp/catpaw-proxy.log</string>
</dict>
</plist>
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容的聊天补全接口 |
| `/v1/models` | GET | 返回可用模型列表 |
| `/v1/ide/capabilities` | GET | 返回 bridge 暴露的 CatPaw IDE Agent 能力 |
| `/v1/ide/agent` | POST | 以 IDE 命令语义调用 Agent，如解释/查 bug/生成测试/评审 |
| `/v1/agent/conversations` | POST | 创建原生 CatPaw 长任务会话，返回 `conversationId` 与 stream URL |
| `/v1/agent/conversations/{id}/stream` | GET | 订阅原生 Agent 事件，输出 OpenAI chunk 与 `catpaw.agent` SSE 事件 |
| `/v1/agent/conversations/{id}/continue` | POST | 继续原生 Agent 会话 |
| `/v1/agent/conversations/{id}/cancel` | POST | 请求停止原生 Agent 会话 |
| `/v1/remote-agent/create` | POST | 创建 Remote Agent 任务，返回 `conversationId` |
| `/v1/remote-agent/detail` | GET | 查询 Remote Agent conversation detail |
| `/v1/remote-agent/wait` | GET | 轮询等待 Remote Agent Pod ready |
| `/v1/remote-agent/open` | GET | 返回嵌入 `podUrl` 的 Remote Agent 容器页 |
| `/v1/remote-agent/pod` | GET | 返回或跳转到 Remote Agent `podUrl` |
| `/health` | GET | 健康检查，返回 token 状态 |

## 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `server.host` | 127.0.0.1 | 监听地址 |
| `server.port` | 4567 | 监听端口 |
| `context.max_total_tokens` | 8000 | CatPaw API 的 token 限制 |
| `context.max_system_prompt` | 3000 | 原始 system prompt 最大字符数 |
| `context.max_tool_result` | 3000 | 单个工具结果最大字符数 |
| `context.max_tool_prompt` | 4000 | 工具列表注入最大字符数 |
| `tools.always_include` | [...] | 始终注入的核心工具列表 |

## ⚠️ 免责声明

本项目仅供学习和研究用途。使用前请确保：
1. 你已了解并遵守 CatPaw IDE 的使用条款
2. 你不会将代理用于商业用途或分享给他人
3. 敏感信息（MIS ID、tenant 等）已通过环境变量配置，不要硬编码

## License

MIT
