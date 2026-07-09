# CatPaw Bridge Bundle

CatPaw IDE Linux 版本 + OpenAI 兼容 API 代理。

## 结构

```
.
├── bridge/            # OpenAI 兼容 API 代理服务
├── scripts/           # 启动脚本
└── CatPaw-linux.tar.xz  # CatPaw IDE Linux 版本（Release 附件）
```

## 快速开始

### 1. 下载并解压 CatPaw IDE

从 [Releases](https://github.com/Svenmax/catpaw-bundle/releases) 下载 `CatPaw-linux.tar.xz`：

```bash
tar -xf CatPaw-linux.tar.xz
```

### 2. 启动 CatPaw IDE

```bash
./CatPaw-linux/VSCode-linux-x64/bin/catpaw
```

### 3. 启动 API 代理

```bash
cd bridge && python3 proxy.py
```

默认监听 `127.0.0.1:4567`，兼容 OpenAI API 格式。

## 配置

编辑 `bridge/config.yaml`：

- `state_db`: CatPaw IDE 状态数据库路径
- `listen`: 代理监听地址

## 环境变量

- `CATPAW_TENANT` - 租户 ID（默认 `5282fa6645`）
- `CATPAW_MIS_ID` - MIS ID