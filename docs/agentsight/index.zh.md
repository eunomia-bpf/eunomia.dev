# AgentSight：基于 eBPF 的零侵入 LLM 智能体可观测性工具

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/eunomia-bpf/agentsight/actions/workflows/ci.yml/badge.svg)](https://github.com/eunomia-bpf/agentsight/actions/workflows/ci.yml)

[English](https://github.com/eunomia-bpf/agentsight/blob/master/README.md) | **中文**

AgentSight 是一款专为监控 LLM 智能体行为而设计的可观测性工具，通过 SSL/TLS 流量拦截和进程监控来实现。与传统的应用级插桩不同，AgentSight 使用 eBPF 技术在系统边界进行观测，以极低的性能开销提供对 AI 智能体交互的全面洞察。

**零侵入** - 无需修改代码，无需引入新依赖，无需集成 SDK。开箱即用，兼容任何 AI 框架或应用。

## 快速开始

```bash
cargo install agentsight
# 或：wget https://github.com/eunomia-bpf/agentsight/releases/latest/download/agentsight && chmod +x agentsight
sudo agentsight top
```

<div align="center">
  <img src="docs/top-mode-demo.png" alt="AgentSight top 实时 session 视图" width="1000">
  <p><em>按 model、session token、health、进程族、工具调用、文件活动和网络活动排序的实时智能体视图</em></p>
</div>

如果使用下载到当前目录的二进制，请运行 `sudo ./agentsight top`。
AgentSight 在你忘记 sudo 时可以自动请求提权，但推荐命令仍然显式写
`sudo`。`top` 会加载 eBPF probes，也会优先读取 Claude、Codex、Gemini、OpenCode
和 OpenClaw 的本地 session 日志。

## 为什么选择 AgentSight？

### 传统可观测性 vs. 系统级监控

[LangSmith](https://docs.langchain.com/langsmith/observability-concepts)、[Langfuse](https://langfuse.com/docs/observability/overview)、[Phoenix](https://arize.com/docs/phoenix/) 这类应用级工具很适合在你拥有应用代码时追踪 trace、prompt、token、eval 和延迟。[Helicone](https://docs.helicone.ai/getting-started/integration-method/gateway) 这类 gateway/proxy 工具则适合在你可以把 provider 流量路由到托管入口时使用。

AgentSight 关注的是这些工具经常遗漏的一层：智能体在系统边界实际做了什么。它无需 SDK 或代理，就能从应用外部观察已有二进制和 CLI 智能体，并把 LLM 流量与进程执行、文件访问和系统活动关联起来。

| **挑战** | **应用级工具** | **AgentSight 方案** |
|----------|--------------|---------------------|
| **框架接入** | 每个应用需要接入 SDK、callback 或 gateway | 系统级追踪器，无需修改代码 |
| **闭源 CLI** | 只能看到工具主动暴露或记录的内容 | 从应用外部观察已有二进制和 CLI 智能体 |
| **智能体可控日志** | 日志可能不完整、被关闭或被修改 | 独立于应用日志的内核级事件 |
| **TLS LLM 流量** | 通过 SDK/代理路由时可见 | 无需代理，在 SSL/TLS 调用处捕获明文 |
| **系统动作** | 容易遗漏子进程和本地文件活动 | 追踪进程执行、文件访问和资源使用 |
| **跨边界行为** | Trace 通常停在框架或进程边界 | 关联 LLM 流量、进程事件和文件事件 |

AgentSight 能捕获应用级工具遗漏的关键交互：

- 绕过插桩的子进程执行
- SSL/TLS 调用边界处的 LLM 明文载荷
- 文件操作和系统资源访问
- 跨 LLM、进程和文件事件的边界行为

## 使用方法

### 前置要求

- **Linux 内核**：4.1+ 且支持 eBPF（推荐 5.0+）
- **sudo 权限**：eBPF probes 会按需自动提权，智能体仍以普通用户运行

从源码构建时还需要 Rust 1.88.0+、Node.js 18+、clang、llvm 和 libelf-dev。

### 安装

#### Cargo 或 Release Binary

本地使用优先安装 CLI，然后运行 `sudo agentsight top`。需要记录某个具体命令或查看历史 session 时，再参考下面的使用示例。

#### Docker

AgentSight 通过 Docker 运行，使用 `--privileged` 以支持 eBPF，`--pid=host` 以访问宿主机进程，`-v /sys:/sys:ro` 用于进程监控，`-v /usr:/usr:ro -v /lib:/lib:ro` 用于访问 SSL 库（在共享库如 `libssl.so` 上附加 uprobe 所需）。示例：

```bash
# 监控 Python AI 工具
docker run --privileged --pid=host --network=host \
  -v /sys:/sys:ro -v /usr:/usr:ro -v /lib:/lib:ro \
  -v $(pwd)/logs:/logs \
  ghcr.io/eunomia-bpf/agentsight:latest \
  record --comm python --log-file /logs/record.log

# 监控 Claude Code（挂载 home 目录以访问二进制文件）
docker run --privileged --pid=host --network=host \
  -v /sys:/sys:ro -v /usr:/usr:ro -v /lib:/lib:ro \
  -v $HOME/.local/share/claude:/claude:ro \
  -v $(pwd)/logs:/logs \
  ghcr.io/eunomia-bpf/agentsight:latest \
  record --comm claude --binary-path /claude/versions/2.1.39 --log-file /logs/record.log
```

#### 从源码构建

```bash
# 克隆仓库（含子模块）
git clone https://github.com/eunomia-bpf/agentsight.git --recursive
cd agentsight

# 安装系统依赖（Ubuntu/Debian）
make install

# 构建所有组件（前端、eBPF 和 Rust）
make build

# 或单独构建：
# make build-frontend  # 构建前端资源
# make build-bpf       # 构建 eBPF 程序
# make build-rust      # 构建 Rust collector
```

### Web 界面

`stat -- <command>` 和 `record` 默认启动 Web UI。低层 `debug trace` 需要传入 `--server`：
- **时间线视图**：http://127.0.0.1:7395/timeline
- **进程树**：http://127.0.0.1:7395/tree
- **原始日志**：http://127.0.0.1:7395/logs

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-tree.png" alt="AgentSight 演示 - 进程树可视化" width="800">
  <p><em>进程树视图，展示智能体子进程和文件活动</em></p>
</div>

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-timeline.png" alt="AgentSight 演示 - 时间线可视化" width="800">
  <p><em>时间线视图，展示 LLM、进程、文件和网络事件</em></p>
</div>

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-metrics.png" alt="AgentSight 演示 - 指标可视化" width="800">
  <p><em>指标视图，展示内存和 CPU 使用情况</em></p>
</div>

### 使用示例

#### 零配置：`record`

`record` 是追踪智能体最简单的方式。把要运行的命令写在 `record --` 后面，其余的 AgentSight 全部自动处理：

```bash
# 启动并追踪 Claude Code —— 无需 --binary-path 或 --comm
sudo ./agentsight record -- claude

# 适用于任何智能体：命令照常书写即可
sudo ./agentsight record -- claude -p "审查我的最后一次提交"
sudo ./agentsight record -- python my_agent.py
sudo ./agentsight record -- node ./cli.js
```

`record -- <command>` 自动完成的事情：

1. **发现 SSL 二进制** —— 通过 `$PATH` 解析命令，跟随符号链接（如
   `claude` → `~/.local/share/claude/versions/2.1.150`），并追踪 shebang 包装脚本
   （如 `#!/usr/bin/env node` 脚本 → 真正的 `node` ELF），使 uprobe 附加到正确的可执行文件。
2. **推导 `--comm` 进程过滤器**（来自命令名）。
3. **启动智能体** 并保持终端连接（其 TUI/REPL 正常工作），SSL + 进程 + 系统监控在后台静默运行。
4. **自动停止** —— 智能体进程退出时结束监控。

> **`sudo` 提示**：在 `sudo` 下，`record` 仍会找到*你自己*的用户级安装
> （它会读取 `$SUDO_USER` 的主目录下的 `~/.local/bin`、`~/bin` 和 `~/.nvm`），
> 因此 `sudo ./agentsight record -- claude` 追踪的是你主目录里的 claude，而不是 root `$PATH` 上的其他版本。

常用选项：`--binary-path <路径>` 覆盖自动发现，`--no-server` 关闭 web UI，
`--server-port <端口>`，`-o <日志文件>`。

#### 监控 Claude Code

Claude Code 是基于 Bun 的应用，静态链接了 BoringSSL 且符号被剥离。提供 `--binary-path` 时，AgentSight 通过字节模式匹配自动检测 BoringSSL 函数：

```bash
# 找到 Claude 二进制版本
CLAUDE_BIN=~/.local/share/claude/versions/$(claude --version | head -1)

# 记录所有 Claude 活动并启用 Web UI
sudo ./agentsight record -c claude --binary-path "$CLAUDE_BIN"
# 打开 http://127.0.0.1:7395 查看时间线

# 高级用法：使用自定义过滤器的完整追踪
sudo ./agentsight debug trace --ssl true --process true --comm claude \
  --binary-path "$CLAUDE_BIN" --server true --server-port 8080
```

这将捕获：
- **对话 API**：`POST /v1/messages` 请求，包含完整的提示词/响应 SSE 流
- **遥测数据**：心跳、事件日志、Datadog 日志
- **进程活动**：文件操作、子进程执行

> **注意**：Claude 中所有 SSL 流量都通过内部 "HTTP Client" 线程传输，而非主 "claude" 线程。当指定 `--binary-path` 时，`--comm` 过滤器会自动跳过 SSL 监控（但仍应用于进程监控），以确保流量被正确捕获。

#### 监控 Python AI 工具

```bash
# 监控 aider、open-interpreter 或任何基于 Python 的 AI 工具
sudo ./agentsight record -c "python"

# 自定义端口和日志文件
sudo ./agentsight record -c "python" --server-port 8080 --log-file /tmp/agent.log
```

#### 监控 Node.js AI 工具（Gemini CLI 等）

> **重要**：Node.js（NVM 和系统安装都一样）**将 OpenSSL 静态链接进了 `node` 二进制**——
> 没有系统 `libssl.so` 可供 hook。因此 SSL 捕获需要让 sslsniff 指向 `node` 二进制本身。

最简单的方式是 `record -- <command>`，它会自动发现 `node` 二进制：

```bash
# Gemini CLI 基于 Node 运行 —— record 会找到正确的二进制并追踪它
sudo ./agentsight record -- gemini
```

使用 `record` 时，AgentSight 现在会从 `-c node` 自动发现 Node 二进制
（检测到 Node 内嵌了 OpenSSL，于是附加到二进制而非系统库），因此无需 `--binary-path` 即可工作：

```bash
# 监控 Gemini CLI 或其他 Node.js AI 工具 —— 二进制自动发现
sudo ./agentsight record -c node

# 若自动发现选错了 Node 安装，可显式指定二进制
sudo ./agentsight record -c node --binary-path ~/.nvm/versions/node/v20.0.0/bin/node
```

> **使用 HTTP/HTTPS 代理？** 流量在 Node 进程内仍是 TLS 加密的（代理只是隧道转发），
> 因此 AgentSight 的捕获方式不变——在加密之前的 `SSL_read`/`SSL_write` 调用处捕获。

#### 高级监控

```bash
# SSL 和进程组合监控，启用 Web 界面
sudo ./agentsight debug trace --ssl true --process true --server true

# 自定义端口和日志文件
sudo ./agentsight record -c "python" --server-port 8080 --log-file /tmp/agent.log
```

#### 浏览器明文捕获

要进行浏览器特定的明文捕获，请使用独立的 `browsertrace` BPF 工具代替 `sslsniff`：

```bash
# Chrome / Chromium
sudo ./bpf/browsertrace --binary-path /opt/google/chrome/chrome

# Ubuntu Snap 上的 Firefox
sudo ./bpf/browsertrace --binary-path /snap/firefox/current/usr/lib/firefox/firefox
```

> **注意**：在 Ubuntu 上，`/usr/bin/firefox` 通常是一个包装脚本而非真正的浏览器 ELF 文件。请将 `browsertrace` 指向实际的 Firefox 二进制文件。

#### 本地 MCP（stdio 模式）

对于通过 `stdio` 而非 HTTP/TLS 通信的本地 MCP 服务器，请使用独立的 `stdiocap` BPF 工具：

```bash
# 捕获本地 MCP 服务器进程的 stdin/stdout/stderr 载荷
sudo ./bpf/stdiocap -p <mcp_server_pid>
```

AgentSight 还在 [`docs/mcp-test/README.md`](https://github.com/eunomia-bpf/agentsight/blob/master/docs/experiment/mcp-test/README.md) 下包含了一个用于本地测试的最小 MCP 测试套件。它提供了 `stdio` 和 HTTP 两种测试模式，让你可以在接入 Rust collector 之前生成可预测的 MCP 流量。

#### 直接使用 eBPF 程序

```bash
# 直接对 Claude 二进制运行 sslsniff
sudo ./bpf/sslsniff --binary-path ~/.local/share/claude/versions/2.1.39

# 对 NVM Node.js 运行 sslsniff
sudo ./bpf/sslsniff --binary-path ~/.nvm/versions/node/v20.0.0/bin/node --verbose

# 直接对 Chrome 运行 browsertrace
sudo ./bpf/browsertrace --binary-path /opt/google/chrome/chrome

# 直接对本地 MCP 服务器 PID 运行 stdiocap
sudo ./bpf/stdiocap -p 12345

# 运行进程追踪器
sudo ./bpf/process -c python
```

## 常见问题

### 通用问题

**问：AgentSight 与传统 APM 工具有什么区别？**
答：AgentSight 使用 eBPF 在内核级运行，提供独立于应用代码的系统级监控。传统 APM 需要插桩，而插桩可能被修改或禁用。

**问：性能影响如何？**
答：由于采用优化的 eBPF 内核空间数据采集，CPU 开销不到 3%。

**问：智能体能否检测到自己正在被监控？**
答：检测极其困难，因为监控在内核级进行，无需修改代码。

### 技术问题

**问：支持哪些 Linux 发行版？**
答：任何内核 4.1+（推荐 5.0+）的发行版。已在 Ubuntu 20.04+、CentOS 8+、RHEL 8+ 上测试通过。

**问：能否同时监控多个智能体？**
答：可以，使用组合监控模式可以对多个智能体进行并发观测和关联分析。

**问：如何过滤敏感数据？**
答：内置 Analyzer 可以移除认证头信息并过滤特定内容模式。

**问：为什么 AgentSight 无法捕获 Claude Code、Node.js 或 Gemini CLI 的流量？**
答：这些应用把 SSL 库静态链接进了自己的二进制（Claude/Bun 使用 BoringSSL，**所有** Node.js——NVM 和系统安装都是——使用 OpenSSL），而非使用系统 `libssl.so`，所以默认没有可供 sslsniff hook 的目标。AgentSight 已为你处理：`record -- <command>` 总会自动发现二进制，`record -c node` 现在也会自动发现 Node 二进制。对于 Claude attach 模式，请传 `--binary-path`。详见"零配置：record"和"监控 Node.js AI 工具"章节。

**问：为什么 `--comm claude` 无法捕获 SSL 流量？**
答：Claude Code 的 SSL 流量运行在内部 "HTTP Client" 线程上，而非主 "claude" 线程。sslsniff 中的 `--comm` 过滤器匹配的是线程名（来自 `bpf_get_current_comm()`），而非进程名。使用 `--binary-path` 时，collector 会自动跳过 SSL 监控的 `--comm` 过滤。

### 故障排除

**问："Permission denied" 错误**
答：确保使用 `sudo` 运行或拥有 `CAP_BPF` 和 `CAP_SYS_ADMIN` 能力。

**问："Failed to load eBPF program" 错误**
答：验证内核版本是否满足要求（见前置要求）。如需要，请为你的架构更新 vmlinux.h。


## 参与贡献

欢迎贡献！克隆并构建后（见上方安装章节），你可以：

```bash
# 运行测试
make test

# 前端开发服务器
cd frontend && npm run dev

# 使用 AddressSanitizer 构建调试版本
make -C bpf debug
```

### 关键资源

- [CLAUDE.md](https://github.com/eunomia-bpf/agentsight/blob/master/CLAUDE.md) - 项目指南和架构
- [docs/design/README.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/design/README.md) - 归档的设计笔记和研究草稿

## 许可证

MIT 许可证 - 详见 [LICENSE](https://github.com/eunomia-bpf/agentsight/blob/master/LICENSE)。

---

**AI 可观测性的未来**：随着 AI 智能体变得更加自主且具备自我修改能力，传统的可观测性方法变得力不从心。AgentSight 为大规模安全部署 AI 提供了独立的系统级监控。
