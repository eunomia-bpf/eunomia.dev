---
title: "MCPtrace：使用bpftrace进行AI驱动的内核调试"
description: "通过bpftrace MCP服务器，使AI助手能够使用自然语言调试Linux内核问题。无需eBPF专业知识。"
keywords: MCPtrace, bpftrace, MCP服务器, AI内核调试, eBPF追踪, Linux内核, GPTtrace
---

# bpftrace MCP服务器：生成eBPF来追踪Linux内核

一个最小化的MCP（模型上下文协议）服务器，为AI助手提供访问bpftrace内核追踪功能的能力。

**GitHub 仓库**: [https://github.com/eunomia-bpf/MCPtrace](https://github.com/eunomia-bpf/MCPtrace) ⭐

**现已使用Rust实现**，使用`rmcp` crate以获得更好的性能和类型安全。Python实现仍可在git历史记录中找到。

![bpftrace MCP服务器演示](./doc/compressed_output.gif)

## 功能特性

- **AI驱动的内核调试**：使AI助手能够通过自然语言帮助您调试复杂的Linux内核问题 - 无需eBPF专业知识
- **发现系统追踪点**：浏览和搜索数千个内核探测点，找到您需要监控的确切内容 - 从系统调用到网络数据包
- **丰富的上下文和示例**：访问精心策划的生产就绪bpftrace脚本集合，用于常见的调试场景，如性能瓶颈、安全监控和系统故障排除
- **安全执行模型**：安全地运行内核追踪，而不给AI直接的root访问权限 - MCPtrace作为具有适当身份验证的安全网关
- **异步操作**：启动长时间运行的追踪并稍后检索结果 - 非常适合监控间歇性发生的生产问题
- **系统能力检测**：自动发现您的内核支持哪些追踪功能，包括可用的帮助器、映射类型和探测类型

## 为什么选择MCPtrace？

调试内核问题传统上需要深厚的eBPF专业知识。MCPtrace改变了这一点。

通过将AI助手与bpftrace（完美的eBPF追踪语言）连接起来，MCPtrace让您通过自然对话调试复杂的系统问题。只需描述您想要观察的内容 - "显示哪些进程正在打开文件"或"追踪缓慢的磁盘操作" - 然后让AI生成适当的内核追踪。

AI永远不会获得root访问权限。MCPtrace充当安全网关，凭借其丰富的示例脚本和探测信息集合，AI拥有帮助您了解内核内部情况所需的一切。无需eBPF专业知识。

## 安装

### 先决条件

1. 安装Rust（如果尚未安装）：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. 确保已安装bpftrace：

```bash
sudo apt-get install bpftrace  # Ubuntu/Debian
# 或
sudo dnf install bpftrace      # Fedora
```

### 从crates.io安装（推荐）

```bash
cargo install bpftrace-mcp-server
```

这将把`bpftrace-mcp-server`二进制文件安装到您的Cargo bin目录（通常是`~/.cargo/bin/`）。

### 从源码构建

或者，您可以从源码构建：

```bash
git clone https://github.com/yunwei37/MCPtrace
cd MCPtrace
cargo build --release
```

二进制文件将位于`./target/release/bpftrace-mcp-server`。

### 快速设置

使用我们的自动化设置脚本：

- **Claude Desktop**：`./setup/setup_claude.sh`
- **Claude Code**：`./setup/setup_claude_code.sh`

有关详细的设置说明和手动配置，请参见[setup/SETUP.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/SETUP.md)。

## 运行服务器

### 如果通过cargo install安装

```bash
bpftrace-mcp-server
```

### 如果从源码构建

```bash
./target/release/bpftrace-mcp-server
```

### 开发模式（从源码）

```bash
cargo run --release
```

### 手动配置

有关Claude Desktop或Claude Code的手动设置说明，请参见[setup/SETUP.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/SETUP.md)。

## 使用示例

### 列出系统调用探测点

```python
await list_probes(filter="syscalls:*read*")
```

### 获取BPF系统信息

```python
info = await bpf_info()
# 返回系统信息、内核帮助器、功能、映射类型和探测类型
```

### 执行简单追踪

```python
result = await exec_program(
    'tracepoint:syscalls:sys_enter_open { printf("%s\\n", comm); }',
    timeout=10
)
exec_id = result["execution_id"]
```

### 获取结果

```python
output = await get_result(exec_id)
print(output["output"])
```

## 安全说明

- 服务器需要bpftrace的sudo访问权限
- **密码处理**：创建一个包含您的sudo密码的`.env`文件：
  ```bash
  echo "BPFTRACE_PASSWD=your_sudo_password" > .env
  ```
- **替代方案**：为bpftrace配置无密码sudo：
  ```bash
  sudo visudo
  # 添加：your_username ALL=(ALL) NOPASSWD: /usr/bin/bpftrace
  ```
- 无脚本验证 - 信任AI客户端生成安全的脚本
- 资源限制：60秒最大执行时间，10k行缓冲区
- 有关详细的安全配置，请参见[SECURITY.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/SECURITY.md)

## 架构

Rust服务器使用：
- Tokio异步运行时进行并发操作
- bpftrace执行的子进程管理
- DashMap用于线程安全的内存缓冲
- 自动清理旧缓冲区
- rmcp crate用于MCP协议实现

## 限制

- 无实时流（使用get_result进行轮询）
- 简单的密码处理（生产环境需改进）
- 执行结果无持久存储
- 基本的错误处理

## 文档

- [设置指南](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/SETUP.md) - 详细的安装和配置
- [Claude Code设置](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/CLAUDE_CODE_SETUP.md) - Claude Code特定说明
- [CLAUDE.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/CLAUDE.md) - AI助手的开发指导
- [设计文档](https://github.com/eunomia-bpf/MCPtrace/blob/main/doc/mcp-bpftrace-design.md) - 架构和设计细节

## 未来增强

- 添加SSE传输以实现实时流
- 实现适当的身份验证
- 添加脚本验证和沙箱
- 支持保存/加载追踪会话
- 与eBPF程序集成