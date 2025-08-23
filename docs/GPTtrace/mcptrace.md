---
title: "MCPtrace: AI-Powered Kernel Debugging with bpftrace"
description: "Enable AI assistants to debug Linux kernel issues through natural language using bpftrace MCP server. No eBPF expertise required."
keywords: MCPtrace, bpftrace, MCP server, AI kernel debugging, eBPF tracing, Linux kernel, GPTtrace
---

# bpftrace MCP Server: generate eBPF to trace linux kernel

A minimal MCP (Model Context Protocol) server that provides AI assistants with access to bpftrace kernel tracing capabilities.

**GitHub Repository**: [https://github.com/eunomia-bpf/MCPtrace](https://github.com/eunomia-bpf/MCPtrace) ⭐

**Now implemented in Rust** using the `rmcp` crate for better performance and type safety. The Python implementation is still available in the git history.

![bpftrace MCP Server Demo](./doc/compressed_output.gif)

## Features

- **AI-Powered Kernel Debugging**: Enable AI assistants to help you debug complex Linux kernel issues through natural language - no eBPF expertise required
- **Discover System Trace Points**: Browse and search through thousands of kernel probes to find exactly what you need to monitor - from system calls to network packets
- **Rich Context and Examples**: Access a curated collection of production-ready bpftrace scripts for common debugging scenarios like performance bottlenecks, security monitoring, and system troubleshooting
- **Secure Execution Model**: Run kernel traces safely without giving AI direct root access - MCPtrace acts as a secure gateway with proper authentication
- **Asynchronous Operation**: Start long-running traces and retrieve results later - perfect for monitoring production issues that occur intermittently
- **System Capability Detection**: Automatically discover what tracing features your kernel supports, including available helpers, map types, and probe types

## Why MCPtrace?

Debugging kernel issues traditionally requires deep eBPF expertise. MCPtrace changes that.

By bridging AI assistants with bpftrace (the perfect eBPF tracing language), MCPtrace lets you debug complex system issues through natural conversation. Just describe what you want to observe - "show me which processes are opening files" or "trace slow disk operations" - and let AI generate the appropriate kernel traces.

AI never gets root access. MCPtrace acts as a secure gateway, and with its rich collection of example scripts and probe information, AI has everything needed to help you understand what's happening inside your kernel. No eBPF expertise required.

## Installation

### Prerequisites

1. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Ensure bpftrace is installed:
```bash
sudo apt-get install bpftrace  # Ubuntu/Debian
# or
sudo dnf install bpftrace      # Fedora
```

### Install from crates.io (Recommended)

```bash
cargo install bpftrace-mcp-server
```

This will install the `bpftrace-mcp-server` binary to your Cargo bin directory (usually `~/.cargo/bin/`).

### Build from Source

Alternatively, you can build from source:

```bash
git clone https://github.com/yunwei37/MCPtrace
cd MCPtrace
cargo build --release
```

The binary will be available at `./target/release/bpftrace-mcp-server`.

### Quick Setup

Use our automated setup scripts:

- **Claude Desktop**: `./setup/setup_claude.sh`
- **Claude Code**: `./setup/setup_claude_code.sh`

For detailed setup instructions and manual configuration, see [setup/SETUP.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/SETUP.md).

## Running the Server

### If installed via cargo install
```bash
bpftrace-mcp-server
```

### If built from source
```bash
./target/release/bpftrace-mcp-server
```

### Development mode (from source)
```bash
cargo run --release
```

### Manual Configuration

For manual setup instructions for Claude Desktop or Claude Code, see [setup/SETUP.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/SETUP.md).

## Usage Examples

### List System Call Probes
```python
await list_probes(filter="syscalls:*read*")
```

### Get BPF System Information
```python
info = await bpf_info()
# Returns system info, kernel helpers, features, map types, and probe types
```

### Execute a Simple Trace
```python
result = await exec_program(
    'tracepoint:syscalls:sys_enter_open { printf("%s\\n", comm); }',
    timeout=10
)
exec_id = result["execution_id"]
```

### Get Results
```python
output = await get_result(exec_id)
print(output["output"])
```

## Security Notes

- The server requires sudo access for bpftrace
- **Password Handling**: Create a `.env` file with your sudo password:
  ```bash
  echo "BPFTRACE_PASSWD=your_sudo_password" > .env
  ```
- **Alternative**: Configure passwordless sudo for bpftrace:
  ```bash
  sudo visudo
  # Add: your_username ALL=(ALL) NOPASSWD: /usr/bin/bpftrace
  ```
- No script validation - trust the AI client to generate safe scripts
- Resource limits: 60s max execution, 10k lines buffer
- See [SECURITY.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/SECURITY.md) for detailed security configuration

## Architecture

The Rust server uses:
- Tokio async runtime for concurrent operations
- Subprocess management for bpftrace execution
- DashMap for thread-safe in-memory buffering
- Automatic cleanup of old buffers
- rmcp crate for MCP protocol implementation

## Limitations

- No real-time streaming (use get_result to poll)
- Simple password handling (improve for production)
- No persistent storage of executions
- Basic error handling

## Documentation

- [Setup Guide](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/SETUP.md) - Detailed installation and configuration
- [Claude Code Setup](https://github.com/eunomia-bpf/MCPtrace/blob/main/setup/CLAUDE_CODE_SETUP.md) - Claude Code specific instructions
- [CLAUDE.md](https://github.com/eunomia-bpf/MCPtrace/blob/main/CLAUDE.md) - Development guidance for AI assistants
- [Design Document](https://github.com/eunomia-bpf/MCPtrace/blob/main/doc/mcp-bpftrace-design.md) - Architecture and design details

## Future Enhancements

- Add SSE transport for real-time streaming
- Implement proper authentication
- Add script validation and sandboxing
- Support for saving/loading trace sessions
- Integration with eBPF programs