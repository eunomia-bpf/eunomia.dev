# AgentSight: Zero-Instrumentation LLM Agent Observability with eBPF

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/eunomia-bpf/agentsight)

AgentSight is a observability tool designed specifically for monitoring LLM agent behavior through SSL/TLS traffic interception and process monitoring. Unlike traditional application-level instrumentation, AgentSight observes at the system boundary using eBPF technology, providing tamper-resistant insights into AI agent interactions with minimal performance overhead.

**✨ Zero Instrumentation Required** - No code changes, no new dependencies, no SDKs. Works with any AI framework or application out of the box.

- Github: <https://github.com/eunomia-bpf/agentsight/>
- Arxiv: <https://arxiv.org/abs/2508.02736>

## Quick Start

```bash
wget https://github.com/eunomia-bpf/agentsight/releases/download/v0.1.1/agentsight && chmod +x agentsight
# Record agent behavior from claude
sudo ./agentsight record -c "claude"
# Record agent behavior from gemini-cli (comm is "node")
sudo ./agentsight record -c "node"
# For Python AI tools
sudo ./agentsight record -c "python"
# Record claude or gemini activity with NVM Node.js, if bundle OpenSSL statically
sudo ./agentsight record --binary-path /usr/bin/node -c node
```

Visit [http://127.0.0.1:8080](http://127.0.0.1:8080) to view the recorded data.

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-tree.png" alt="AgentSight Demo - Process Tree Visualization" width="800">
  <p><em>Real-time process tree visualization showing AI agent interactions and file operations</em></p>
</div>

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-timeline.png" alt="AgentSight Demo - Timeline Visualization" width="800">
  <p><em>Real-time timeline visualization showing AI agent interactions and system calls</em></p>
</div>

Visit [http://127.0.0.1:8080](http://127.0.0.1:8080) to view the captured data in real-time.

## 🚀 Why AgentSight?

### Traditional Observability vs. System-Level Monitoring

| **Challenge** | **Application-Level Tools** | **AgentSight Solution** |
|---------------|----------------------------|------------------------|
| **Framework Adoption** | ❌ New SDK/proxy for each framework | ✅ Drop-in daemon, no code changes |
| **Closed-Source Tools** | ❌ Limited visibility into operations | ✅ Complete visibility into prompts & behaviors |
| **Dynamic Agent Behavior** | ❌ Logs can be silenced or manipulated | ✅ Kernel-level hooks, tamper-resistant |
| **Encrypted Traffic** | ❌ Only sees wrapper outputs | ✅ Captures real unencrypted requests/responses |
| **System Interactions** | ❌ Misses subprocess executions | ✅ Tracks all process behaviors & file operations |
| **Multi-Agent Systems** | ❌ Isolated per-process tracing | ✅ Global correlation and analysis |

AgentSight captures critical interactions that application-level tools miss:

- Subprocess executions that bypass instrumentation
- Raw encrypted payloads before agent processing
- File operations and system resource access  
- Cross-agent communications and coordination

## 🏗️ Architecture

```ascii
┌─────────────────────────────────────────────────┐
│              AI Agent Runtime                   │
│   ┌─────────────────────────────────────────┐   │
│   │    Application-Level Observability      │   │
│   │  (LangSmith, Helicone, Langfuse, etc.)  │   │
│   │         🔴 Tamper Vulnerable             │   │
│   └─────────────────────────────────────────┘   │
│                     ↕ (Can be bypassed)         │
├─────────────────────────────────────────────────┤ ← System Boundary
│  🟢 AgentSight eBPF Monitoring (Tamper-proof)   │
│  ┌─────────────────┐  ┌─────────────────────┐   │
│  │   SSL Traffic   │  │    Process Events   │   │
│  │   Monitoring    │  │    Monitoring       │   │
│  └─────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Rust Streaming Analysis Framework       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────┐  │
│  │   Runners   │  │  Analyzers   │  │ Output │  │
│  │ (Collectors)│  │ (Processors) │  │        │  │
│  └─────────────┘  └──────────────┘  └────────┘  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           Frontend Visualization                │
│     Timeline • Process Tree • Event Logs       │
└─────────────────────────────────────────────────┘
```

### Core Components

1. **eBPF Data Collection** (Kernel Space)
   - **SSL Monitor**: Intercepts SSL/TLS read/write operations via uprobe hooks
   - **Process Monitor**: Tracks process lifecycle and file operations via tracepoints
   - **<3% Performance Overhead**: Operates below application layer with minimal impact

2. **Rust Streaming Framework** (User Space)
   - **Runners**: Execute eBPF programs and stream JSON events (SSL, Process, Agent, Combined)
   - **Analyzers**: Pluggable processors for HTTP parsing, chunk merging, filtering, logging
   - **Event System**: Standardized event format with rich metadata and JSON payloads

3. **Frontend Visualization** (React/TypeScript)
   - **Timeline View**: Interactive event timeline with zoom and filtering
   - **Process Tree**: Hierarchical process visualization with lifecycle tracking
   - **Log View**: Raw event inspection with syntax highlighting
   - **Real-time Updates**: Live data streaming and analysis

### Data Flow Pipeline

```
eBPF Programs → JSON Events → Runners → Analyzer Chain → Frontend/Storage/Output
```

## Usage

### Prerequisites

- **Linux kernel**: 4.1+ with eBPF support (5.0+ recommended)
- **Root privileges**: Required for eBPF program loading
- **Rust toolchain**: 1.88.0+ (for building collector)
- **Node.js**: 18+ (for frontend development)
- **Build tools**: clang, llvm, libelf-dev

## 📄 License

MIT License - see [LICENSE](https://github.com/eunomia-bpf/agentsight/blob/master/LICENSE) for details.
