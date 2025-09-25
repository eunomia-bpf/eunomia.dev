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

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
