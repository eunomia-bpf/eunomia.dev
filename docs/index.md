---
title: Eunomia - Unlock the potential of eBPF
description: Learn eBPF programming with comprehensive tutorials, tools, and frameworks. Explore eunomia-bpf, bpftime, and the complete eBPF ecosystem for building high-performance Linux kernel programs.
keywords: eBPF, BPF, Linux kernel programming, eunomia-bpf, bpftime, eBPF tutorials, kernel tracing
hide:
  - navigation
  - toc
---

<div class="hero" markdown>
  <div class="hero-text">
    <h1>Unlock the Power of eBPF</h1>
    <p>Explore and enhance eBPF with our open-source tools and frameworks.</p>
    <a href="https://github.com/eunomia-bpf/" class="md-button md-button--primary">
      Get started
    </a>
    <a href="https://eunomia.dev/tutorials" class="md-button">
      View Tutorials
    </a>
  </div>
</div>

<style>
.hero {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 5rem 1rem 4.5rem;
  border-radius: 4px;
  overflow: hidden;
  color: #fff;
  /* Layered gradients to avoid the muddy mid-tone & add depth */
  background:
    radial-gradient(circle at 78% 60%, rgba(255,182,41,0.55), rgba(255,182,41,0) 60%),
    linear-gradient(140deg, rgba(8,27,52,0.05) 0%, rgba(255,182,41,0.08) 70%),
    linear-gradient(130deg, #061a33 0%, #0d305d 38%, #154173 55%, #1b4d85 70%, #225b95 85%);
  box-shadow: 0 4px 24px -4px rgba(0,0,0,0.35), 0 2px 4px rgba(0,0,0,0.15);
}

.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(60deg, rgba(255,255,255,0.06) 0 2px, transparent 2px 90px),
    repeating-linear-gradient(-15deg, rgba(255,255,255,0.04) 0 1px, transparent 1px 55px);
  mix-blend-mode: overlay;
  pointer-events: none;
  opacity: .55;
}
.hero-text {
  max-width: 600px;
}
.hero h1 {
  font-size: 3rem;
  font-weight: 700;
  margin-bottom: 1rem;
  letter-spacing: .5px;
  color: #fff;
}
.hero p {
  font-size: 1.2rem;
  line-height: 1.6;
  margin-bottom: 2rem;
  color: #e2e8f0;
}
.hero .md-button {
  margin: 0.5rem;
  font-weight: 600;
  backdrop-filter: blur(4px);
}
.hero .md-button--primary {
  background: linear-gradient(100deg, #4356d6 0%, #4a5fe6 60%, #5870f0 100%);
  border: none;
  box-shadow: 0 4px 16px -2px rgba(40,60,160,0.55), 0 2px 4px rgba(0,0,0,0.25);
}
.hero .md-button--primary:hover {
  filter: brightness(1.08);
}
.hero .md-button:not(.md-button--primary) {
  border: 1.5px solid rgba(255,255,255,0.45);
  color: #fff;
}
.hero .md-button:not(.md-button--primary):hover {
  border-color: #ffb629;
  color: #ffcf6b;
}
@media (max-width: 680px) {
  .hero { padding: 4rem 1rem 3.5rem; }
  .hero h1 { font-size: 2.35rem; }
  .hero p { font-size: 1.05rem; }
}
}
</style>

## About Eunomia

Eunomia Lab is an open-source organization dedicated to exploring and enhancing the eBPF ecosystem. Our mission is to innovate and optimize eBPF technologies, enabling developers to build more efficient, extensible, and powerful eBPF applications.

## Our Projects

### bpftime

![bpftime](https://eunomia.dev/bpftime/documents/bpftime.png){ align=left width="400" }

bpftime is a high-performance userspace eBPF runtime and General Extension Framework designed for userspace. It allows extending eBPF to various applications as a *General Extension Framework*, compatible with the current eBPF ecosystem. Enables faster Uprobe, USDT, Syscall hooks, XDP, GPU, and more by bypassing the kernel and utilizing an optimized compiler like LLVM.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/bpftime){ .md-button }
[Documentation :material-file-document:](https://eunomia.dev/bpftime){ .md-button }

[:material-presentation: OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) ·
[:material-presentation: eBPF Summit 2024](https://ebpf.io/summit-2024-schedule/) ·
[:material-presentation: Linux Plumbers Conference 2023](https://lpc.events/event/17/abstracts/1741/) ·
[:material-file-document: Arxiv](https://arxiv.org/abs/2311.07923)

---

### Learn eBPF by examples

![tutorial](ebpf_arch.png){ align=right width="400" }

Too much Concepts? Let's master eBPF through practical, step-by-step tutorials that focus on real, executable examples to help you learn by doing.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/bpf-developer-tutorial){ .md-button }
[Tutorial :material-school:](https://eunomia.dev/tutorials){ .md-button }

---

### llvmbpf

![llvmbpf](llvmbpf.png){ align=left width="400" }

Userspace eBPF VM with LLVM JIT/AOT compiler. It serves as the core component for bpftime without application, event, or map support.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/llvmbpf){ .md-button }
[Documentation :material-file-document:](https://eunomia.dev/bpftime/llvmbpf){ .md-button }

[:material-post: Blog](https://eunomia.dev/blogs/llvmbpf/) ·
[:material-file-document: Arxiv](https://arxiv.org/abs/2311.07923)

---

### Wasm-bpf

![Wasm-bpf](https://raw.githubusercontent.com/eunomia-bpf/wasm-bpf/refs/heads/main/docs/wasm-bpf-no-bcc.png){ align=right width="400" }

In cooperation with [WasmEdge](https://github.com/WasmEdge/WasmEdge), we built the first user-space development library, toolchain, and runtime for general eBPF programs based on WebAssembly. This allows lightweight Wasm sandboxes to deploy and control eBPF applications in Kubernetes clusters.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/wasm-bpf){ .md-button }
[Documentation :material-file-document:](https://eunomia.dev/wasm-bpf){ .md-button }

[:material-presentation: KubeCon North American 2023](https://sched.co/1R2uf) ·
[:material-file-document: Arxiv](https://arxiv.org/abs/2408.04856v1)

---

### GPTtrace

![GPTtrace](https://eunomia.dev/GPTtrace/doc/trace.png){ align=left width="400" }

The first tool to generate eBPF programs and trace the Linux kernel through natural language. With our AI agents, it can produce correct eBPF programs 80% of the time, compared to a GPT-4 baseline of 30%.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/GPTtrace){ .md-button }
[Documentation :material-file-document:](https://eunomia.dev/GPTtrace){ .md-button }

[:material-presentation: eBPF'24](https://dl.acm.org/doi/10.1145/3672197.3673434) ·
[:material-file-document: Arxiv](https://arxiv.org/abs/2311.07923)

---

### eunomia-bpf

![eunomia-bpf](https://raw.githubusercontent.com/eunomia-bpf/eunomia-bpf/master/documents/src/img/logo.png){ align=right width="400" }

A tool to help developers build, distribute, and run eBPF programs more easily using JSON and WebAssembly OCI images.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/eunomia-bpf){ .md-button }
[Documentation :material-file-document:](https://eunomia.dev/eunomia-bpf){ .md-button }

[:material-presentation: Apsara Conference 2022](https://www.alibabacloud.com/blog/eunomia-bpf-the-lightweight-development-framework-for-ebpf-and-webassembly-is-now-available_599688)
