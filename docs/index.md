---
title: Eunomia - eBPF documentation and tools
description: Docs-first eBPF tutorials, runtimes, and developer tooling from the eunomia open-source ecosystem.
keywords: eBPF, BPF, Linux kernel programming, eunomia-bpf, bpftime, eBPF tutorials, kernel tracing
hide:
  - navigation
  - toc
---

<div class="home-cover">
  <p class="home-kicker">Open-source eBPF documentation</p>
  <h1>Build practical eBPF systems with eunomia</h1>
  <p class="home-summary">
    Learn with runnable tutorials, explore userspace runtimes, and ship tracing or extension workflows
    from a documentation-first toolchain.
  </p>
  <div class="home-actions">
    <a href="/tutorials/" class="md-button md-button--primary">Start with tutorials</a>
    <a href="/bpftime/" class="md-button">Read bpftime docs</a>
    <a href="https://github.com/eunomia-bpf/" class="home-inline-link">GitHub organization</a>
  </div>
</div>

## Start here

<div class="home-panel-grid">
  <div class="home-panel">
    <p class="home-panel-label">Learn</p>
    <h3><a href="/tutorials/">Hands-on eBPF tutorials</a></h3>
    <p>Follow small, runnable examples that move from first probes to sched-ext, userspace tracing, and GPU cases.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Build</p>
    <h3><a href="/bpftime/">Userspace runtimes and tooling</a></h3>
    <p>Browse bpftime, llvmbpf, and related building blocks for fast iteration outside the kernel.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Ship</p>
    <h3><a href="/eunomia-bpf/">Package and distribute eBPF programs</a></h3>
    <p>Use eunomia-bpf to build, package, and run eBPF workloads with JSON and OCI-based workflows.</p>
  </div>
</div>

## Core documentation

<div class="home-project-stack">
  <div class="home-project">
    <img src="/bpftime/documents/bpftime.png" alt="bpftime" width="320" />
    <div class="home-project-copy">
      <h3><a href="/bpftime/">bpftime</a></h3>
      <p>A userspace eBPF runtime and general extension framework for fast uprobes, USDT, syscalls, XDP, GPU, and more.</p>
      <p class="home-project-links">
        <a href="/bpftime/">Documentation</a> ·
        <a href="https://github.com/eunomia-bpf/bpftime">GitHub</a> ·
        <a href="https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng">OSDI 2025</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/ebpf_arch.png" alt="eBPF tutorial architecture" width="320" />
    <div class="home-project-copy">
      <h3><a href="/tutorials/">bpf-developer-tutorial</a></h3>
      <p>Step-by-step tutorials that emphasize executable examples instead of isolated concepts.</p>
      <p class="home-project-links">
        <a href="/tutorials/">Tutorials</a> ·
        <a href="https://github.com/eunomia-bpf/bpf-developer-tutorial">GitHub</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://raw.githubusercontent.com/eunomia-bpf/eunomia-bpf/master/documents/src/img/logo.png" alt="eunomia-bpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/eunomia-bpf/">eunomia-bpf</a></h3>
      <p>Tooling for building, distributing, and running eBPF programs with JSON and WebAssembly OCI images.</p>
      <p class="home-project-links">
        <a href="/eunomia-bpf/">Documentation</a> ·
        <a href="https://github.com/eunomia-bpf/eunomia-bpf">GitHub</a> ·
        <a href="https://www.alibabacloud.com/blog/eunomia-bpf-the-lightweight-development-framework-for-ebpf-and-webassembly-is-now-available_599688">Apsara 2022</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/llvmbpf.png" alt="llvmbpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/bpftime/llvmbpf/">llvmbpf</a></h3>
      <p>A userspace eBPF VM with LLVM JIT and AOT support that serves as the compiler core for bpftime.</p>
      <p class="home-project-links">
        <a href="/bpftime/llvmbpf/">Documentation</a> ·
        <a href="https://github.com/eunomia-bpf/llvmbpf">GitHub</a> ·
        <a href="/blogs/llvmbpf/">Blog</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://raw.githubusercontent.com/eunomia-bpf/wasm-bpf/refs/heads/main/docs/wasm-bpf-no-bcc.png" alt="wasm-bpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/wasm-bpf/">wasm-bpf</a></h3>
      <p>WebAssembly-based tooling and runtime support for deploying and controlling eBPF programs in lightweight sandboxes.</p>
      <p class="home-project-links">
        <a href="/wasm-bpf/">Documentation</a> ·
        <a href="https://github.com/eunomia-bpf/wasm-bpf">GitHub</a> ·
        <a href="https://sched.co/1R2uf">KubeCon NA 2023</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://eunomia.dev/GPTtrace/doc/trace.png" alt="GPTtrace" width="320" />
    <div class="home-project-copy">
      <h3><a href="/GPTtrace/">GPTtrace</a></h3>
      <p>Natural-language-assisted eBPF tracing for Linux kernel exploration and rapid debugging workflows.</p>
      <p class="home-project-links">
        <a href="/GPTtrace/">Documentation</a> ·
        <a href="https://github.com/eunomia-bpf/GPTtrace">GitHub</a> ·
        <a href="https://dl.acm.org/doi/10.1145/3672197.3673434">eBPF 2024</a>
      </p>
    </div>
  </div>
</div>

## Keep exploring

<div class="home-panel-grid">
  <div class="home-panel">
    <p class="home-panel-label">Updates</p>
    <h3><a href="/blog/">Blog and release notes</a></h3>
    <p>Browse maintainer notes, research write-ups, and project updates without leaving the documentation site.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Research</p>
    <h3><a href="/others/">Talks, papers, and references</a></h3>
    <p>Find papers, conference talks, and supporting ecosystem material collected across the broader eunomia work.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Community</p>
    <h3><a href="https://github.com/eunomia-bpf/">GitHub organization</a></h3>
    <p>Follow repositories, releases, and experiments across the eunomia open-source ecosystem.</p>
  </div>
</div>
