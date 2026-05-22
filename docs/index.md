---
title: Eunomia - Unlock the potential of eBPF
description: Learn eBPF programming with comprehensive tutorials, tools, and frameworks. Explore eunomia-bpf, bpftime, and the complete eBPF ecosystem for building high-performance Linux kernel programs.
keywords: eBPF, BPF, Linux kernel programming, eunomia-bpf, bpftime, eBPF tutorials, kernel tracing
hide:
  - navigation
  - toc
---

<div class="home-cover">
  <p class="home-kicker">Open source eBPF documentation</p>
  <h1>Build practical eBPF systems with eunomia</h1>
  <p class="home-summary">
    Start with runnable tutorials, continue into userspace runtimes and toolchain docs, and connect tracing workflows to real systems.
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
    <p>Move from basic probes to sched-ext, userspace tracing, and GPU scenarios through runnable examples.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Build</p>
    <h3><a href="/bpftime/">Userspace runtime and toolchain</a></h3>
    <p>Explore bpftime, llvmbpf, and related components for fast iteration outside the kernel.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Ship</p>
    <h3><a href="/eunomia-bpf/">Package and distribute eBPF programs</a></h3>
    <p>Use eunomia-bpf to build, package, and run eBPF workloads through JSON and OCI workflows.</p>
  </div>
</div>

## Core documentation

<div class="home-project-stack">
  <div class="home-project">
    <img src="/bpftime/documents/bpftime.png" alt="bpftime" width="320" />
    <div class="home-project-copy">
      <h3><a href="/bpftime/">bpftime</a></h3>
      <p>A high-performance userspace eBPF runtime and general extension framework for fast uprobe, USDT, syscall, XDP, GPU, and related workflows.</p>
      <p class="home-project-links">
        <a href="/bpftime/">Docs</a> ·
        <a href="https://github.com/eunomia-bpf/bpftime">GitHub</a> ·
        <a href="https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng">OSDI 2025</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/ebpf_arch.png" alt="eBPF tutorial structure" width="320" />
    <div class="home-project-copy">
      <h3><a href="/tutorials/">bpf-developer-tutorial</a></h3>
      <p>A tutorial collection built around executable examples, focused on learning by running code first.</p>
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
      <p>A toolchain for building, distributing, and running eBPF programs with JSON metadata and WebAssembly OCI images.</p>
      <p class="home-project-links">
        <a href="/eunomia-bpf/">Docs</a> ·
        <a href="https://github.com/eunomia-bpf/eunomia-bpf">GitHub</a> ·
        <a href="https://www.alibabacloud.com/blog/eunomia-bpf-the-lightweight-development-framework-for-ebpf-and-webassembly-is-now-available_599688">Apsara 2022</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/llvmbpf.png" alt="llvmbpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/bpftime/llvmbpf/">llvmbpf</a></h3>
      <p>A userspace eBPF VM with LLVM JIT/AOT support, used as the compiler and execution core of bpftime.</p>
      <p class="home-project-links">
        <a href="/bpftime/llvmbpf/">Docs</a> ·
        <a href="https://github.com/eunomia-bpf/llvmbpf">GitHub</a> ·
        <a href="/blogs/llvmbpf/">Blog</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://raw.githubusercontent.com/eunomia-bpf/wasm-bpf/refs/heads/main/docs/wasm-bpf-no-bcc.png" alt="wasm-bpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/wasm-bpf/">wasm-bpf</a></h3>
      <p>WebAssembly-based eBPF tooling and runtime support for lightweight sandboxing and cloud native deployment.</p>
      <p class="home-project-links">
        <a href="/wasm-bpf/">Docs</a> ·
        <a href="https://github.com/eunomia-bpf/wasm-bpf">GitHub</a> ·
        <a href="https://sched.co/1R2uf">KubeCon NA 2023</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/GPTtrace/doc/trace.png" alt="GPTtrace" width="320" />
    <div class="home-project-copy">
      <h3><a href="/GPTtrace/">GPTtrace</a></h3>
      <p>Generate eBPF tracing programs with natural language and use LLMs to explore Linux kernel behavior.</p>
      <p class="home-project-links">
        <a href="/GPTtrace/">Docs</a> ·
        <a href="https://github.com/eunomia-bpf/GPTtrace">GitHub</a> ·
        <a href="https://dl.acm.org/doi/10.1145/3672197.3673434">eBPF 2024</a>
      </p>
    </div>
  </div>
</div>

## Keep reading

<div class="home-panel-grid">
  <div class="home-panel">
    <p class="home-panel-label">Updates</p>
    <h3><a href="/blog/">Blog and release notes</a></h3>
    <p>Read maintainer updates, research notes, and project progress inside the same documentation site.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Research</p>
    <h3><a href="/others/">Papers, talks, and references</a></h3>
    <p>Browse papers, conference talks, and background material for eunomia-related work.</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">Community</p>
    <h3><a href="https://github.com/eunomia-bpf/">GitHub organization</a></h3>
    <p>Follow repositories, releases, and experiments across the open source ecosystem.</p>
  </div>
</div>
