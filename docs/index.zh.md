---
title: Eunomia - eBPF 文档与工具
description: 面向文档的 eBPF 教程、运行时与开发工具，覆盖 eunomia 开源生态中的学习、构建与交付流程。
keywords: eBPF, BPF, Linux 内核开发, eunomia-bpf, bpftime, eBPF 教程, 内核追踪
hide:
  - navigation
  - toc
---

<div class="home-cover">
  <p class="home-kicker">开源 eBPF 文档站</p>
  <h1>用 eunomia 构建实用的 eBPF 系统</h1>
  <p class="home-summary">
    从可运行教程入门，继续阅读 userspace 运行时和工具链文档，再把 tracing 和扩展工作流落到实际系统里。
  </p>
  <div class="home-actions">
    <a href="/zh/tutorials/" class="md-button md-button--primary">从教程开始</a>
    <a href="/zh/bpftime/" class="md-button">阅读 bpftime 文档</a>
    <a href="https://github.com/eunomia-bpf/" class="home-inline-link">GitHub 组织</a>
  </div>
</div>

## 从这里开始

<div class="home-panel-grid">
  <div class="home-panel">
    <p class="home-panel-label">学习</p>
    <h3><a href="/zh/tutorials/">动手实践的 eBPF 教程</a></h3>
    <p>从基础 probe 到 sched-ext、userspace tracing 和 GPU 场景，按可执行示例逐步推进。</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">构建</p>
    <h3><a href="/zh/bpftime/">userspace 运行时与工具链</a></h3>
    <p>查看 bpftime、llvmbpf 及相关基础组件，理解如何在内核外高效迭代。</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">交付</p>
    <h3><a href="/zh/eunomia-bpf/">打包和分发 eBPF 程序</a></h3>
    <p>使用 eunomia-bpf 通过 JSON 与 OCI 工作流构建、打包和运行 eBPF 负载。</p>
  </div>
</div>

## 核心文档

<div class="home-project-stack">
  <div class="home-project">
    <img src="/bpftime/documents/bpftime.png" alt="bpftime" width="320" />
    <div class="home-project-copy">
      <h3><a href="/zh/bpftime/">bpftime</a></h3>
      <p>面向 userspace 的 eBPF 运行时和通用扩展框架，可覆盖快速 uprobe、USDT、syscall、XDP、GPU 等场景。</p>
      <p class="home-project-links">
        <a href="/zh/bpftime/">文档</a> ·
        <a href="https://github.com/eunomia-bpf/bpftime">GitHub</a> ·
        <a href="https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng">OSDI 2025</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/ebpf_arch.png" alt="eBPF 教程结构图" width="320" />
    <div class="home-project-copy">
      <h3><a href="/zh/tutorials/">bpf-developer-tutorial</a></h3>
      <p>围绕可执行示例组织的教程集合，强调“先跑起来，再理解原理”。</p>
      <p class="home-project-links">
        <a href="/zh/tutorials/">教程</a> ·
        <a href="https://github.com/eunomia-bpf/bpf-developer-tutorial">GitHub</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://raw.githubusercontent.com/eunomia-bpf/eunomia-bpf/master/documents/src/img/logo.png" alt="eunomia-bpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/zh/eunomia-bpf/">eunomia-bpf</a></h3>
      <p>帮助开发者通过 JSON 和 WebAssembly OCI 镜像更方便地构建、分发和运行 eBPF 程序。</p>
      <p class="home-project-links">
        <a href="/zh/eunomia-bpf/">文档</a> ·
        <a href="https://github.com/eunomia-bpf/eunomia-bpf">GitHub</a> ·
        <a href="https://www.alibabacloud.com/blog/eunomia-bpf-the-lightweight-development-framework-for-ebpf-and-webassembly-is-now-available_599688">Apsara 2022</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="/llvmbpf.png" alt="llvmbpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/zh/bpftime/llvmbpf/">llvmbpf</a></h3>
      <p>支持 LLVM JIT/AOT 的 userspace eBPF VM，也是 bpftime 的编译执行核心组件。</p>
      <p class="home-project-links">
        <a href="/zh/bpftime/llvmbpf/">文档</a> ·
        <a href="https://github.com/eunomia-bpf/llvmbpf">GitHub</a> ·
        <a href="/zh/blogs/llvmbpf/">Blog</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://raw.githubusercontent.com/eunomia-bpf/wasm-bpf/refs/heads/main/docs/wasm-bpf-no-bcc.png" alt="wasm-bpf" width="320" />
    <div class="home-project-copy">
      <h3><a href="/zh/wasm-bpf/">wasm-bpf</a></h3>
      <p>基于 WebAssembly 的 eBPF 工具链和运行时支持，适合轻量级沙箱和云原生部署场景。</p>
      <p class="home-project-links">
        <a href="/zh/wasm-bpf/">文档</a> ·
        <a href="https://github.com/eunomia-bpf/wasm-bpf">GitHub</a> ·
        <a href="https://sched.co/1R2uf">KubeCon NA 2023</a>
      </p>
    </div>
  </div>
  <div class="home-project">
    <img src="https://eunomia.dev/GPTtrace/doc/trace.png" alt="GPTtrace" width="320" />
    <div class="home-project-copy">
      <h3><a href="/zh/GPTtrace/">GPTtrace</a></h3>
      <p>用自然语言生成 eBPF 程序并追踪 Linux 内核，适合快速定位和探索内核行为。</p>
      <p class="home-project-links">
        <a href="/zh/GPTtrace/">文档</a> ·
        <a href="https://github.com/eunomia-bpf/GPTtrace">GitHub</a> ·
        <a href="https://dl.acm.org/doi/10.1145/3672197.3673434">eBPF 2024</a>
      </p>
    </div>
  </div>
</div>

## 继续阅读

<div class="home-panel-grid">
  <div class="home-panel">
    <p class="home-panel-label">更新</p>
    <h3><a href="/zh/blog/">Blog 与发布动态</a></h3>
    <p>在站内继续阅读维护者更新、研究总结和项目进展，而不是跳到单独的媒体页面。</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">研究</p>
    <h3><a href="/zh/others/">论文、演讲与参考资料</a></h3>
    <p>查看论文、会议演讲和扩展阅读，了解 eunomia 相关工作的更完整背景。</p>
  </div>
  <div class="home-panel">
    <p class="home-panel-label">社区</p>
    <h3><a href="https://github.com/eunomia-bpf/">GitHub 组织</a></h3>
    <p>跟踪仓库、release 和实验项目，继续沿着整个开源生态深入。</p>
  </div>
</div>
