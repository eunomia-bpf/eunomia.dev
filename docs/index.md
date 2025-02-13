---
title: Eunomia - Unlock the potential of eBPF
hide:
  - navigation
  - toc
---

<div class="grid cards" markdown>

-   :fontawesome-brands-linux:{ .lg .middle } __Unlock the Power of eBPF__

    ---

    Explore and enhance the eBPF ecosystem with our open-source tools and frameworks

    [:octicons-arrow-right-24: Get started](https://github.com/eunomia-bpf/){ .md-button }

</div>

<style>
.md-typeset .grid {
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    grid-gap: 1rem;
    margin: 1rem 0;
}
.md-typeset .grid.cards > :is(ul, ol) {
    display: contents;
}
.md-typeset .grid.cards > :is(ul, ol) > li,
.md-typeset .grid > .card {
    border: .05rem solid var(--md-default-fg-color--lightest);
    border-radius: .5rem;
    display: block;
    margin: 0;
    padding: 1rem;
    transition: border .25s,box-shadow .25s;
}
.md-typeset .grid.cards > :is(ul, ol) > li:focus-within,
.md-typeset .grid.cards > :is(ul, ol) > li:hover,
.md-typeset .grid > .card:focus-within,
.md-typeset .grid > .card:hover {
    border-color: #0FF1CE;
    box-shadow: var(--md-shadow-z2);
}
.lg.middle {
    font-size: 2.5rem;
    vertical-align: middle;
}
</style>

## About Eunomia

Eunomia is an open-source organization dedicated to exploring and enhancing the eBPF ecosystem. Our mission is to innovate and optimize eBPF technologies, enabling developers to build more efficient, extensible, and powerful eBPF applications.

## Our Projects

### bpftime

![bpftime](https://eunomia.dev/bpftime/documents/bpftime.png){ align=left width="400" }

bpftime is a high-performance userspace eBPF runtime and General Extension Framework designed for userspace. It allows extending eBPF to various applications as a *General Extension Framework*, compatible with the current eBPF ecosystem. Enables faster Uprobe, USDT, Syscall hooks, XDP, and more by bypassing the kernel and utilizing an optimized compiler like LLVM.

[Github :fontawesome-brands-github:](https://github.com/eunomia-bpf/bpftime){ .md-button }
[Documentation :material-file-document:](https://eunomia.dev/bpftime){ .md-button }

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
