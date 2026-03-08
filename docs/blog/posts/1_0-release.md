---
date: 2024-02-11
---

# Introducing eunomia-bpf v1.0: Simplifying eBPF with CO-RE and WebAssembly

The world of eBPF (Extended Berkeley Packet Filter) has been rapidly evolving, offering developers powerful tools to monitor and modify the behavior of systems at the kernel level. Today, we're thrilled to introduce the latest milestone in this journey - **eunomia-bpf v1.0**. This release is a testament to our commitment to simplifying and enhancing the eBPF development experience with CO-RE (Compile Once, Run Everywhere) and WebAssembly.
<!-- more -->

## Introduction

**eunomia-bpf** is not just another tool in the eBPF ecosystem. It's a dynamic loading library/runtime and a compile toolchain framework designed with a singular vision - to make building and distributing eBPF programs easier and more efficient.

With the rise of cloud-native applications and the need for fine-grained system monitoring, eBPF has become an indispensable tool for developers. However, the complexities associated with writing, compiling, and distributing eBPF programs have often been a barrier. This is where eunomia-bpf steps in, offering a suite of tools and features that streamline the entire process.

## What's New in eunomia-bpf v1.0

The eunomia-bpf v1.0 release marks a significant milestone in our journey to simplify and enhance the eBPF development experience. Here's a deep dive into the new features and enhancements that cater to both novice and experienced eBPF developers:

- **Expanded Architectural Support**:
  - **`aarch64` Compatibility**: Recognizing the growing adoption of ARM-based systems, we've expanded our support to include the `aarch64` architecture. Whether you're on `x86_64` or `aarch64`, expect a consistent eunomia-bpf experience.
  - **Cross-Compilation**: Our commitment to `aarch64` goes beyond mere compatibility. Given our build servers are x86_64-based, we've employed cross-compilation techniques to produce `aarch64` executable files and docker images. For the tech-savvy, our build scripts and workflow files are readily available in our repository.

- **Reduced External Dependencies with AppImage**:
  - **Self-Contained Binaries**: Our precompiled binaries for `ecc` and `ecli` are now packaged as AppImages. This means all required dependency libraries are bundled within, eliminating pesky version conflicts, especially with libraries like `glibc` and `libclang`.
  - **Universal Compatibility**: Thanks to static linking with `libfuse`, the released `ecli` and `ecc` AppImages can run seamlessly across distributions, independent of the locally provided `glibc`. All you need is kernel support for fuse. For those interested, our repository houses the workflow files detailing the AppImage construction with all dependencies.

- **Diverse Attachment Types**:
  - **Broadened eBPF Program Support**: Version `1.0` introduces support for a variety of eBPF program types, including:
    - **tc**: Monitor traffic control with precision.
    - **xdp**: Keep an eye on XDP-related packets.
    - **profile**: Gain insights into the kernel stack and user stack activities on specific processor cores.
  - **Hands-on Learning**: To help developers get started, we offer tests and examples tailored for each of these attachment types.

- **Revamped OCI in ecli**:
  - **Modular and Efficient**: The OCI component of ecli has been meticulously refactored, leading to a more modular and efficient design.
  - **Introducing `ecli-server`**: With the new `ecli-server`, developers can harness the power of OpenAPI to run ecli programs on their local machines remotely. Whether you're using the OpenAPI interface to run and fetch logs or executing programs remotely via `ecli`, the experience is as intuitive as local runs.

- **bpf-compatible Evolution**:
  - **Enhanced Cross-Kernel Execution**: Building on our legacy of supporting cross-kernel version execution without local BTF dependencies, v1.0 takes it a step further.
  - **A Dedicated Project**: We've carved out this functionality into a standalone project, [bpf-compatible](https://github.com/eunomia-bpf/bpf-compatible). This project focuses on leveraging btfhub to trim BTF files across different distributions and kernel versions. The trimmed files are then embedded into the executable, ready to be accessed via specific APIs.

## Core Features of eunomia-bpf

eunomia-bpf v1.0 is more than just a set of new features. It's a culmination of our vision to provide a comprehensive framework for eBPF development:

- **Simplified eBPF Program Writing**: With eunomia-bpf, developers can focus solely on writing the kernel code. The framework takes care of data exposure, command-line argument generation, and more.

- **Building eBPF with Wasm**: Our integration with WebAssembly opens up new avenues for eBPF development. Whether you're working in C/C++, Rust, Go, or other languages, eunomia-bpf has got you covered.

- **Distributing eBPF Programs with Ease**: Our tools for pushing, pulling, and running pre-compiled eBPF programs as OCI images in Wasm modules simplify the distribution process. Plus, with support for dynamic loading through JSON or Wasm modules, deployment is a breeze.

## Getting Started with eunomia-bpf

Ready to dive in? Here are some resources to kickstart your journey with eunomia-bpf:

- **Github Template**: Explore our template at [eunomia-bpf/ebpf-template](https://github.com/eunomia-bpf/ebpm-template) for a hands-on introduction.

- **Example BPF Programs**: Delve into real-world applications with our [example programs](https://github.com/eunomia-bpf/eunomia-bpf/blob/master/examples/bpftools).

- **Tutorials**: For a deeper understanding, check out our comprehensive [developer tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial).

## Conclusion

eunomia-bpf v1.0 is more than just a release; it's a promise. A promise to continually simplify and enhance the eBPF development experience. As the eBPF landscape evolves, so will eunomia-bpf, ensuring that developers always have the best tools at their disposal.

For more insights and detailed documentation, head over to <https://github.com/eunomia-bpf/eunomia-bpf>
