# talk draft in kubecon

## IntroductionOpening (1 minute)

Welcome, everyone.

Thank you for joining me to discuss the innovative intersection of eBPF and WebAssembly—technologies revolutionizing observability within our systems.

Combining eBPF with Wasm provides a robust solution for non-intrusive deployment and advanced security checks within Kubernetes pods. Today, we'll explore how these technologies can be leveraged for efficient and secure software development. Let's dive into a detailed discussion on their benefits, challenges, and the future they hold.

My name is Yusheng Zheng, currently maintaining a small community called eunomia-bpf, building open source projects to make eBPF easier to use, and exploring new technologies and runtimes related to eBPF.

## Slide 3: Agenda (2 minutes)

In the next minute, we'll outline our exploration into eBPF and WebAssembly, or Wasm—a journey into enhancing the deployment and security of eBPF programs. We begin by introducing eBPF, the kernel-level technology that powers efficient networking and security, followed by Wasm, which brings a multi-language, secure execution environment to user space.

We'll uncover how Wasm streamlines the eBPF experience, enabling non-intrusive deployments into Kubernetes pods, offering separation from application workloads for greater flexibility. We'll emphasize the simplicity of implementing security with declarative checks at deployment and discuss how eBPF data analytics can drive insights for performance tuning. By supporting user-space eBPF, it opens the door to more secure and versatile observability tools.

Finally, we'll flip the script and look at eBPF through the lens of Wasm, enhancing security sandboxing, host call observability, and providing a robust debugging toolkit. This bidirectional enrichment is pivotal for cloud-native development, and we're excited to delve into these transformative technologies with you.

## Slide 4: Introduction to eBPF (1 minutes)

First, let's talk about eBPF, or Extended Berkeley Packet Filter. This is not just a technology but a paradigm shift, allowing developers to dynamically and safely program the Linux kernel. It's at the heart of performance-sensitive tasks like networking and security, and it's changing the game for kernel-level instrumentation.

## Slide 5: Introduction to WebAssembly and WasmEdge (Wasm) (1 minutes)

Turning our attention to WebAssembly, or Wasm, we enter the realm of a revolutionary binary format designed for user space security. Wasm is not only language-agnostic, supporting a wide array of programming languages, but it's also incredibly lightweight and swift, making it ideal for performance-critical applications.

What sets Wasm apart is its capability-based security model, which allows fine-grained access to host resources, ensuring that applications remain secure while performing at their best. Moreover, Wasm's cross-platform nature facilitates an unprecedented level of portability, enabling code to run consistently across different environments.

## Slide 6: How Wasm Improves eBPF Developer Experience (3 minutes)

Stepping into the developer's shoes, we see how Wasm refines the eBPF development workflow.

## Slide 7-10: eBPF Deployment Models (5 minutes)

We'll dissect the eBPF deployment models, contrasting the integrated control plane against the decoupled sidecar approach. The former, despite its direct control benefits, raises concerns about security and multi-user conflicts. The latter, while modular, introduces complexity in maintaining consistency and kernel feature integration.

## Slide 11-12: Wasm + eBPF: The Synergy (3 minutes)

Here, we'll illustrate a typical Kubernetes pod setup, integrating eBPF into containers running in an LXC environment, alongside other workload-specific containers. This hybrid model capitalizes on the strengths of Wasm and eBPF, creating a robust, modular observability framework.

## Slide 13-14: WasmEdge eBPF plugin (2 minutes)

We'll showcase the WasmEdge eBPF plugin, wasm-bpf. Its compactness, ease of management, and security enhancements over traditional containerized eBPF deployments signal a leap forward in deploying eBPF programs.

## Slide 15-17: Faster, easier & safer eBPF deployment and Trade-offs (3 minutes)

However, there are trade-offs. The migration of libraries and toolchains to this new model is not trivial, with considerations around limited eBPF features in Wasm environments. But the familiar development experience, akin to that provided by libbpf-bootstrap, is a testament to our progress.

## Slice 16: Use container tools to run Wasm + eBPF

This is a demo of how to use container tools, like the podman to run Wasm + eBPF. As the image shows, we can using a podman container to start and run a eBPF program in WebAssembly, which can trace the run queue (scheduler) latency as a histogram for the linux system. The eBPF program is integrated into the userspace WebAssembly application, which can be packed into a OCI image and started as a container with the WasmEdge runtime.

We can also list the existing podman containers, and see the container we just started, or stop and remove the container.

## Slide 17: Developer Experience

libbpf-bootstrap is a widely used framework for developing eBPF programs in C/C++.
The developer experience in WebAssembly is similar to that of libbpf-bootstrap, includes automatically generating skeleton (bpf code framework) and type definitions, just like the bpftool and libbpf-bootstrap does.

## Slide 18: Examples

"Let's take a moment to walk through some hands-on examples where eBPF in WebAssembly can support.

First, we have 'Uprobe' for Observability or Tracing. This is like setting up a watchtower inside your applications, letting you keep an eye on how functions are running without modify them.

Next is 'XDP' for Networking. This can be used to process packets at the lowest level before they reach the network stack, allowing you to filter, redirect, or drop packets as needed.

And then we have 'LSM' for Security. This allows you to set rules on what the system can and cannot do, like blocking a process from accessing a file or network port in the kernel.

## Slide 19: Challenges (2 mins)

However, there are challenges to overcome before we can fully use eBPF's capabilities within Wasm for Kubernetes.

Firstly, we've got some of libraries for C/C++, Rust, and Go, each enabling eBPF interactions in their respective languages. These are libbpf for C/C++, libbpf-rs for Rust, and cilium/go for Go. We need to port these libraries to Wasm to enabling developing eBPF programs in these languages with WebAssembly.

Another challenge is the data layout. The data layout of eBPF programs, which is 64 bit, maybe different from that of Wasm, maybe 32 bit or 64 bit. We need to convert the data layout of kernel eBPF programs to the correct data layout of Wasm programs, when we need to communicate between kernel eBPF programs and userspace Wasm programs.

The last challenge is the kernel compatibility. The eBPF programs may need a specific kernel version to run and also require enable the kernel configuration. We can use Compile-Once, Run Everywhere (CO-RE) technology to enhance the portability of eBPF programs, but we still need to ensure that the kernel version is compatible with the eBPF programs. We can also look at a userspace eBPF runtimeto run eBPF programs in userspace, or a compatibility layer for different kernel features.

## Slide 20: The Operational Framework of wasm-bpf (2 minutes)

"Let's take a look at how wasm-bpf works.  This exciting project transforms WebAssembly into a space where it can run alongside the OS, similar to eBPF's role in user-space.  This means we can use Wasm to develop and deploy eBPF userspace programs, but with the added benefits of security and ease of moving from one system to another.

In the runtime,  a Wasm module can managing multiple eBPF programs, and allow`dynamically load` eBPF programs from the Wasm sandbox into the kernel, select the desired event to attach them, unattach  them, control the complete lifecycle of multiple eBPF objects, and support most eBPF program types.

Communication is a two-way street with wasm-bpf. It sets up a path for back-and-forth conversations with the kernel using eBPF Maps, making data transfer smooth and efficient with ring buffers.  The eBPF can efficient sending of messages from the kernel state to the user state (and vice versa) via `ring buffering` and perf event polling, or accessing hashmaps from the Wasm virtual machine. The bpf  can also be accessed with share memory between kernel and Wasm runtime. This setup is not only flexible but also ready to grow with new kernel features without the need to tweak the Wasm environment."

## Slide 21: The wasm-bpf Development Process (2 minutes)

Now let's move on to how we create eBPF applications with wasm.

To develop an eBPF program, we first need to compile the corresponding source code into bpf bytecode using the clang/LLVM toolchain, which contains the corresponding data structure definitions, maps and progs definitions. progs are program segments, and maps can be used to store data or for bidirectional communication with the user space. After that, we can implement a complete eBPF application with the help of the user state development framework and the loading framework. Wasm-bpf also follows a similar approach.

In the Wasm-bpf project, with the support of code generation techniques and BTF (BPF type format) information in the toolchain, all communications between Wasm and eBPF do not need to `go through serialization` and deserialization mechanisms. At the same time, the eBPF-Wasm development experience for user-state programs is improved by `automatically generating skeleton` (bpf code framework) and type definitions, just like the bpftool and libbpf-bootstrap does.

The lightweight nature of compiled eBPF-Wasm modules, which are around 90Kb, and their ability to be dynamically loaded and executed in under 100ms, illustrates the efficiency of this framework. It's optimized for rapid deployment and execution, aligning perfectly with the fast-paced, dynamic requirements of cloud-native environments.

## Slide 22: Wasm with user space eBPF (2 minutes)

Today, we're also examining a new development in system observability and interaction: the combination of WebAssembly, or Wasm, with user space eBPF runtimes.

Wasm with kernel eBPF unlocks a lot of potential, allowing us to engage deeply with the Linux kernel. However, it does come with a need for specific kernel versions supports eBPF, and privileges to load the eBPF into the kernel.

Enter bpftime, our new approach to eBPF that operates entirely in user space. It means we can deploy existing eBPF tracing programs without special permissions and without depending on the kernel's version, and even not limited to Linux system.

bpftime allows us to use eBPF tools with Uprobes and Syscall tracepoints to monitor and trace applications in userspace. These tools are lightweight and don't require stopping or tweaking the applications they monitor. They're like the user space counterparts to kprobes , but with the added advantage of being up to 10 times faster than kernel uprobe.

Plus, bpftime plays nicely with existing eBPF toolchains. It supports inter-process communication with maps, and you can run familiar tools like bcc and bpftrace right in user space without any change to their code.

So, we're looking at a more flexible and supercharged way to gather insights and manage systems, potentially integrating with technologies like DPDK for network tasks. It's a step forward in making powerful system tools more accessible and efficient.

## Slide 23-26: How eBPF Enhances Wasm Developer Experience (3 minutes)

To wrap up our technical discussion, we will explore how eBPF elevates the Wasm development experience, particularly through advanced security mechanisms for WASI and sophisticated tracing capabilities that simplify debugging.

## Closing (1 minute)

In closing, the fusion of Wasm and eBPF is more than just a technological innovation; it's a new frontier in Kubernetes pod deployment, data analytics, and system security. We're excited for you to explore these possibilities and contribute to their evolution. Thank you for joining us today, and we look forward to your questions and insights.
