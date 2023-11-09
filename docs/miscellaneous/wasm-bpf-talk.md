# talk draft in kubecon

## IntroductionOpening (1 minute)

Welcome, everyone.

Thank you for joining me to discuss the innovative intersection of eBPF and WebAssembly—technologies revolutionizing observability within our systems.

Combining eBPF with Wasm provides a robust solution for non-intrusive deployment and advanced security checks within Kubernetes pods. Today, we'll explore how these technologies can be leveraged for efficient and secure software development. Let's dive into a detailed discussion on their benefits, challenges, and the future they hold.

My name is Yusheng Zheng, currently maintaining a small community called eunomia-bpf, building open source projects to make eBPF easier to use, and exploring new technologies and runtimes related to eBPF.

## Slide 2: Agenda (2 minutes)

In the next minute, we'll outline our exploration into eBPF and WebAssembly, or Wasm—a journey into enhancing the deployment and security of eBPF programs. We begin by introducing eBPF, the kernel-level technology that powers efficient networking and security, followed by Wasm, which brings a multi-language, secure execution environment to user space.

We'll uncover how Wasm streamlines the eBPF experience, enabling non-intrusive deployments into Kubernetes pods, offering separation from application workloads for greater flexibility. We'll emphasize the simplicity of implementing security with declarative checks at deployment and discuss how eBPF data analytics can drive insights for performance tuning. By supporting user-space eBPF, it opens the door to more secure and versatile observability tools.

Finally, we'll flip the script and look at eBPF through the lens of Wasm, enhancing security sandboxing, host call observability, and providing a robust debugging toolkit. This bidirectional enrichment is pivotal for cloud-native development, and we're excited to delve into these transformative technologies with you.

## Slide 3: Introduction to eBPF (1 minutes)

First, let's talk about eBPF, or Extended Berkeley Packet Filter. This is not just a technology but a paradigm shift, allowing developers to dynamically and safely program the Linux kernel. It's at the heart of performance-sensitive tasks like networking and security, and it's changing the game for kernel-level instrumentation. 

## Slide 4: Introduction to WebAssembly (Wasm) (1 minutes)

Turning our attention to WebAssembly, or Wasm, we enter the realm of a revolutionary binary format designed for user space security. Wasm is not only language-agnostic, supporting a wide array of programming languages, but it's also incredibly lightweight and swift, making it ideal for performance-critical applications.

What sets Wasm apart is its capability-based security model, which allows fine-grained access to host resources, ensuring that applications remain secure while performing at their best. Moreover, Wasm's cross-platform nature facilitates an unprecedented level of portability, enabling code to run consistently across different environments.

## Slide 5: 

## Slide 6: How Wasm Improves eBPF Developer Experience (3 minutes)

"Stepping into the developer's shoes, we see how Wasm refines the eBPF development workflow. It ensures non-intrusive, Kubernetes-native deployment processes and facilitates security checks at deployment time, making our systems more resilient and our developer experience smoother."

## Slide 7-10: eBPF Deployment Models (5 minutes)

"We'll dissect the eBPF deployment models, contrasting the integrated control plane against the decoupled sidecar approach. The former, despite its direct control benefits, raises concerns about security and multi-user conflicts. The latter, while modular, introduces complexity in maintaining consistency and kernel feature integration."

## Slide 11-12: Wasm + eBPF: The Synergy (3 minutes)

Here, we'll illustrate a typical Kubernetes pod setup, integrating eBPF into containers running in an LXC environment, alongside other workload-specific containers. This hybrid model capitalizes on the strengths of Wasm and eBPF, creating a robust, modular observability framework.

## Slide 13-14: WasmEdge eBPF plugin (2 minutes)

We'll showcase the WasmEdge eBPF plugin, wasm-bpf. Its compactness, ease of management, and security enhancements over traditional containerized eBPF deployments signal a leap forward in deploying eBPF programs.

## Slide 15-17: Developer Experience and Trade-offs (3 minutes)

However, there are trade-offs. The migration of libraries and toolchains to this new model is not trivial, with considerations around limited eBPF features in Wasm environments. But the familiar development experience, akin to that provided by libbpf-bootstrap, is a testament to our progress.

## Slide 18-19: Examples and Challenges (3 minutes)

We will present practical examples showcasing eBPF's versatility in observability, networking, and security. Simultaneously, we'll confront the challenges: porting libraries, reconciling data layouts, and ensuring kernel compatibility, all of which are critical to mainstream adoption.

## Slide 20-22: How it Works (3 minutes)

We will then delve into the mechanics—how we use toolchains to facilitate seamless integration and development of eBPF programs. This process not only supports kernel-level eBPF but also enables a fully user space runtime, expanding eBPF's applicability.

Slide 23-26: How eBPF Enhances Wasm Developer Experience (3 minutes)

To wrap up our technical discussion, we will explore how eBPF elevates the Wasm development experience, particularly through advanced security mechanisms for WASI and sophisticated tracing capabilities that simplify debugging.

Closing (1 minute)

In closing, the fusion of Wasm and eBPF is more than just a technological innovation; it's a new frontier in Kubernetes pod deployment, data analytics, and system security. We're excited for you to explore these possibilities and contribute to their evolution. Thank you for joining us today, and we look forward to your questions and insights.
