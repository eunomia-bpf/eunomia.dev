# eBPF + Wasm: Lightweight Observability on Steroids

In this blog, after a brief introduction to eBPF and Wasm and the differences between them, we will discuss the challenges of deploying eBPF in Kubernetes, and how a webassembly runtime and toolchain like WasmEdge and wasm-bpf can address these challenges. We will also discuss how eBPF applications can help improve the WebAssembly (Wasm) runtime and ecosystem.

## Background: eBPF and Wasm in Cloud-Native Ecosystems

In the cloud-native ecosystem, eBPF (Extended Berkeley Packet Filter) and WebAssembly (Wasm) serve as emerging technologies facilitating advanced operational capabilities.

`eBPF` (Extended Berkeley Packet Filter) enables dynamic and secure programming within the Linux kernel, enhancing networking, observability, tracing, and security capabilities efficiently. This technology allows for the injection of bytecode into the kernel at runtime, which the kernel can then execute in response to various events. As a result, eBPF is essential in modern cloud-native environments for its ability to extend kernel functionality dynamically and safely without the need to alter kernel source code or load modules, providing a flexible and powerful means of system optimization and monitoring.

`WebAssembly` (Wasm), on the other hand, is a binary instruction format that functions as a virtual machine and a compilation target for high-level languages, enabling developers to run sandboxed executable code in web browsers and server-side applications. Wasm is designed to be portable, secure, and execute at near-native speeds, making it an ideal runtime for web applications and increasingly for server-side containerized applications.

`The differences between eBPF and Wasm`: Wasm operates within a sandboxed environment, emphasizing security. It incorporates run-time checks for arbitrary code execution, which, while bolstering safety, introduces some performance overhead. Additionally, Wasm boasts a robust language support and ecosystem, making it versatile and conducive to various applications. Contrastingly, eBPF is finely tuned towards performance optimization. It employs static verification methods, minimizing run-time checks as the functionalities are predetermined before execution. eBPF is predominantly used with small C programs, emphasizing its focus on performance and efficiency.

In essence, while both technologies are instrumental in enhancing the execution of code, Wasm leans more towards flexibility and security, whereas eBPF focuses on performance and efficiency. This delineation marks the fundamental differences in their design philosophies and application domains.

`WasmEdge` is a lightweight and high-performance Wasm runtime that has been integrated into Docker CLI, allowing for seamless deployment of both containerized and Wasm-based applications within the same infrastructure. This integration underscores the cloud-native commitment to interoperability and the ability to abstract underlying infrastructure complexities away from developers and operators.

## Current deploy models of eBPF

In large-scale projects, like cilium, pixie, tetragon, falco, the prevalent strategy is to `closely integrate the monitoring or management tools within the core application`, often referred to as the "control plane." This approach allows for a seamless interaction with the system's internals, providing an efficient means of observation and manipulation of low-level operations.

However, this tight integration is not without its drawbacks. A significant challenge is the complexity introduced by the need to `manage multi-user environments`. Without a standardized method to handle the interplay between various applications, coordination becomes more intricate, potentially leading to conflicts. Moreover, the application requires `extensive permissions`, particularly to the `bpf(2)` syscall, which governs powerful network and system monitoring capabilities within the kernel.

An alternative to this integration, like bumblebee, inspektor-gadget, bpfd, is the use of Remote Procedure Calls (RPCs) to communicate between the control plane application and a dedicated `BPF daemon`. This BPF daemon acts as an intermediary, managing the BPF lifecycle and permissions. While this decouples the BPF functionality from the application, it introduces its own set of issues.

The daemon delegation model implies an `additional critical component` in the production environment, which increases the risk of failures. Troubleshooting and debugging become more challenging when another layer is involved. When `new kernel features` are introduced, not only does the kernel dependency need to be managed, but the daemon itself also requires updates and deployment, which can slow down the adoption of new capabilities. This model also imposes an `additional support burden` as loaders have to be compatible with both tight integration and daemon delegation scenarios.

Moreover, `maintaining consistency` during updates is another pain point. Ensuring that the application and the daemon are upgraded or downgraded atomically to avoid compatibility issues is a complex task. The problem is further compounded when considering that operating system distributions or cloud providers may each introduce their proprietary daemons, leading to a fragmented ecosystem.

In summary, while tight integration offers efficiency and direct control, it requires careful coordination and broad permissions. On the other hand, daemon delegation provides a layer of abstraction at the cost of additional complexity and potential delays in leveraging new features. Each model carries its own set of "cons," and the choice between them would depend on the specific requirements and constraints of the project in question.

## Summary: challenges for eBPF in Kubernetes

Deploying eBPF in Kubernetes clusters also adds another layer of complexity. To summarize the challenges of deploying eBPF with different models:

1. **Security Risks with Privileged Access**: eBPF applications necessitate elevated access levels in Kubernetes, often requiring privileged pods. The minimum requirement is CAP_BPF capabilities, but the actual permissions may extend further, depending on the eBPF program type. The broad scope of Linux capabilities complicates the restriction of privileges to the essential minimum. This can inadvertently introduce security vulnerabilities, for instance, a compromised eBPF program could extract sensitive data from the host or other containers and could potentially execute in the host namespace, breaching container isolation.

2. **Compatibility Issues with eBPF Hooks and Kernel Versions**: The eBPF infrastructure in Linux kernels can have limitations, such as certain hooks not supporting concurrent multiple programs. This can lead to conflicts where one eBPF program overrides another, resulting in silent failures or unpredictable behavior. While CO-RE (Compile Once - Run Everywhere) enhances portability across various kernel versions, disparities in feature support, such as 'perf event' and 'ring buffer', remain due to kernel version differences, affecting cross-version compatibility.

3. **Complex Lifecycle and Deployment Management**: Orchestrating the lifecycle of eBPF programs in Kubernetes is complex. Deployment typically involves creating a DaemonSet, which increases architectural complexity and security considerations. The process includes writing a custom agent to load eBPF bytecode and managing this agent with a DaemonSet. This necessitates an in-depth understanding of the eBPF program lifecycle to ensure persistence across pod restarts and to manage updates efficiently. Using traditional Linux containers for this purpose can be counterproductive and heavy, negating the lightweight advantage of eBPF.

4. **Challenges with Versioning and Plugability**: Current eBPF deployments in Kubernetes often embed the eBPF program within the user space binary responsible for its loading and operation. This tight coupling hinders the ability to version the eBPF program independently of its user space counterpart. If users need to customize eBPF programs, for instance, to track proprietary protocols or analyze specific traffic, they must recompile both the eBPF program and user space library, release a new version, and redeploy. The absence of a dedicated package management system for eBPF programs further complicates management and version control.

There is not a single solution to all these challenges, but maybe we can leverage the advantages of WebAssembly (Wasm) to address some of these issues. In the next section, we will discuss how Wasm can help improve the deployment of eBPF in Kubernetes.

## How WebAssembly can bring to eBPF deployments in Kubernetes

WebAssembly (Wasm) could address some of these challenges:

- **lightweight**: Wasm is designed to be lightweight, portable, and secure, making it an ideal runtime for cloud-native environments. Wasm workloads can seamlessly run alongside containers, easing integration into the current infrastructure. With its lightweight containers — mere 1/100th the size of typical LXC images — and rapid startup times, Wasm addresses eBPF’s deployment headaches.

- **Fine-Grained Permissions:** Wasm's runtime environment prioritizes security with a deny-by-default mode, a stark contrast to the broad permissions eBPF necessitates. By leveraging WebAssembly System Interface (WASI), eBPF deployments gain the advantage of fine-grained permission controls. This setup not only tightens security by restricting access to crucial kernel resources but also creates a configurable environment where permissions are granted explicitly and judiciously.
  
- **Portability and Isolation:** Wasm is designed to run in a portable, isolated environment, which could simplify deployment and reduce the risk of programs interfering with one another. with Wasm for eBPF, we can build a abstraction layer to improve the portability of eBPF programs, and also provide a isolated environment for eBPF programs to run in. With userspace eBPF runtime like bpftime, we can also monitor the userspace with same toolchains, but without kernel version support and without root priviledge.

- **Lifecycle Management:**  Wasm's design inherently facilitates better lifecycle management tools and practices. Kubernetes can leverage existing container orchestration tools to manage Wasm applications, streamlining the process.

- **Versioning and Update Management:** With Wasm plugins, eBPF programs can be versioned and updated independently of their user space counterparts. By treating eBPF packages as OCI images within Wasm modules, versioning and updating become more manageable. This enables the sandboxing of eBPF programs separate from the user space, allowing for a modular and flexible approach in tooling, particularly in observability.

In summary, while eBPF offers powerful capabilities for Kubernetes, it comes with significant challenges related to security, manageability, and operational complexity. WebAssembly could provide solutions to these challenges by offering a more secure, isolated, and manageable environment.

## Wasm-bpf: A Paradigm Shift in eBPF Deployments

Wasm-bpf is a WebAssembly eBPF library, toolchain and runtime powered by CO-RE(Compile Once – Run Everywhere) libbpf. It allows the construction of eBPF programs into Wasm with little to no changes to the code, and run them cross platforms with Wasm sandbox. Wasm-bpf can be used as a plugin for WasmEdge, a high-performance Wasm runtime optimized for cloud-native environments, to integrate with Kubernetes. The project allows you:

- `Create Wasm-based eBPF control plane applications`: Wasm-bpf empowers developers to create Wasm-based eBPF control plane applications. These applications can tap into the control and communication mechanisms eBPF provides, but with the added advantages of Wasm's lightweight and secure environment. The result is a robust control plane capable of intricate networking and security operations, all managed with Kubernetes' native tooling.
- `Enable Streamlined Management of eBPF programs in k8s pods with lightweight Wasm container`: With Wasm-bpf, managing eBPF programs becomes an integrated part of Kubernetes' orchestration:
  - **Lightweight Containers**: Utilizing Wasm containers that are a fraction of the size of traditional LXC images, Wasm-bpf ensures that eBPF programs can be deployed rapidly and with less overhead.
  - **Kubernetes Pods**: eBPF programs are deployed as Wasm modules within k8s pods, aligning with existing container orchestration practices and enhancing manageability.
  - **WasmEdge Integration**: As a plugin for WasmEdge, Wasm-bpf benefits from a high-performance runtime optimized for Kubernetes, ensuring seamless cloud-native integration.
- `Enables Wasm plugin in eBPF core applications`: allow dynamic loading and unloading of eBPF programs, promoting a modular and flexible approach towards system observability and interactions. The user can write their own eBPF programs and load them into the observability agents, without the need to recompile and redeploy it. This allows for a more agile and efficient development process, where eBPF programs can be updated independently of the core application for complex data processing and private protocol analysis.

## Enhancing eBPF Deployment: Efficiency and Ease

Wasm-bpf addresses the deployment challenges by offering a runtime that's optimized for eBPF within a Wasm lightweight container:

- **Size and Performance**: The containers are just 1% the size of standard LXC images, coupled with a fast startup time for eBPF programs, ensuring quick deployments. Wasm also has a near native runtime performance.
- **Cross-Platform Portability**: With CO-RE, these eBPF programs are not just portable across different platforms but also across kernel versions, obviating the need for kernel-specific adaptations. With user space eBPF runtime, no kernel eBPF support is needed.
- **Version Control**: It allows for independent versioning and updating of eBPF programs, enabling them to be treated as OCI images within Wasm modules, thereby simplifying versioning and updates.

## Elevating Security in eBPF Deployments

Wasm-bpf not only makes deployment easier but also significantly safer:

- **Configurable WASI Behavior**:  It provides a configurable environment with limited eBPF WASI behavior, enhancing security and control. This allows for fine-grained permissions, restricting access to kernel resources and providing a more secure environment. For instance, eBPF programs can be restricted to specific types of useage, such as network monitoring, without the need for broad permissions.
- **Access Control**: it can also apply RBAC to control the access of eBPF programs easily.
- **Sandboxed Environment**: By sandboxing the user space, Wasm-bpf enables the safe execution of eBPF programs, avoiding the risks associated with privileged access levels in traditional deployments. This

In essence, Wasm-bpf is crafted to mitigate the inherent challenges faced when deploying eBPF in Kubernetes environments. It leverages the strengths of WebAssembly to make eBPF deployments not only easier and more efficient but also significantly more secure. By encapsulating eBPF programs in lightweight, portable, and secure Wasm modules, Wasm-bpf streamlines the lifecycle and versioning management, offering a sophisticated solution that aligns with the dynamic and scalable nature of cloud-native ecosystems. As Kubernetes continues to evolve, Wasm-bpf stands ready to play a critical role in simplifying and securing eBPF deployments across the cloud-native landscape.

## trade offs

While Wasm-bpf presents a promising solution for deploying eBPF programs within Kubernetes, it's essential to consider the trade-offs and limitations that come with this new approach:

1. **Library Portability**:

    Existing libraries like libbpf and libbpf-rs need to be ported to work within the Wasm environment. This requires additional development effort and could introduce compatibility issues or feature limitations.

1. **Learning Curve**:

    Developers may need to learn new language SDKs tailored for Wasm. This investment in time and resources can be significant, especially for teams already accustomed to existing eBPF toolchains.

1. **Feature Parity**:

    eBPF features available within the Wasm environment might be limited compared to those in the native Linux kernel eBPF. Some advanced eBPF functionalities may not be fully supported or may require significant workarounds to be operational in Wasm.

1. **Kernel Version Support**:

    Even though Wasm-bpf leverages CO-RE to enable cross-platform compatibility, the underlying eBPF programs may still require specific kernel version support. This could limit the deployment of certain eBPF programs to environments with the requisite kernel versions.

1. **Performance Overheads**:

    Running eBPF programs in a Wasm sandbox may introduce performance overheads due to additional abstraction layers. This might be acceptable for some use cases but could be a bottleneck for performance-critical applications.

1. **Complexity in Debugging**:

    Debugging issues across the Wasm and eBPF boundary might become more complex. The encapsulation provided by Wasm's sandbox can also obscure problems that would be more apparent in a native environment.

1. **Ecosystem Maturity**:

    The Wasm ecosystem, particularly in the context of eBPF, is relatively new compared to the mature tooling available for native eBPF. This can lead to challenges in finding support, documentation, and community-tested practices.

The introduction of Wasm-bpf is undoubtedly an exciting development, yet it's important to weigh these trade-offs when considering its adoption. For organizations with existing eBPF workloads or those looking to exploit the full range of eBPF capabilities, a careful evaluation of the potential impacts on performance, compatibility, and developer productivity is necessary.

## Challenges of eBPF for Wasm: Bridging Architecture and Kernel Dependencies

The integration of eBPF with WebAssembly (Wasm) within Kubernetes is a promising approach but comes with its own set of challenges:

1. **Porting Libraries and Preparing Toolchains**:

   For different programming languages, there are corresponding libraries to interface with eBPF. These need to be ported to work with Wasm:

   - **C/C++**: `libbpf` is the standard library for working with eBPF in C/C++. Adapting this to work within the Wasm environment requires ensuring compatibility with Wasm's execution model.
   - **Rust**: `libbpf-rs` provides Rust bindings for `libbpf`. These bindings must be made compatible with Wasm to maintain performance and functionality.
   - **Go**: The Go language has its own eBPF library, `cilium/go`. This too must be adapted for a Wasm runtime, which can be non-trivial given Go's runtime and garbage collection features.

2. **Differences in Data Layout Between Wasm and eBPF**:

   Wasm currently operates in a 32-bit environment, whereas eBPF is designed for a 64-bit architecture. This difference in data layout can lead to complications, particularly when handling data structures that are designed with 64-bit systems in mind.

   - To address this, toolchains are utilized to generate bindings that can translate between the two architectures. This minimizes the need for serialization and deserialization, which can degrade performance.

3. **Kernel Version Support for eBPF**:

   eBPF's functionality can depend heavily on the Linux kernel version, with newer features often requiring the latest kernel releases.

   - **CO-RE**: This feature allows eBPF programs to be more portable across different kernel versions by providing a stable ABI that doesn't require recompilation. This can help mitigate kernel compatibility issues.
   - **A compatible layer for kernel features**: Some eBPF features may not be supported in all kernel versions. For instance, the `ringbuf` feature is only available in kernel versions 5.4 and above, while `perf_event` is available in kernel versions 4.9. This can be addressed by providing a compatible layer that emulates the missing functionality.

   - **Userspace eBPF Runtime**: For systems where updating the kernel isn't feasible, a userspace eBPF runtime can be employed. This allows eBPF programs to run without direct kernel support, which is crucial for environments where kernel modifications are restricted.

Once these hurdles are overcome, developers can leverage the power of eBPF in a more flexible and secure manner, enabled by the capabilities of Wasm.

## How can eBPF enhance Wasm: WASI and Debugging

WebAssembly (Wasm) has been a significant leap forward in the realm of portable binary instruction formats. Accoding to a survey in 2023, the top two features that WebAssembly needs to enhance are a better WASI (WebAssembly System Interface) and a better debugging toolchain.That's where eBPF comes into play, offering solutions to elevate Wasm’s utility within the ecosystem.

### Enhancing WASI with eBPF

The WebAssembly System Interface (WASI) is pivotal in managing how Wasm modules interact with the host system. Currently, WASI ensures that an accessed path is within an allowed list before granting access. This implementation, while functional, is not without its challenges. It can be error-prone, often relying heavily on code reviews to prevent security lapses. Furthermore, its configurability is limited, offering only directory-level granularity instead of more precise file-level control.

eBPF can significantly enhance WASI by introducing programmable access control within the Linux kernel. With eBPF's Linux Security Module (LSM) and seccomp capabilities, developers can write custom policies that offer granular control over what Wasm modules can access, down to specific files and operations.

Consider an example where there's a need to hook into directory removal operations to check permissions for a specific directory. Here, eBPF can be employed to intercept these operations at the kernel level and execute custom verification logic, providing a more robust and flexible access-control mechanism for WASI.

### Advancing Debugging Tools with eBPF

When it comes to debugging, Wasm's current tracing methodologies are somewhat rudimentary, lacking the depth required for intricate analysis. eBPF's uprobes (user-space probes) can bridge this gap by enabling detailed tracing of any user-space function invoked by a Wasm module, all without the need for additional code instrumentation.

For instance, memory allocation within a Wasm runtime like WasmEdge could be traced using uprobes. This would allow developers to gain insights into the memory behavior of their applications, identifying bottlenecks or leaks that could affect performance and stability.

Additionally, user-space eBPF runtimes such as `bpftime` facilitate rapid and powerful uprobes that don't require kernel modifications or root privileges, making the debugging process less invasive and more accessible.

In essence, eBPF's integration with Wasm paves the way for more sophisticated and secure system interfaces and debugging capabilities. As these technologies converge, we witness the emergence of a more powerful and developer-friendly platform, capable of driving innovation across diverse computing environments. The synergy of eBPF and Wasm is set to redefine what's possible, opening new horizons for application performance, security, and manageability.

## reference

- Wasm-bpf: <https://github.com/eunomia-bpf/wasm-bpf>
- WasmEdge eBPF plugin: <https://github.com/WasmEdge/WasmEdge/tree/master/plugins/wasm_bpf>
- bpfd: <https://bpfd.dev/#why-ebpf-in-kubernetes>
- Cross Container Attacks: The Bewildered eBPF on Clouds: <https://www.usenix.org/conference/usenixsecurity23/presentation/he>
- eBPF - The Silent Platform Revolution from Cloud Native: <https://conferences.sigcomm.org/sigcomm/2023/files/workshop-ebpf/1-CloudNative.pdf>
- POSTER: Leveraging eBPF to enhance sandboxing of WebAssembly runtimes: <https://dl.acm.org/doi/fullHtml/10.1145/3579856.3592831> - We have done similar works early this year
