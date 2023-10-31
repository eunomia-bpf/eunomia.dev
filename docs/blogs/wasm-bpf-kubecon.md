# eBPF + Wasm: Lightweight Observability on Steroids

In this blog, we will discuss the motivation behind the Wasm-bpf project: the challenges of deploying eBPF in Kubernetes, and how Wasm-bpf can address these challenges. We will also discuss how eBPF applications can help improve the WebAssembly (Wasm) runtime and ecosystem.

## Background: eBPF and Wasm in Cloud-Native Ecosystems

In the cloud-native ecosystem, eBPF (Extended Berkeley Packet Filter) and WebAssembly (Wasm) serve as pivotal technologies facilitating advanced operational capabilities. eBPF, a highly efficient virtual machine-like construct in the Linux kernel, allows developers to run sandboxed programs in response to events such as network packets or system calls, thereby providing a mechanism for extending kernel functionality dynamically without changing kernel source code or loading kernel modules. Primarily utilized for performance monitoring, network traffic filtering, and security enforcement, eBPF is integral to modern observability and networking solutions in cloud environments.

WebAssembly (Wasm), on the other hand, is a binary instruction format that functions as a virtual machine and a compilation target for high-level languages, enabling developers to run sandboxed executable code in web browsers and server-side applications. Wasm is designed to be portable, secure, and execute at near-native speeds, making it an ideal runtime for web applications and increasingly for server-side containerized applications. WasmEdge exemplifies this utility as a lightweight and high-performance Wasm runtime that has been integrated into Docker CLI, allowing for seamless deployment of both containerized and Wasm-based applications within the same infrastructure. This integration underscores the cloud-native commitment to interoperability and the ability to abstract underlying infrastructure complexities away from developers and operators.

## Challenges for eBPF in Kubernetes

Deploying eBPF (Extended Berkeley Packet Filter) within Kubernetes clusters presents several technical hurdles. These challenges are primarily in the realms of security, compatibility, lifecycle management, and versioning:

1. **Security Risks with Privileged Access**: eBPF applications necessitate elevated access levels in Kubernetes, often requiring privileged pods. The minimum requirement is CAP_BPF capabilities, but the actual permissions may extend further, depending on the eBPF program type. The broad scope of Linux capabilities complicates the restriction of privileges to the essential minimum. This can inadvertently introduce security vulnerabilities, for instance, a compromised eBPF program could extract sensitive data from the host or other containers and could potentially execute in the host namespace, breaching container isolation.

2. **Compatibility Issues with eBPF Hooks and Kernel Versions**: The eBPF infrastructure in Linux kernels can have limitations, such as certain hooks not supporting concurrent multiple programs. This can lead to conflicts where one eBPF program overrides another, resulting in silent failures or unpredictable behavior. While CO-RE (Compile Once - Run Everywhere) enhances portability across various kernel versions, disparities in feature support, such as 'perf event' and 'ring buffer', remain due to kernel version differences, affecting cross-version compatibility.

3. **Complex Lifecycle and Deployment Management**: Orchestrating the lifecycle of eBPF programs in Kubernetes is complex. Deployment typically involves creating a DaemonSet, which increases architectural complexity and security considerations. The process includes writing a custom agent to load eBPF bytecode and managing this agent with a DaemonSet. This necessitates an in-depth understanding of the eBPF program lifecycle to ensure persistence across pod restarts and to manage updates efficiently. Using traditional Linux containers for this purpose can be counterproductive and heavy, negating the lightweight advantage of eBPF.

4. **Challenges with Versioning and Plugability**: Current eBPF deployments in Kubernetes often embed the eBPF program within the user space binary responsible for its loading and operation. This tight coupling hinders the ability to version the eBPF program independently of its user space counterpart. If users need to customize eBPF programs, for instance, to track proprietary protocols or analyze specific traffic, they must recompile both the eBPF program and user space library, release a new version, and redeploy. The absence of a dedicated package management system for eBPF programs further complicates management and version control.

## How WebAssembly can bring to eBPF deployments in Kubernetes

WebAssembly (Wasm) could address some of these challenges:

- **lightweight**: Wasm is designed to be lightweight, portable, and secure, making it an ideal runtime for cloud-native environments. Wasm workloads can seamlessly run alongside containers, easing integration into the current infrastructure. With its lightweight containers — mere 1/100th the size of typical LXC images — and rapid startup times, Wasm addresses eBPF’s deployment headaches.

- **Fine-Grained Permissions:** Wasm's runtime environment prioritizes security with a deny-by-default mode, a stark contrast to the broad permissions eBPF necessitates. By leveraging WebAssembly System Interface (WASI), eBPF deployments gain the advantage of fine-grained permission controls. This setup not only tightens security by restricting access to crucial kernel resources but also creates a configurable environment where permissions are granted explicitly and judiciously.
  
- **Portability and Isolation:** Wasm is designed to run in a portable, isolated environment, which could simplify deployment and reduce the risk of programs interfering with one another. with Wasm for eBPF, we can build a abstraction layer to improve the portability of eBPF programs, and also provide a isolated environment for eBPF programs to run in. With userspace eBPF runtime like bpftime, we can also monitor the userspace with same toolchains, but without kernel version support and without root priviledge.

- **Lifecycle Management:**  Wasm's design inherently facilitates better lifecycle management tools and practices. Kubernetes can leverage existing container orchestration tools to manage Wasm applications, streamlining the process.

- **Versioning and Update Management:** With Wasm plugins, eBPF programs can be versioned and updated independently of their user space counterparts. By treating eBPF packages as OCI images within Wasm modules, versioning and updating become more manageable. This enables the sandboxing of eBPF programs separate from the user space, allowing for a modular and flexible approach in tooling, particularly in observability.

In summary, while eBPF offers powerful capabilities for Kubernetes, it comes with significant challenges related to security, manageability, and operational complexity. WebAssembly could provide solutions to these challenges by offering a more secure, isolated, and manageable environment.

## Wasm-bpf

Wasm-bpf is a toolkit designed to bridge the capabilities of WebAssembly (Wasm) with eBPF, making it easier to build, manage, and deploy eBPF applications within Kubernetes. Here are the key points about Wasm-bpf:

1. **CO-RE Powered**: Utilizes the Compile Once – Run Everywhere (CO-RE) libbpf, enabling eBPF programs to run on different kenrel verisons without modification.
2. **Streamlined Development**: Allows the construction of eBPF programs into Wasm with little to no changes to the code, simplifying the development process.
3. **k8s and container integration**: Enables the deployment of eBPF programs as Wasm modules within Kubernetes, leveraging existing container orchestration tools and practices.
4. **Enhanced Version Control**: Wasm-bpf enables independent versioning and updating of eBPF programs, which can be packaged as OCI images within Wasm modules.
5. **Plugibility**: Wasm-bpf allows eBPF programs to be loaded and unloaded dynamically, enabling a more flexible and modular approach to observability, with a sandboxed environment for eBPF programs to run in safely.
6. **Security**: Deploy eBPF programs as Wasm modules, benefiting from the inherent security features of Wasm. This means that eBPF programs can be contained within a well-defined boundary, reducing the risk of unauthorized access or escape to the host system.

The Wasm-bpf project is a step towards a more secure and manageable use of eBPF within Kubernetes, promising to simplify the deployment and operation of eBPF-based applications. For further details, the project's GitHub repository offers comprehensive information and updates.

Wasm-bpf can be used as a plugin for WasmEdge, a high-performance Wasm runtime optimized for cloud-native environments, to integrate with Kubernetes.

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

- bpfd: <https://bpfd.dev/#why-ebpf-in-kubernetes>
- Cross Container Attacks: The Bewildered eBPF on Clouds: <https://www.usenix.org/conference/usenixsecurity23/presentation/he>
