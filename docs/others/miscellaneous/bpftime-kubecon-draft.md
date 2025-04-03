# eBPF and Wasm: Unifying Userspace Extensions With Bpftime - Yusheng Zheng, eunomia-bpf

Thursday April 3, 2025 14:15 - 14:45 BST
Level 1 | Hall Entrance S10 | Room D

## Abstract

In cloud-native systems, extending and customizing applications is key to improving development, deployment, and observability. eBPF is powerful for kernel-level enhancements, and WebAssembly brings extension to userspace. Yet, both face challenges when userspace extensions need to interact deeply with host applications. eBPF's kernel-focused design struggles in diverse userspace environments, and Wasm's sandboxing introduces overhead and complexity due to extra checks and data copying. Enter bpftime, a framework that extends eBPF's capabilities into userspace. Using dynamic binary instrumentation, bytecode verification, and hardware isolation, bpftime allows secure, high-performance extensions without the overhead of Wasm's sandboxing. This talk explores how bpftime works with the eBPF Interface to simplify userspace extensions, compares the evolution of eBPF and Wasm, and shows how bpftime can power observability, networking, and other cloud-native extensions.

## Slides and texts

### Slide 1: Title

eBPF and Wasm: Unifying Userspace Extensions With Bpftime

**[VISUAL: Title slide with bpftime logo, eBPF and Wasm logos on opposite sides with an arrow connecting them through bpftime]**

> Good afternoon, everyone! My name is Yusheng Zheng, I'm a PhD student at UC Santa Cruz and I'm maintaining several eBPF-related open source projects in the eunomia-bpf community.
> Today, I'm going to talk about something that has been around in the software industry for a really long time—software extensions.

### Slide 2: Agenda

**[VISUAL: Agenda with icons for each section]**

- Introduction to Application Extensions
- Challenges: Safety vs. Interconnectedness
- Limitations of Existing Solutions
- EIM: A Fine-Grained Interface Model
- bpftime: Efficient Runtime Design
- Use Cases & Evaluation
- Conclusion

> Specifically, I want to talk about why we need extensions, what makes them challenging to handle correctly, and how our current approaches to managing extensions might not be good enough. Then I'll introduce a new approach to managing extensions called the Extension Interface Model (EIM) and our userspace eBPF runtime called bpftime that implements these principles. The research paper of it, has been accepted by OSDI 2025.

### Slide 3: Software Extensions

**[VISUAL: Timeline list of extension systems with names and icons for each example, handle writting style]**

- Extensions have deep roots in software development
- Common examples across the industry:
  - Web servers: nginx/Apache modules
  - Databases: Redis and PostgreSQL extensions
  - Editors: Vim, Emacs, VSCode plugins
  - Cloud-native: Kubernetes extensions, wasm modules in cloud-native systems
  - Kernel: eBPF programs, kernel modules

> So, first, software extensions aren't new—they have a very long history. Web servers like nginx and Apache use modules for authentication, caching, and security. Databases like Redis and PostgreSQL have extensions for new query types and data formats. Editors like Vim, VSCode and Emacs rely heavily on plugins. In cloud-native systems, Kubernetes uses extensions for observability, networking with CNI plugins, and security features. At the kernel level, we have eBPF programs and kernel modules to extend functionality.

### Slide 4: Why Use Extensions?

**[VISUAL: Balance scale showing flexibility and isolation on opposite sides]**

- **Flexibility**: Adapt software without core codebase changes it also includes Customization, Meet specific requirements without waiting for core developers
- **Isolation**: Critical for security and stability
  - External code may contain bugs or malicious elements
  - Need to protect the core application

> But there's a question a lot of people may ask: why don't we just integrate everything into the main codebase directly? Why bother with extensions at all?
>
> The short answer is—flexibility and isolation.
>
> We want flexibility and customization because it makes our software adaptable. Users and administrators want to tweak things to meet their specific requirements without waiting for the core developers to implement changes. But flexibility without isolation is risky. Extensions, by definition, are third-party or at least externally-developed code. You might trust your core engineering team, but trusting external code is a different story. Even if it's not malicious, external code can have bugs, causing crashes, performance degradation, or security vulnerabilities.

### Slide 5: Real-World Extension Failures

**[VISUAL: "Incident Report" style graphics showing each failure case with impact metrics]**

- **Bilibili**: Production outage from nginx extension infinite loop
  - Service disruption affecting millions of users
- **Apache HTTP Server**: Buffer overflows in Lua modules
  - Security vulnerabilities and system crashes
- **Redis**: Improperly sandboxed Lua scripts
  - Remote code execution vulnerabilities
- These aren't theoretical risks—they've cost companies money and reputation

https://github.com/eunomia-bpf/cve-extension-study​

We find 1217 CVEs related to extensions in 17279 total CVEs from Postgres, MySQL, Redis, Nginx, Apache httpd, Chrome, Firefox, Kubernetes, Docker, Minecraft

> Here are some real-world examples. For example, a few years back, the popular video streaming site Bilibili suffered a serious production outage because one of their nginx extensions got stuck in an infinite loop. Apache HTTP Server had similar issues where buffer overflow bugs in a Lua-based module caused crashes and security holes. Redis also had cases where improperly sandboxed Lua scripts resulted in remote code execution vulnerabilities. These are things that have actually hurt big companies and cost money.
>
> Of course, the extension-related vulnerabilities are not just limited to the above examples. We recently did a study to analyze CVE reports from some open source projects. We search for all the CVEs in these softwares, and found that there are 1217 CVEs related to extensions in 17279 total CVEs from these projects. What we found was that extension-related vulnerabilities make up a significant portion - about 7% - of all CVEs. Many of them could lead to system crashes or data leakage.
>
> So, isolation and safety become absolutely critical. We don't want a bug in one extension to crash our entire system. We don't want a poorly-written plugin causing our service to slow down. We definitely don't want external code exposing our internal data to attackers.

### Slide 6: Three Core Requirements for Extension Frameworks

**[VISUAL: Triangle diagram with "Interconnectedness," "Safety," and "Efficiency" at the corners, showing the tension between them. also the extension runtime figture]**

- **Interconnectedness**: Extension's ability to interact with host
  - Reading data, modifying state, calling internal functions
  - Example: nginx extension reading request details
- **Safety**: Limiting extensions' ability to harm the application
  - Memory safety, control flow restrictions, resource limits
- **Efficiency**: Performance impact on the system
- **The Fundamental Tension**: More interconnectedness typically means less safety

> So this figure shows how a regular application can be extended using a separate extension runtime. Think of the host application as the original app, which has its own state (like variables) and code (like functions). Instead of directly modifying that code, the user adds new behavior through extensions. These extensions run in a separate component called the extension runtime. Each extension connects back to the host using defined entry points (entry1, entry2). For example, ext1 read or modify a variable in the host, while ext2 can actually call a function in the host app, like foo()
>
> As we can see from these real-world failures and the extension usecases, there are three core requirements for extension runtime frameworks: **interconnectedness**, **safety** and **efficiency**.
>
> First, what do I mean by interconnectedness? Simply put, interconnectedness is how much power we give an extension to interact with the host application. Extensions need to do something meaningful—they need to read data, modify state, or call existing functions inside the application. Different extensions need different level of interconnectednes. For example, a security extension needs to read request details and block suspicious requests, while observability extensions just need to read request details.
>
> As we've discussed, safety is how much we limit an extension's ability to harm the main application. If there's a bug in your extension, this bug shouldn't crash your whole web server or compromise your entire application. Without safety boundaries, a single small mistake in an extension could take down a production system—exactly what happened in those examples I just mentioned.
>
> Efficiency is about performance—how much overhead the extension framework adds to your application. It's easy to understand that different software has different performance requirements.
>
> The key challenge is that interconnectedness and safety are fundamentally at odds. The more interconnectedness you allow, the less inherently safe it becomes. To keep things safe, you have to restrict interconnectedness, which limits extension usefulness. Balancing this tension while maintaining efficiency is what makes extension frameworks so challenging to design.

### Slide 7: Limitation of Extension Framework

| **Approach**             | **Example(s)**                                                                 | **Strengths**                                             | **Limitations**                                                                                                                                                     |
|--------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Native Execution**     | `LD_PRELOAD`, `nginx dynamic modules`, `GDB-style instrumentation`            | ✅ High performance<br>✅ Simple integration                | ❌ No isolation between host and extension<br>❌ No fine-grained safety/interconnectedness control<br>❌ Extension crash = app crash                                 |
| **SFI-based Tools**      | `WebAssembly`, `Lua`, `NaCl`, `RLBox`, `XFI`                                  | ✅ Software fault isolation<br>✅ Some cross-platform use   | ❌ Runtime overhead<br>❌ Coarse or no safety/interconnectedness interface<br>❌ Relies on manual host-side checks (often buggy)                                     |
| **Subprocess Isolation** | `Wedge`, `Shreds`, `lwC`, `Orbit`                                             | ✅ Strong isolation<br>✅ Host can't modify extension state | ❌ Context switch overhead<br>❌ Some (like lwC, Shreds) lack per-extension control<br>❌ Others (like Orbit) require host source code changes to support tradeoffs  |
| **eBPF Uprobes**         | `eBPF-based user-space tracing (e.g., perf, bcc tools)`                        | ✅ Safe execution<br>✅ Existing eBPF ecosystem compatible   | ❌ No fine-grained control over extension capabilities<br>❌ Each extension call triggers a kernel trap → **inefficient for high-frequency hooks**                   |

> Let's look at how existing extension frameworks try to balance these three requirements, and where they fall short.
>
> Native execution approaches like LD_PRELOAD and dynamic modules offer excellent performance and simple integration, but they provide no isolation—a bug in a dynamically-loaded module crashes your entire application. There's no safety boundary and no fine-grained control over what extensions can access, but the interconnectedness is maximized.
>
> Software Fault Isolation (SFI) tools like WebAssembly and Lua provide better isolation through runtime checks. But they introduce performance overhead from these checks and boundary crossings. They also often rely on the host application to implement security boundaries correctly, which as we've seen from Redis and Apache incidents, is error-prone.
>
> Subprocess isolation or RPC based approaches, like a lot of cloud-native applications, Model Context Protocol in LLM applications, and some other reseach projects, offer strong isolation by running extensions in separate processes. But they suffer from context switch overhead, making them too slow for performance-critical applications. Some lack per-extension control, while others require significant changes to the host application.
>
> eBPF-based userspace tracing, like uprobe, is also a way to extend user applications. However, current implementations lack fine-grained control over extension capabilities, and each extension call requires a costly kernel context switch, making them inefficient for high-frequency hooks.
>
> We can see current software frameworks have not handled this tradeoff very well. Either they allow too much interconnectedness, like dynamically loaded modules, or they provide strong safety through heavy isolation, like sandboxed scripting environments or subprocess isolation methods. But these can cripple interconnectedness and performance—extensions often become slow and limited in what they can do.
>
> So, what we've found is that the key to managing this tension—this interconnectedness versus safety tradeoff—is the interface we choose for extensions. If your extension framework's interface can carefully define exactly which resources and functions an extension can use, you can precisely manage this tension. Ideally, you give the extension just enough interconnectedness to do its job—but absolutely no more. This sounds simple, but current systems struggle to achieve this.
>
> Let's take 2 popular approaches for extensions, WebAssembly and eBPF, and see how they approach this problem.
>
> WebAssembly is a binary instruction format used in many cloud-native systems as extension mechanism, it provides software fault isolation through runtime checks. eBPF is a restricted instruction set designed for Linux kernel, and is also used in Windows. Compared with Wasm, eBPF has a history of focusing on performance first, which led to the design of using a verifier to check the safety of the extension at load time, instead of runtime checking or SFI (software fault isolation) like Wasm.

> Let's look at how these two technologies approach the interface design challenge.

### Slide 10: WebAssembly Component Model - Overview

**[VISUAL: Diagram showing Component Model architecture with interfaces, imports/exports, and type system]**

- WebAssembly's approach to standardizing extension interfaces
- Core goals:
  - Define portable, language-agnostic interfaces
  - Enable capability-safe, statically-analyzable extensions
  - Support virtualization across diverse environments
- Key concepts:
  - Components: Composable units built from Wasm modules
  - Interfaces: Explicit contracts between components and hosts
  - WIT (WebAssembly Interface Types): Rich type system for interfaces

> The WebAssembly Component Model represents Wasm's approach to solving the extension interface challenge. It's a specification that defines how Wasm modules can be composed together and how they interact with their host environment.
>
> The Component Model was designed with several key goals: defining portable interfaces that work across languages, ensuring capability-safety through explicit interfaces, and supporting virtualization in diverse environments from browsers to cloud systems.
>
> At its core, the Component Model introduces "components" as composable units built from Wasm modules. These components interact through well-defined interfaces written in WIT (WebAssembly Interface Types), which provides a rich type system including records, variants, enums, and more complex types.
>
> This approach standardizes how modules or extensions declare their requirements and capabilities, making them more portable and easier to reason about.

### Slide 11: Component Model - Runtime Capabilities & Limitations

**[VISUAL: Diagram showing capability-based security model with handles and interface boundaries]**

- Runtime capabilities through resource handles:
  - Unforgeable references to resources
  - First-class values that can be passed between components
  - Explicit permission model for resource access
- Tradeoffs:
  - ✅ Strong capability-based security
  - ✅ Rich type system for interfaces
  - ❌ Still requires runtime checks at interface boundaries
  - ❌ Performance overhead from data marshaling

```wit
// wit/types.wit
interface types {
  resource request { ... }
  resource response { ... }
}

// wit/handler.wit
interface handler {
  use types.{request, response};
  handle: func(r: request) -> response;
}

// wit/proxy.wit
world proxy {
  import wasi:logging/logger;
  import handler;
  export handler;
}
```

> The Component Model implements capability-based security through "resource handles" - unforgeable references that grant access to specific resources. These handles can be passed between components, allowing fine-grained control over which extensions can access which resources.
>
> For example, in these WIT interface definitions, we see how components interact through well-defined interfaces. The 'request' and 'response' resources are handles that can only be accessed through the defined functions. A component must explicitly import the handler interface to process requests, enforcing capability-based access control.
>
> This capability-based approach aligns with the principle of least privilege we discussed earlier - extensions only get access to exactly what they need.
>
> However, the Component Model still faces performance challenges. While it provides better interface definition than basic Wasm, it requires runtime checks at interface boundaries and data copying when crossing those boundaries. This creates overhead, especially for extensions that frequently interact with the host.


### Slide 12: eBPF Interface in Kernel

**[VISUAL: Diagram showing eBPF verifier analyzing code paths at load time]**

- Load-time verification through static analysis
- Originally based on:
  - Helper functions
  - Program types
  - Attach types
- To extend eBPF, kernel introduced:
  - `struct_ops`
  - `kfuncs` (kernel functions)
- Uses **BPF Type Format (BTF)** for type system

> In contrast to Wasm's runtime checks, eBPF uses a verifier to analyze programs before they run. This verifier checks all possible execution paths to ensure memory safety and prevent infinite loops. This approach shifts safety checks from runtime to load-time, eliminating runtime overhead once the program is loaded.
>
> The eBPF interface in the kernel was originally based on three main concepts: helper functions that extensions can call, program types that define what an extension can do, and attach types that determine where an extension can hook into the kernel.
>
> As eBPF evolved, to avoid the growing number of helper functions and entry points, which would need to modify the verifier, the kernel introduced more interface mechanisms like struct_ops and kfuncs, which we'll look at next. These provide more flexible ways for extensions to interact with the kernel while maintaining safety.
>
> The kernel uses BPF Type Format (BTF) as its type system, which provides rich type information that the verifier can use to ensure type safety. This is similar in concept to Wasm's interface types, but applied at load time rather than runtime. Besides using for verify, it can also support CO-RE (Compile Once, Run Everywhere) features, like adapting to different kernel versions.

### Slide 13: eBPF Interface - `struct_ops`

**[VISUAL: Code example showing struct_ops definition and registration]**

- Works like `export`—used to **register new eBPF program types** that the kernel can call (from kernel modules)
- Examples:
  ```c
  struct tcp_congestion_ops tcp_reno = {
    .flags = TCP_CONG_NON_RESTRICTED,
    .name = "reno",
    .owner = THIS_MODULE,
    .ssthresh = tcp_reno_ssthresh,
    .cong_avoid = tcp_reno_cong_avoid,
    .undo_cwnd = tcp_reno_undo_cwnd,
  };
  ```

  ```c
  SEC(".struct_ops")
  struct tcp_congestion_ops dctcp_nouse = {
    .init = (void *)dctcp_init,
    .set_state = (void *)dctcp_state,
    .flags = TCP_CONG_NEEDS_ECN,
    .name = "bpf_dctcp_nouse",
  };
  ```

> One of the key interface mechanisms in eBPF is struct_ops, which works somewhat like an "export" in other systems. It allows eBPF programs to register new program types that the kernel can call.
>
> Think of struct_ops as a way for eBPF programs to implement interfaces defined by the kernel. For example, this is the implemention of TCP congestion control through the tcp_congestion_ops structure. Below it, we see how an eBPF program can attach to the same interface. The kernel can call into these eBPF implementations just like it would call into native kernel functions.

### Slide 14: eBPF Interface - `kfunc`

**[VISUAL: Code example showing kfunc definition and registration with flags]**

- Works like `import`—used to **expose kernel functions** to eBPF programs
- Verification Support:
  - **Annotations**: `__sz`, `__k`, `__uninit`, `__opt`, `__str`
  - **Flags**: `KF_ACQUIRE`, `KF_RET_NULL`, `KF_RELEASE`, `KF_TRUSTED_ARGS`, `KF_SLEEPABLE`, `KF_DESTRUCTIVE`, `KF_RCU`, `KF_DEPRECATED`
- Example:
  ```c
  BTF_KFUNCS_START(bpf_task_set)
  BTF_ID_FLAGS(func, bpf_get_task_pid, KF_ACQUIRE | KF_RET_NULL)
  BTF_ID_FLAGS(func, bpf_put_pid, KF_RELEASE)
  BTF_KFUNCS_END(bpf_task_set)
  ```

> The counterpart to struct_ops is kfuncs, which works like an "import" mechanism. Kfuncs allow the kernel to expose specific functions to eBPF programs, giving extensions controlled access to kernel functionality.
>
> What's interesting about kfuncs is how they handle safety. The kernel uses annotations and flags to tell the verifier what safety properties to check. For example, the KF_ACQUIRE flag indicates that a function returns a resource that must be released later, while KF_RELEASE indicates a function that releases such a resource.
>
> This is similar to how Wasm's Component Model handles resource lifetimes with its resource types, but again, the key difference is that eBPF verifies these properties at load time.
>
> In this example, we see how the kernel registers two functions: bpf_get_task_pid which acquires a resource and might return null, and bpf_put_pid which releases that resource. The verifier will ensure that any eBPF program that calls these functions follows the correct pattern of acquiring and releasing resources.

### Slide 15: eBPF Interface in Kernel - Tradeoffs

**[VISUAL: Pros and cons list with checkmarks and X marks]**

- ✅ **Pros:**
  - Strong **verify-based security** — bugs caught before load time
  - Better performance compared to runtime-check-based models

- ❌ **Cons:**
  - Tightly coupled with kernel eBPF implementation
  - Limited expressiveness (restricted C)
  - Hard to define fine-grained safety and abstraction

> To summarize the eBPF interface approach, let's look at its tradeoffs.
>
> The main advantages are strong security through verification before execution, and better performance compared to runtime-check-based models like Wasm. By catching bugs at load time, eBPF eliminates the need for runtime checks, which can significantly improve performance for frequently executed code.
>
> However, there are also limitations. The eBPF interface is tightly coupled with the kernel implementation, making it less portable and useful for userspace extensions like Wasm. Also, eBPF programs are written in a restricted subset of C, which limits expressiveness compared to languages supported by Wasm. And while struct_ops and kfuncs provide more flexibility, it can still be challenging to define fine-grained safety properties and abstractions.
>
> What if we could combine eBPF's fine-grained load-time verification with efficient userspace execution, while addressing these limitations? That's where our approach with the Extension Interface Model and bpftime comes in.

### Slide 16: Extension Interface Model (EIM)

**[VISUAL: Diagram showing EIM as a bridge between extensions and host applications with capability controls]**

- Our approach: Extension Interface Model (EIM)
- Inspired by **eBPF's verifier** and **Wasm's component model**
- Core insight: Treat all extension-host interactions as **explicit capabilities**
- Follows principle of **least privilege** for extensions
- Designed for both **safety and performance** in userspace extensions
- **Limitation**: expressiveness

> Now that we've seen how both WebAssembly and eBPF approach extension interfaces, let me introduce our approach: the Extension Interface Model, or EIM.
>
> EIM takes inspiration from both worlds - eBPF's load-time verification approach and Wasm's component model for interface definition. The core insight of EIM is treating every interaction between an extension and its host as an explicit capability that must be declared and verified.
>
> This follows the principle of least privilege - each extension should only have access to exactly what it needs to function, including the host functions and memories.
>
> Unlike previous approaches that force you to choose between safety and performance, EIM is designed to provide both by shifting safety checks to load time while maintaining fine-grained control over what extensions can do.
>
> We do acknowledge one limitation: like eBPF, there are some constraints on expressiveness compared to general-purpose languages.

### Slide 17: EIM – Specification Example

**[VISUAL: Split screen with explanation on left and code example on right]**

- Use **kernel eBPF verifier** to verify **userspace extensions**
- Use **static analysis** and **BTF** to generate Specification from source code
- Defines three types of capabilities:
  - **State Capabilities**: Read/write access to application variables
  - **Function Capabilities**: Ability to call host functions
  - **Extension Entries**: Points where extensions can hook into application

**Code Example (Right Side):**
```python
State_Capability(
  name = "readPid",
  operation = read(ngx_pid)
)

Function_Capability(
  name = "nginxTime",
  prototype = (void) -> time_t,
  constraints = { rtn > 0 }
)

Extension_Entry(
  name="ProcessBegin",
  extension_entry = "ngx_http_process_request",
  prototype = (Request *r) -> int
)

Extension_Entry(
  name="updateResponseContent",
  extension_entry = "ngx_http_content_phase",
  prototype = (Request *r) -> int*
)
```

> Let's look at how EIM works in practice. EIM leverages the existing kernel eBPF verifier, but applies it to userspace extensions. We use static analysis and BPF Type Format (BTF) to generate specifications from source code.
>
> EIM defines three types of capabilities that extensions might need:
>
> First, State Capabilities represent the ability to read or write specific variables in the host application. For example, here we define a capability called "readPid" that allows reading the process ID variable.
>
> Second, Function Capabilities represent the ability to call specific functions in the host application. In this example, we define a capability called "nginxTime" that allows calling nginx's time function, with a constraint that the return value must be positive.
>
> Third, Extension Entries define the specific points in the application where extensions can hook in. Here we define two entry points: one at the beginning of request processing, and another at the content generation phase.
>
> What makes EIM powerful is its rich constraint system that can express complex relationships between arguments and return values, including pre-post conditions, along with high-level semantic facts about your code. For example, we can specify that a function allocates memory, performs I/O operations, or requires specific read/write capabilities for certain data fields. When you define these constraints, our EIM tooling converting relationships into assert statements, encoding annotations into BTF type information, and leveraging the kernel verifier's tag system to track memory allocations. This gives you fine-grained control over what extensions can do while maintaining the performance benefits of load-time verification rather than runtime checking.

### Slide 18: bpftime – Userspace eBPF runtime

**[VISUAL: Architecture diagram showing bpftime as a bridge between userspace applications and kernel eBPF]**

> We build the EIM model on **bpftime**, make it an extension framework:
- A userspace eBPF runtime compatible with kernel
- Support **Uprobe** / **USDT** / **syscall tracepoints** / **XDP** and other eBPF features in userspace
- 10+ map types and 30+ helpers support
- Run together with kernel eBPF
- The VM inside: **ubpf** or **llvm-bpf** (bpftime is a runtime, not just a simple VM like ubpf or rbpf)

📎 GitHub: [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

> Now, let's talk about bpftime, which is our concrete implementation of the EIM principles. bpftime is a userspace eBPF runtime that we've been developing in the past two years.
>
> What makes bpftime special is that it's not just a simple virtual machine like some other userspace eBPF implementations. It's a complete runtime that supports the full range of eBPF features in userspace, including Uprobes for function tracing, USDT for static tracepoints, syscall tracepoints, and like XDP for network processing.
>
> bpftime is designed to work alongside kernel eBPF, not replace it. You can use kernel eBPF for some tasks and bpftime for others, or even have them work together on the same workload. This gives you the flexibility to choose the right approach for each part of your application.
>
> Under the hood, bpftime can use different eBPF virtual machines like ubpf or llvm-bpf, giving you options depending on your performance and compatibility needs.

### Slide 19: Current support features

**[VISUAL: Feature matrix showing supported map types and program types]**

**Userspace eBPF shared memory map types:**
- `BPF_MAP_TYPE_HASH`
- `BPF_MAP_TYPE_ARRAY`
- `BPF_MAP_TYPE_PROG_ARRAY`
- `BPF_MAP_TYPE_RINGBUF`
- `BPF_MAP_TYPE_PERF_EVENT_ARRAY`
- `BPF_MAP_TYPE_PERCPU_ARRAY`
- `BPF_MAP_TYPE_PERCPU_HASH`
- ...

**User-kernel shared maps:**
- `BPF_MAP_TYPE_HASH`
- `BPF_MAP_TYPE_ARRAY`
- `BPF_MAP_TYPE_PERCPU_ARRAY`
- `BPF_MAP_TYPE_PERF_EVENT_ARRAY`
- ...

**Program types attachable in userspace:**
- `tracepoint:raw_syscalls:sys_enter`
- `tracepoint:syscalls:sys_exit_*`
- `tracepoint:syscalls:sys_enter_*`
- `uretprobe:*`
- `uprobe:*`
- `usdt:*`
- `xdp`
- GPUs via dynamic PTX injection

**Other capabilities:**
- Define static tracepoints and prog types in userspace apps
- ~30 kernel helper functions and **ufunc** (similar to `kfunc`)
- Kernel or userspace verifier supported
- JIT tested with `bpf_conformance`
- **Automatic safety verification without developer effort**:
  - No manual bounds checking
  - No custom sandboxing code
  - No performance-costly runtime checks

> here is a table of the support features of eBPF in bpftime. One of our longtime goals was to maintain compatibility with the kernel eBPF ecosystem, so we've implemented a wide range of map types and program types.
>
> For maps, which are the data structures eBPF programs use to store and share data, we support all the common types like hash maps, arrays, ring buffers, and perf event arrays. These can be shared between different userspace eBPF programs, and some can even be shared between userspace and kernel eBPF programs.
>
> For program types, we support the full range of userspace tracing mechanisms, including function entry and exit points (uprobes and uretprobes), static tracepoints (USDT), syscall tracepoints, and even XDP for network processing. We've even extended support to GPUs through dynamic PTX injection.
>
> Beyond these core features, bpftime offers several additional capabilities that make it a powerful platform for userspace extensions. You can define your own static tracepoints and program types in userspace applications, similar to how the kernel defines them. We've implemented about 30 kernel helper functions and added support for ufuncs, which are similar to kernel kfuncs but for userspace.
>
> A key advantage of bpftime is that neither extension developers nor host application developers need to implement manual safety checks. The verifier automatically ensures memory safety, prevents infinite loops, and enforces capability restrictions. This eliminates an entire class of security vulnerabilities that have historically plagued extension systems.

### Slide 20: bpftime Design

**[VISUAL: Architecture diagram showing components and data flow]**

> White components are from **eBPF**; orange components are new to **bpftime**.  
Blue arrows: compiling/loading flow  
Green arrows: runtime execution  
White arrows (black border): interact with eBPF maps

**Diagram Components:**
- eBPF compiler → eBPF bytecode
- bpftime Loader: Binary Rewriter, JIT Compiler, Verifier, Syscall Interposition
- Target Process:
  - bpftime user runtime
  - eBPF maps
  - hooks: `uprobe`, `userspace tracepoint`, `syscall tracepoint`
- Kernel side:
  - eBPF runtime
  - `tracepoint`, `kprobe`, `socket`

> This diagram shows the architecture of bpftime and how it fits into the overall eBPF ecosystem. The white components are standard eBPF components, while the orange ones are new additions from bpftime.
>
> The flow starts with the eBPF compiler, which generates eBPF bytecode just like it would for kernel eBPF. This bytecode is then processed by the bpftime loader, which includes several key components:
>
> - A binary rewriter that can dynamically inject hooks into running applications
> - A JIT/AOT compiler that translates eBPF bytecode to native machine code for performance
> - A verifier that ensures the eBPF program is safe to run
> - Syscall interposition to intercept and handle eBPF-related system calls, which supports the libbpf and eBPF control plane applications and keep compatibility with kernel eBPF.
>
> Once loaded, the eBPF program runs in the target process using the bpftime user runtime. It can interact with eBPF maps, both in userspace and shared with the kernel, and can hook into various points in the application through uprobes, userspace tracepoints, and syscall tracepoints.
>
> This architecture allows bpftime to provide the same capabilities as kernel eBPF, but in userspace, with better performance and without requiring kernel privileges.

### Slide 21: Get Started

**[VISUAL: Command line example with output]**

- Use `uprobe` to monitor `malloc` in `libc`, with hash maps in userspace
- Try eBPF in GitHub Codespaces (**unprivileged container**)

```bash
# Build and load eBPF program
make -C example/malloc
export PATH=$PATH:~/.bpftime/
bpftime load ./example/malloc/malloc
```

```bash
# Run the instrumented target
bpftime start ./example/malloc/victim

Hello malloc!
malloc called from pid 250215
continue malloc...
malloc called from pid 250215
```

>  Here's a simple example that uses uprobes to monitor malloc calls in libc.
>
> First, you build and load the eBPF program using the bpftime CLI. Then, you can run your target application with bpftime, and it will automatically instrument the application to trace malloc calls.
>
> You can run this in unprivileged containers, like GitHub Codespaces.
>
> This makes bpftime ideal for development, testing, and educational purposes, as well as for production environments where kernel access is restricted.
>
> Now, let's look at some specific use cases and performance benchmarks to see how bpftime performs in real-world scenarios.

### Slide 22: Evaluation & Use Cases

**[VISUAL: Overview diagram showing different use case categories with icons]**

Existing eBPF use cases can be run without or with minor fixes, with some additional customization cases

- **Observability**: Tracing, profiling, monitoring
- **Networking**: XDP, load balancing, packet filtering
- **Security**: Runtime verification, access control
- **Custom Extensions**: Application-specific plugins

> Now, let's look at some specific use cases and performance benchmarks to see how bpftime performs in real-world scenarios. The good news is that most eBPF use cases can run in bpftime with minimal or no changes, while also enabling new customization opportunities that weren't possible before.

### Slide 23: bpftime for Observability

**[VISUAL: Performance comparison chart showing kernel vs userspace tracing latency]**

**Why userspace tracing? Faster and More Flexible**
- **Faster** userspace `uprobe`
- **Faster** userspace memory access
- No tracing overhead on untraced processes (syscall tracepoint)

**What can we run in userspace?**
- Tools like `bcc` and `bpftrace`
- Complex observability agents with `kprobe` and `uprobe` (e.g., **Deepflow**)

🔗 [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

> So why do we want to do observability with eBPF in userspace?
>
> It's simple: userspace tracing is **faster and more flexible**. For example, Uprobes in the kernel take about **1000 nanoseconds**, but in userspace, we've brought that down to just **100 nanoseconds**. Similarly, memory access in userspace is about **10 times faster**—**4 nanoseconds** versus **40 nanoseconds** in the kernel. This speed difference happens because the kernel often has to translate memory addresses or run additional checks to access userspace memory.
>
> On top of that, there's less overhead on untraced processes, especially when dealing with syscall tracepoints.
>
> What can we run in userspace? With bpftime, popular tools like **bcc** and **bpftrace** can run completely in userspace where kernel eBPF is not available. And you can run more complex observability agents like Deepflow that combine kprobes and uprobes, improving performance by shifting part of the workload to userspace.

### Slide 24: bpftime Microbenchmark

**[VISUAL: Table comparing performance metrics between kernel eBPF and bpftime]**

| Bench Type              | eBPF (ns) | bpftime (ns) |
|------------------------|-----------|--------------|
| Uprobe                 | 3224      | 315          |
| Uretprobe              | 3997      | 381          |
| Syscall Tracepoint     | 152       | 233          |
| User memory read       | 45.0      | 2.2          |
| User memory write      | 46.8      | 8.1          |
| hash_map_update        | 62        | 30           |
| hash_map_delete        | 15        | 23           |
| hash_map_lookup        | 19        | 13           |

📌 Table 4: Microbenchmark comparison of `bpftime` and kernel `eBPF`.

> Let's look at some specific performance numbers. This table shows a microbenchmark comparison between bpftime and kernel eBPF for various operations.
>
> As we can see, bpftime is about 10 times faster than kernel eBPF uprobes. This is because kernel-based uprobes require a context switch into the kernel, while bpftime runs entirely in userspace.
>
> For memory operations, User memory access are also about 10 times faster in bpftime. This is because the kernel needs to use special mechanisms to safely access userspace memory, while bpftime can access it directly.
>
> For map operations, which are the data structures eBPF programs use to store and share data, bpftime is competitive with or faster than kernel eBPF. This is particularly important for high-throughput applications that need to process a lot of data quickly.
>
> The only area where kernel eBPF is faster is syscall tracepoints, which is expected since the kernel is already involved in syscall processing. But even here, bpftime is reasonably competitive.

### Slide 25: Bpftrace and BCC

**[VISUAL: Screenshots showing bpftrace and BCC tools running with bpftime]**

- **Bpftrace**: Can run entirely in userspace, no kernel eBPF/syscall/uprobe required
- **BCC**: Upper-half tools can run in userspace—targeting applications, runtimes, and syscall interface
- Ported and tested **bcc** and **bpftrace**

🔗 [https://github.com/eunomia-bpf/bpftime/tree/master/example/bpftrace](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpftrace)

> One of the most exciting aspects of bpftime is its compatibility with existing eBPF tools. We've ported and tested both bpftrace and BCC to work with bpftime, allowing these popular observability tools to run in userspace.
>
> Bpftrace, which is a high-level tracing language for eBPF, can now run entirely in userspace. This means you can use bpftrace for application tracing without requiring kernel eBPF support. This is particularly valuable in environments where kernel eBPF is not available or restricted, such as containerized environments or older kernels.
>
> Similarly, BCC tools that target applications, runtimes, and the syscall interface can now run in userspace. This includes popular tools for tracing function calls, syscalls, memory allocations, and more.

### Slide 26: Benchmark – syscount and sslsniff

**[VISUAL: Bar charts comparing performance impact of kernel eBPF vs bpftime]**

- **syscount** tool:
  - Kernel eBPF overhead: 10% reduction in RPS
  - Userspace eBPF: only 3% overhead
- **sslsniff** tool:
  - Kernel eBPF overhead: 28.06%
  - Userspace eBPF: only 7.41%

> Moving beyond microbenchmarks, let's look at how bpftime performs with observability tools. We tested two common tracing tools: syscount, which counts system calls, and sslsniff, which traces SSL/TLS traffic.
>
> For syscount, kernel eBPF introduces about a 10% reduction in requests per second (RPS) compared to no tracing. In contrast, bpftime reduces RPS by only 3%. This means that with bpftime, you can trace system calls with significantly less performance impact.
>
> The difference is even more dramatic with sslsniff. Kernel eBPF introduces a 28% overhead, while bpftime reduces this to just 7.4%.
>
> These results demonstrate that bpftime can significantly reduce the performance penalty of tracing tools, making it more feasible to use these tools without sacrificing application performance.

### Slide 27: bpftime for Userspace Networking

**[VISUAL: Diagram showing bpftime integration with kernel-bypass technologies like DPDK and AF_XDP]**

**Why userspace eBPF?**
- Kernel bypass is faster: **DPDK**, **AF_XDP**
- Leverage eBPF ecosystem

**Why bpftime?**
- Compared to eBPF-in-DPDK:
  - Lacks eBPF map/helper support
  - Lacks control-plane apps
  - `bpftime` is faster/more optimized with LLVM

**Results:**
- Up to **3× faster** in simple XDP network functions
- Up to **40% faster** in **katran**

> Now, let's talk about bpftime in the **networking** context. Why using userspace eBPF instead of running ebpf in kernel?

> We’ve seen kernel-bypass solutions like DPDK and AF_XDP. They can offer faster packet processing by bypassing the kernel. But with bpftime, you can combine the performance benefits of these kernel-bypass technologies with the extensive eBPF ecosystem. So, you get the best of both low-latency packet processing and the ability to use eBPF’s safety and existing tools.

> We can also use LLVM optimizations to further boost performance in userspace.

> Using bpftime, you can seamlessly integrate the eBPF XDP ecosystem into kernel-bypass applications.

> For instance, solutions like Katran, a high-performance load balancer, can benefit from the optimizations we’ve made in bpftime for userspace.

> bpftime can work with both AF_XDP and DPDK. You can run your XDP eBPF programs as if they were in the kernel, just load them with bpftime, and they’ll work like normal, while a DPDK app handles the network processing.

> Right now, there are some limitations with XDP_TX and XDP_DROP in userspace, but we’re actively working on solutions. We’re exploring ways to reinject packets into the kernel to support XDP_PASS.

### Slide 29: Benchmark – llvmbpf

**[VISUAL: Bar charts comparing performance of different eBPF runtimes]**

- `llvmbpf`: eBPF VM with LLVM JIT/AOT
- Standalone VM and compiler tool
- Separated from `bpftime` repo
- Easy to use

**Chart Benchmarks:**
- `strcmp`, `log2`, `prime`, `memcpy`, `switch`, etc.
- Comparison: `ubpf`, `rbpf`, `UserBPF`, `native`

🔗 [https://github.com/eunomia-bpf/llvmbpf](https://github.com/eunomia-bpf/llvmbpf)

> A key component of bpftime's performance is its underlying eBPF virtual machine, which we've developed as a separate project called llvmbpf. This is an eBPF VM with LLVM JIT/AOT (Just-In-Time/Ahead-Of-Time) compilation capabilities.
>
> llvmbpf is designed to be a standalone VM and compiler tool, making it easy to use in various contexts. We've separated it from the main bpftime repository to make it more modular and reusable.  As you can see in these benchmarks, llvmbpf outperforms traditional eBPF VMs like ubpf and rbpf across a range of operations.

### Slide 30: More use cases

**[VISUAL: Grid of application icons with performance metrics]**

- **Deepflow** as observability Agent: 1.5× less overhead  
- **Nginx plugin**: 10% less overhead than Lua and Wasm  
- **Fuse cache**: up to 10⁴× latency improvements  
- **Redis Durability Tuning**: 1.5× more throughput  
- **GPU tracing**  
- **Error injection**  
- **Hot patch**  
- ...

Checkout our coming papers for more detail

> We've covered observability and networking in detail, but bpftime has many other experimental applications. Let me briefly highlight a few more use cases we've explored.
>
> For observability, we've integrated bpftime with Deepflow, a popular observability agent, and achieved 1.5 times less overhead compared to the kernel-based approach.
>
> For web servers, we've created Nginx plugins using bpftime that have 10% less overhead than equivalent plugins written in Lua or WebAssembly. This is significant for high-traffic web servers where every bit of performance matters.
>
> In file systems, we've used bpftime to implement a FUSE cache that improves latency by up to 10,000 times in certain operations. This is because we can avoid expensive context switches between userspace and kernel.
>
> For databases, we've tuned Redis durability settings with bpftime and achieved 1.5 times more throughput. This shows how bpftime can be used not just for observability but also for performance optimization.
>
> We've also explored more experimental use cases like GPU tracing, error injection for testing, and hot patching of running applications. These demonstrate the flexibility of the bpftime approach.
>
> For more details on these use cases and our performance evaluations, please check out our upcoming papers.

### Slide 31: Roadmaps

**[VISUAL: Timeline with development milestones]**

> Still a long way to go...
- Not production ready yet  
- Improve stability and fix bugs  
- Make EIM more easy to use  

> For bpftime,we want to be clear that there's still a long way to go and it's not production-ready yet.
>
> Our roadmap focuses on three main areas: we need to improve stability and fix bugs, we want to make the Extension Interface Model easier to use. Right now, defining capabilities and constraints requires a deep understanding of the system. We want to simplify this process and provide better tools and documentation.
>
> Also, we want to expand the range of supported use cases and improve performance even further.
>
> We welcome contributions and feedback from the community. If you're interested in trying bpftime or contributing to its development, please check out our GitHub repository.

### Slide 32: Takeaways

**[VISUAL: Summary diagram showing bpftime bridging eBPF and Wasm approaches with key benefits highlighted]**

> **bpftime**: A new approach to userspace extensions  
- Combines strengths from both worlds:  
  - **eBPF's** verification and performance  
  - **Wasm-like** interface flexibility

**Enables:**
- High-performance extensions  
- Fine-grained safety controls  
- Compatibility with existing **eBPF ecosystem**

> To wrap up, let me leave you with the key takeaways from this talk.
>
> bpftime represents a new approach to userspace extensions that combines the strengths of both eBPF and WebAssembly worlds. From eBPF, it takes the load-time verification approach and performance focus. From WebAssembly, it takes the flexible interface model and capability-based security.
>
> This combination enables high-performance extensions with fine-grained safety controls, all while maintaining compatibility with the existing eBPF ecosystem. This means you can leverage your existing eBPF knowledge and tools while gaining the benefits of userspace execution.
>
> By providing a structured way to define and enforce capabilities, bpftime allows you to give extensions exactly the access they need - no more, no less - while maintaining high performance.
>
> Thank you for your attention!
