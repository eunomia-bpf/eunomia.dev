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

> Specifically, I want to talk about why we need extensions, what makes them challenging to handle correctly, and how our current approaches to managing extensions might not be good enough. Then I'll introduce a new approach to managing extensions called the Extension Interface Model (EIM) and our experimental userspace eBPF runtime called bpftime that implements these principles. This is also a research project that has been accepted by OSDI 2025.

### Slide 3: Software Extensions

**[VISUAL: Timeline list of extension systems with names and icons for each example, handle writting style]**

- Extensions have deep roots in software development
- Common examples across the industry:
  - Web servers: nginx/Apache modules
  - Databases: Redis and PostgreSQL extensions
  - Editors: Vim, Emacs, VSCode plugins
  - Cloud-native: Kubernetes extensions, wasm modules in cloud-native systems
  - Kernel: eBPF programs, kernel modules

> So, first, let's step back a little bit. Software extensions aren't new—they have a very long history. If you've been around software for a while, you might remember Apache HTTP Server modules, browser plugins, or even IDE extensions like VSCode plugins. Extensions are everywhere because we, as engineers, really love flexibility. We love building a core application and then letting other developers or even users customize or add extra features later, without needing to rewrite the original software.
>
> Think about web servers. In nginx or Apache, you can use modules or plugins to add extra functionality like authentication, caching, compression, or even application-level firewalls. Database systems like Redis and PostgreSQL have extensions to support new query types, custom data formats, or security audits. Our editors—Vim, Emacs, VSCode—all thrive thanks to the flexibility offered by extensions or plugins. In cloud-native systems like Kubernetes, you can use extensions to add observability features like custom metrics collection and tracing. You can extend networking with custom CNI plugins or service mesh sidecars. Security extensions can add policy enforcement, vulnerability scanning, or runtime threat detection. The flexibility of extensions is what makes Kubernetes such a powerful platform for building modern applications. In kernel level, for example, you can use eBPF programs or kernel modules to add custom kernel behaviors.

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
> We want flexibility because it makes our software adaptable. Users and administrators want to tweak things to meet their specific requirements without waiting for the core developers to implement changes. But flexibility without isolation is risky. Extensions, by definition, are third-party or at least externally-developed code. You might trust your core engineering team, but trusting external code is a different story. Even if it's not malicious, external code can have bugs. And we all know how easily bugs can creep into our systems, causing all sorts of problems—crashes, performance degradation, or even security vulnerabilities.

### Slide 5: Real-World Extension Failures

**[VISUAL: "Incident Report" style graphics showing each failure case with impact metrics]**

- **Bilibili**: Production outage from nginx extension infinite loop
  - Service disruption affecting millions of users
- **Apache HTTP Server**: Buffer overflows in Lua modules
  - Security vulnerabilities and system crashes
- **Redis**: Improperly sandboxed Lua scripts
  - Remote code execution vulnerabilities
- These aren't theoretical risks—they've cost companies money and reputation

> Let me give you some real-world examples. A few years back, the popular video streaming site Bilibili suffered a serious production outage because one of their nginx extensions got stuck in an infinite loop. Apache HTTP Server had similar issues where buffer overflow bugs in a Lua-based module caused crashes and security holes. Even Redis had cases where improperly sandboxed Lua scripts resulted in remote code execution vulnerabilities. These aren't theoretical risks—these are things that have actually hurt big companies and cost serious money and reputation.
>
> So, isolation and safety become absolutely critical. We don't want a bug in one extension to crash our entire system. We don't want a poorly-written plugin causing our service to slow down. We definitely don't want external code exposing our internal data to attackers.
>
> But achieving isolation isn't easy. There's always a tradeoff. If you isolate too strictly, you lose expressiveness—extensions become so limited they're no longer useful. If you're too loose, bugs or malicious code can harm your system.

### Slide 6: Traditional Extension Approaches

**[VISUAL: Spectrum showing different approaches with isolation vs performance tradeoffs]**

- **Dynamic Loading**: Shared libraries, DLLs, LD_PRELOAD
  - ✅ Fast and flexible
  - ❌ Minimal isolation, no safety boundary
- **Scripting Languages**: Lua, Python
  - ✅ Easy integration
  - ❌ Safety depends on host implementation
- **Process-Level Isolation**: Subprocesses, API calls
  - ✅ Strong isolation
  - ❌ High overhead from context switches

> So, how have engineers traditionally tried to deal with this balance?
>
> We've tried many things. Early on, we used dynamically loadable modules—shared libraries, DLLs, LD_PRELOAD hacks. They're great for speed and flexibility. But they have virtually no isolation. A bug in a dynamically-loaded module is a bug in your entire application. There's no safety boundary.
>
> Later, people adopted scripting languages like Lua and Python for extensions. Languages like Lua are still hugely popular because they're easy to integrate and relatively safe. Redis and nginx are good examples here. But Lua scripts have their own challenges. They rely heavily on the host application to provide security boundaries, which means if the host makes a mistake—maybe not checking array bounds correctly or missing resource limits—the safety promise falls apart. And as we've seen from real incidents, host applications frequently get this wrong.
>
> A lot of extensions, especially in the cloud-native space, are integrated through process level isolation, such as subprocesses or API calls. They use OS context switches, adding microseconds of overhead—which is OK for distributed systems, but not for performance-critical applications.
>
> Wasm and eBPF are two popular extension frameworks that have been around for a while. Let's take a look at how they handle the balance between interconnectedness and safety.

### Slide 7: WebAssembly for Extensions (SFI approach)

**[VISUAL: Diagram showing Wasm sandbox with import/export boundaries between host and extension]**

- Software fault isolation (SFI) provides strong security
- Explicit import/export mechanisms for host communication
- Benefits:
  - Strong memory isolation
  - Cross-platform compatibility
  - Growing ecosystem
  - Reduces need for manual safety checks in extensions
- Limitations:
  - Heavyweight for thousands of small extensions
  - Communication overhead between extension and host
  - Complex interface management
  - Host application still needs careful API design

> Then WebAssembly (or Wasm) came along, promising much better isolation and performance. Wasm uses software fault isolation techniques, which means it's safer because it doesn't blindly trust the extension code. That's why many modern browsers and even server-side applications are moving towards Wasm.
>
> One significant benefit of Wasm is that it reduces the need for extension developers to implement their own safety checks - the runtime handles memory isolation automatically. However, the host application still needs to carefully design its API to prevent security issues when extensions interact with it.
>
> But Wasm introduces another issue—the interfaces between extensions and hosts. Wasm needs explicit import/export mechanisms to talk to the host application, and managing this communication can be tricky. It's powerful, but it's still heavyweight, especially if you're running thousands of little extensions or making frequent calls back and forth.
>
> What do I mean by "heavyweight"? Every time a Wasm module needs to access host functionality, it must cross the sandbox boundary. This involves marshaling data between the Wasm environment and host, which adds overhead. For applications that need frequent host interactions, these costs add up quickly.

### Slide 8: eBPF for Extensions

**[VISUAL: Architecture diagram showing eBPF in kernel space with verifier, JIT compiler, and hooks into various subsystems]**

- Originally for network packet filtering, now widely used
- Key components:
  - Verifier ensures safety before execution
  - JIT compiler for near-native performance
  - Userspace tracing via uprobes and USDT
- Performance-first design:
  - Load-time verification instead of runtime checks
  - No sandboxing overhead
  - Automatic instrumentation without application changes

> At the kernel level, eBPF has become the star of the show. Originally designed for network packet filtering, eBPF is now widely used for security monitoring, observability, and even performance optimization at kernel level. eBPF programs run isolated from the kernel itself, thanks to a verifier and JIT compiler that ensures safe execution.
>
> But eBPF is not limited to kernel. For userspace tracing, eBPF leverages mechanisms like uprobes (dynamic function instrumentation) and USDT probes (User Statically-Defined Tracing points). These probes eliminate the need for manual instrumentation - developers don't need to modify their application code to enable tracing. Uprobes can dynamically instrument any function in userspace programs, while USDT provides static tracepoints that are more efficient but require compile-time integration. This automatic instrumentation capability makes eBPF powerful for observability without requiring changes to the target application.
>
> Compared with Wasm, eBPF has a history of focusing on performance first, which led to the design of using a verifier to check the safety of the extension at load time, instead of runtime checking or SFI (software fault isolation) like Wasm.

### Slide 9: Three Core Requirements for Extension Frameworks

**[VISUAL: Triangle diagram with "Interconnectedness," "Safety," and "Efficiency" at the corners, showing the tension between them]**

- **Interconnectedness**: Extension's ability to interact with host
  - Reading data, modifying state, calling internal functions
  - Example: nginx extension reading request details
- **Safety**: Limiting extensions' ability to harm the application
  - Memory safety, control flow restrictions, resource limits
- **Efficiency**: Performance impact on the system
- **The Fundamental Tension**: More interconnectedness typically means less safety

> As we can see, there are three core requirements for extension frameworks: **interconnectedness**, **safety** and **efficiency**. Efficiency is easy to understand, it's about performance. Let's dive deeper into the core issue we're really focusing on today—this fundamental tension between **interconnectedness** and **safety**. This is what we think makes software extensions challenging, and also why getting the interface right is so hard, yet so important.
>
> So, first off, what do I mean by interconnectedness? Simply put, interconnectedness is how much power we give an extension to interact with the host application. Think of it like this: extensions usually need to do something meaningful—they're not just isolated pieces of code floating around. They actually need to communicate with the main application to read data, modify state, or even call existing functions inside the application. For example, let's say you have an nginx web server. An extension that monitors web requests for security needs to read the request details from the host application. It might also need to call a built-in nginx function to quickly respond or block a suspicious request. This is interconnectedness—extensions working directly with the application's resources or calling its internal functions.
>
> On the flip side, safety is how much we limit an extension's ability to interact with or alter the main application. Safety means that if there's a bug in your extension—let's face it, we all write buggy code sometimes—this bug won't crash your whole web server or compromise your entire application. If we didn't care about safety, extensions could freely do whatever they wanted—read or write any memory, call any function, open files they shouldn't, or even alter sensitive configuration data. Obviously, this would be a recipe for disaster. A single small mistake in an extension could take down a whole production system or open it up to security vulnerabilities. And trust me, that has happened many times before. Like that time Bilibili's nginx extensions got into a loop and brought their production servers offline, causing a major outage.
>
> So here's the key issue: interconnectedness and safety are fundamentally at odds. The more interconnectedness you allow—the more you let extensions interact with the host—the less inherently safe it becomes. And vice versa: to keep things perfectly safe, you have to restrict interconnectedness. And that means extensions become severely limited in their usefulness. You can't have an effective firewall extension if you don't allow it to inspect web requests. You can't meaningfully monitor performance if your monitoring extension can't read internal state from the host application. So, balancing this tension is a core challenge.
>
> Now, historically, software frameworks have not handled this tradeoff very well. Usually, they fall into one of two extremes. Either they allow too much interconnectedness, like dynamically loaded modules—these run fast, sure, but they provide almost no safety at all. One bug and your entire application crashes. Or, on the other extreme, they provide strong safety through heavy isolation, like sandboxed scripting environments or subprocess isolation methods. But these can cripple interconnectedness and performance—extensions often become slow and limited in what they can do.
>
> So, what we've found is that the key to managing this tension—this interconnectedness versus safety tradeoff—is the interface we choose for extensions. If your extension framework's interface can carefully define and verify exactly which resources and functions an extension can use, you can precisely manage this tension. Ideally, you give the extension just enough interconnectedness to do its job—but absolutely no more. This sounds simple, but current systems struggle to achieve this.

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
> This approach is particularly relevant for extensions because it standardizes how extensions declare their requirements and capabilities, making them more portable and easier to reason about. For example, a browser extension written as a Wasm component could clearly specify which DOM elements it needs access to.

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
  interface file-system {
    resource file {
      read: func() -> list<u8>
      write: func(data: list<u8>) -> result<_, errno>
    }
    open: func(path: string) -> result<file, errno>
  }
```

> The Component Model implements capability-based security through "resource handles" - unforgeable references that grant access to specific resources. These handles can be passed between components, allowing fine-grained control over which extensions can access which resources.
>
> For example, in this WIT interface definition for a file system, the "file" resource can only be obtained through the "open" function. An extension would need to be explicitly granted access to this interface to manipulate files.
>
> This capability-based approach aligns with the principle of least privilege we discussed earlier - extensions only get access to exactly what they need.
>
> However, the Component Model still faces performance challenges. While it provides better interface definition than basic Wasm, it still requires runtime checks at interface boundaries and data marshaling when crossing those boundaries. This creates overhead, especially for extensions that frequently interact with the host.
>
> Additionally, the host application still needs to carefully design which capabilities it exposes to extensions. The Component Model provides the mechanism for capability control, but the security policy must still be defined by the host.
>
> This is where our approach with bpftime differs - we shift these checks to load time rather than runtime, eliminating the performance overhead while maintaining the security benefits.

### Slide 12: eBPF Interface Approach

**[VISUAL: Diagram showing eBPF verifier analyzing code paths at load time]**

- Load-time verification through static analysis
- Helper functions and maps for host interaction
- Tradeoffs:
  - ✅ No runtime overhead after verification
  - ✅ Automatic instrumentation capabilities
  - ✅ Efficient host interaction through maps
  - ❌ Limited to kernel context traditionally
  - ❌ Userspace requires costly kernel traps

> In contrast, eBPF uses a verifier to analyze programs before they run. This verifier checks all possible execution paths to ensure memory safety and prevent infinite loops. This approach shifts safety checks from runtime to load-time, eliminating runtime overhead once the program is loaded. 
>
> eBPF programs interact with the host through predefined helper functions and map structures, which are designed for efficient data sharing.
>
> The key difference is when safety is enforced: Wasm checks at runtime (adding overhead to every operation), while eBPF verifies everything upfront (adding overhead only when loading the program).
>
> However, current userspace eBPF implementations like uprobes require costly kernel traps. Every time a userspace extension runs, it triggers a context switch into the kernel, runs the extension, and switches back. This is too slow for many performance-critical scenarios.
>
> What if we could combine eBPF's load-time verification with efficient userspace execution? That's where our approach comes in.

### Slide 13: Extension Interface Model (EIM) - Overview

**[VISUAL: Diagram showing EIM as a bridge between extensions and host applications with capability controls]**

- Our approach: Extension Interface Model (EIM)
- Inspired by eBPF's verifier and Wasm's component model
- Core insight: Treat all extension-host interactions as explicit capabilities
- Follows principle of least privilege for extensions
- Designed for both safety and performance in userspace extensions

> Our approach, the Extension Interface Model or EIM, takes inspiration from eBPF's verifier-based model and Wasm's component model, but applies it at the user-space level for general extension frameworks.
>
> The core insight of EIM is treating every interaction between an extension and its host as an explicit capability that can be controlled. Think of it as a permission system designed specifically for extensions.
>
> EIM follows the principle of least privilege - each extension should only have access to exactly what it needs to function, nothing more. This is critical for security, as it minimizes the potential damage from bugs or exploits.
>
> Unlike previous approaches that force you to choose between safety and performance, EIM is designed to provide both by shifting safety checks to load time while maintaining fine-grained control over what extensions can do.

### Slide 14: EIM - Key Advantages

**[VISUAL: Comparison diagram showing EIM advantages over traditional approaches]**

- Key advantages of the EIM approach:
  - **Load-time safety guarantees**: 
    - Verify safety before execution
    - Eliminate runtime overhead after loading
  - **Fine-grained control**:
    - Precisely specify which capabilities each extension can use
    - Different permissions for different extension types
  - **Comprehensive safety without manual checks**:
    - Extensions don't need to implement safety checks
    - Verification happens automatically at load time

> Why did we choose a verifier-based approach for EIM? There are three key reasons:
>
> 1. **Load-time safety guarantees** - We verify everything before execution, eliminating runtime overhead. This is similar to eBPF's approach but applied to userspace extensions. Once an extension passes verification, it can run at full speed without additional checks.
>
> 2. **Fine-grained control** - We can precisely tailor what each extension can access. For example, a logging extension might only get read access to request headers, while a security extension might get both read and write access to modify suspicious requests.
>
> 3. **Elimination of most manual safety checks** - We remove the burden from developers, preventing the bugs and security flaws that have plagued extension systems
>
> This third point is important. Unlike traditional approaches where either the extension developer needs to implement safety checks or the host application needs to create custom security boundaries, EIM handles safety verification automatically. This prevents the exact kind of bugs we saw in the real-world failures like Redis and Apache HTTP Server, where manually implemented safety checks had flaws.

### Slide 15: EIM - Usage Model and Principals

**[VISUAL: Triangle diagram showing the three principals in EIM: Application Developers, Extension Manager, and Input Provider]**

- Key roles in the EIM model:
  - **Application Developers**: Trusted but fallible developers of host application and extensions
  - **Extension Manager**: Trusted administrator who configures extension permissions
  - **Input Provider**: Potentially untrusted source of inputs to the application
- Security goal: Limit impact of exploitable bugs, not prevent malicious extensions
- Example: Web browser with password manager and ad blocker extensions
  - Extensions need different capabilities (DOM access, storage access)
  - EIM limits what each extension can do if exploited

> Let me explain EIM's usage model. It involves three key roles: First, we have application developers who create both the host application and extensions. They're trusted but fallible - meaning they're not malicious, but their code might contain bugs. Second, we have the extension manager - typically a system administrator - who configures which extensions can access which capabilities. Third, we have input providers - potentially untrusted sources of input to the application, like websites in a browser.
>
> EIM doesn't try to prevent malicious extensions from being installed - that's a different problem. Instead, it aims to limit the damage that could happen if a buggy extension gets exploited through malicious input. For example, in a web browser with a password manager and an ad blocker, the extension manager would give the ad blocker permission to modify DOM elements related to ads, but not access to stored passwords. If a malicious website exploits the ad blocker, it would only be able to modify ad-related DOM elements, not steal passwords.

### Slide 16: EIM - Development-Time Specification

**[VISUAL: Code example showing development-time EIM specification with annotations]**

- Created by application developers
- Defines three types of capabilities:
  - **State Capabilities**: Read/write access to application variables
    - Example: `readPid` for accessing process ID
  - **Function Capabilities**: Ability to call host functions
    - Example: `nginxTime` with constraint that return value > 0
  - **Extension Entries**: Points where extensions can hook into application
    - Example: `processBegin` at request processing start

> EIM specifications come in two parts. First, at development time, application developers create a specification that defines all the possible capabilities that extensions might need. This includes three types of capabilities:
>
> State capabilities represent the ability to read or write specific variables in the host application. For example, in nginx, we might define a capability called "readPid" that allows reading the process ID variable.
>
> Function capabilities represent the ability to call specific functions in the host application. For instance, we might define a capability called "nginxTime" that allows calling nginx's time function. We can also add constraints, like requiring that the return value must be positive.
>
> Extension entries define the specific points in the application where extensions can hook in. For example, in nginx, we might define extension entries at the beginning of request processing and at the content generation phase.
>
> This development-time specification essentially creates a catalog of all the ways extensions might interact with the host application.

### Slide 17: EIM - Deployment-Time Specification

**[VISUAL: Code example showing deployment-time EIM specification with annotations]**

- Created by system administrators/extension managers
- Defines **Extension Classes** that specify:
  - Which extension entry point to use
  - Exactly which capabilities are allowed
- Example extension classes:
  - `observeProcessBegin`: Can read request data but not modify it
  - `updateResponse`: Can both read and write to response data
- Enforces principle of least privilege for each extension

> The second part of EIM is the deployment-time specification, created by system administrators or extension managers. This is where the actual security boundaries get defined.
>
> The deployment-time specification defines extension classes, each associated with a specific extension entry point. For each class, it specifies exactly which capabilities from the development-time specification are allowed.
>
> For example, we might define an "observeProcessBegin" class that allows extensions to read request data but not modify it. We might also define an "updateResponse" class that allows both reading and writing to response data.
>
> This approach enforces the principle of least privilege - each extension gets only the capabilities it needs to do its job, and nothing more. If an extension gets compromised, the damage is limited to the capabilities it was granted.

### Slide 18: bpftime: EIM's Efficient Runtime Implementation

**[VISUAL: Architecture diagram showing how bpftime implements EIM principles]**

- bpftime: Our userspace eBPF runtime that implements EIM
- Key features:
  - Brings eBPF concepts fully into userspace
  - Uses eBPF verifier for load-time safety checks
  - Near-native performance without kernel traps
  - Compatible with kernel eBPF ecosystem
  - Supports 10+ map types and 30+ helpers
  - **Automatic safety verification without developer effort**:
    - No manual bounds checking
    - No custom sandboxing code
    - No performance-costly runtime checks

> Now, let's talk about bpftime, which is our concrete implementation of the EIM principles. bpftime is a userspace eBPF runtime that supports tracing features like Uprobe, USDT, syscall tracepoints, and even network features like XDP, all in userspace. 
>
> It supports more than 10 map types and 30 helpers, so it's highly compatible with the kernel eBPF ecosystem. It builds on top of eBPF concepts but brings them fully into user-space. bpftime uses the same verification and safety guarantees as kernel-level eBPF, ensuring that an extension can't step outside its defined boundaries. Plus, it leverages modern hardware isolation features (like Intel's Memory Protection Keys—MPK) to give near-native performance without the heavyweight overhead of context switches or traps that traditional solutions like uprobes suffer from.
>
> A key advantage of bpftime is that neither extension developers nor host application developers need to implement manual safety checks. The verifier automatically ensures memory safety, prevents infinite loops, and enforces capability restrictions. This eliminates an entire class of security vulnerabilities that have historically plagued extension systems, where manually implemented safety checks had flaws or were forgotten entirely.
>
> You can use your familiar way to develop and deploy eBPF programs, but in userspace. bpftime can run alongside kernel eBPF, using kernel eBPF maps and working together with kprobe.
>
> We started developing bpftime 2 years ago, and adapted it to efficiently enforce these permissions from the EIM model this year.

### Slide 19: bpftime - Concealed Extension Entries

**[VISUAL: Before/after diagram showing binary code without hooks, then with dynamically injected hooks]**

- Novel approach to reduce overhead when extensions aren't active
- Traditional approach: Static hooks always present in code
  - Causes performance overhead even when unused
- bpftime's approach: Concealed extension entries
  - Uses binary rewriting to dynamically inject hooks only when needed
  - Zero overhead when extensions aren't in use
  - Automatically activates when extensions are loaded
- Example: Web server with 50 potential extension points
  - Traditional: All 50 checks happen on every request
  - bpftime: Only points with active extensions have hooks

> One of our most innovative features in bpftime is what we call "concealed extension entries." This addresses a common problem with extension frameworks: performance overhead at extension points, even when no extensions are active.
>
> Think of it like toll booths on a highway. Traditional extension systems are like having toll booths at every possible exit, even if nobody ever takes that exit. Every car still has to slow down to check if they need to pay. This slows down traffic unnecessarily.
>
> Traditionally, applications with extension support have static hooks in their code. These hooks check if an extension is present and, if so, call into it. But these checks happen every time the code runs, even if no extensions are loaded, causing unnecessary overhead.
>
> bpftime takes a different approach. We use binary rewriting to dynamically inject hooks into applications only when extensions are actually loaded. When no extensions are active at a particular point, there's zero overhead - the application runs at full speed as if it had no extension support at all. It's like only building toll booths at exits people actually use, and leaving the highway clear everywhere else.
>
> This is actually the opposite of how binary rewriting is typically used. Most systems use it to add new extension points to applications that weren't designed for extensions. We use it to hide extension points that are already there, activating them only when needed.
>
> For example, imagine a web server with 50 potential extension points. In a traditional system, all 50 checks happen on every request, even if only 2 extensions are active. With bpftime, only the 2 points with active extensions have hooks, eliminating overhead at the other 48 points.
>
> This approach dramatically improves performance for applications with many potential extension points that are only occasionally used.

### Slide 20: bpftime for Observability

**[VISUAL: Performance comparison chart showing kernel vs userspace tracing latency]**

- Userspace tracing advantages:
  - **10x faster**: Uprobes (100ns vs 1000ns in kernel)
  - **10x faster**: Memory access (4ns vs 40ns in kernel)
  - Reduced overhead on untraced processes
- Compatible with existing tools:
  - Run BCC and bpftrace in userspace
  - Combine kprobes and uprobes for hybrid approaches
  - Shift workload to improve overall performance

> So why do we want to do observability with eBPF in userspace?
>
> It's simple: userspace tracing is **faster and more flexible**. For example, Uprobes in the kernel take about **1000 nanoseconds**, but in userspace, we've brought that down to just **100 nanoseconds**. Similarly, memory access in userspace is about **10 times faster**—**4 nanoseconds** versus **40 nanoseconds** in the kernel. This speed difference happens because the kernel often has to translate memory addresses or run additional checks to access userspace memory.
>
> On top of that, there's less overhead on untraced processes, especially when dealing with syscall tracepoints.
>
> What can we run in userspace?
>
> With userspace tracing, tools like **bcc** and **bpftrace** can run completely in userspace where kernel eBPF is not available. And you can run more complex observability agents that combine kprobes and uprobes, improving performance by shifting part of the workload to userspace.

### Slide 21: bpftime for Userspace Networking

**[VISUAL: Diagram showing bpftime integration with kernel-bypass technologies like DPDK and AF_XDP]**

- Advantages over kernel networking:
  - Combines kernel-bypass performance with eBPF ecosystem
  - Works with DPDK and AF_XDP
  - Leverages LLVM optimizations for userspace
- Use cases:
  - High-speed packet processing
  - Network function virtualization
  - Advanced load balancing

> Now, let's talk about bpftime in the **networking** context. Why use userspace eBPF instead of running eBPF in the kernel?
>
> We've seen kernel-bypass solutions like **DPDK** (Data Plane Development Kit) and **AF_XDP** (Address Family XDP). They can offer faster packet processing by bypassing the kernel. But with bpftime, you can combine the performance benefits of these kernel-bypass technologies with the extensive eBPF ecosystem. So, you get the best of both low-latency packet processing and the ability to use eBPF's safety and existing tools.
>
> We can also use **LLVM optimizations** to further boost performance in userspace.

### Slide 22: bpftime with Userspace Network Integration

**[VISUAL: Architecture diagram showing XDP program flow in userspace with bpftime]**

- Seamless integration with eBPF XDP ecosystem
- Compatible with high-performance applications:
  - Katran load balancer integration
  - Works with both AF_XDP and DPDK
- Current work in progress:
  - XDP_TX and XDP_DROP limitations
  - Solutions for XDP_PASS packet reinjection

> Using **bpftime**, you can seamlessly integrate the **eBPF XDP ecosystem** into kernel-bypass applications. 
>
> For instance, solutions like **Katran**, a high-performance load balancer from Facebook, can benefit from the optimizations we've made in bpftime for userspace. 
>
> bpftime can work with both **AF_XDP** and **DPDK**. You can run your XDP eBPF programs as if they were in the kernel, just load them with bpftime, and they'll work like normal, while a DPDK app handles the network processing.
>
> Right now, there are some limitations with **XDP_TX** (packet transmission) and **XDP_DROP** (packet dropping) in userspace, but we're actively working on solutions. We're exploring ways to reinject packets into the kernel to support **XDP_PASS** (passing packets to the network stack).

### Slide 23: Control Plane Support

**[VISUAL: Diagram showing bpftime control plane architecture with syscall hooking and userspace runtime]**

- Maintains compatibility with eBPF ecosystem:
  - Loading/unloading programs
  - Configuring maps
  - Monitoring and debugging interfaces
- Implementation methods:
  - LD_PRELOAD for syscall hooking
  - Kernel eBPF integration
  - Seamless connection to userspace runtime

"One of the core features that allows bpftime to remain compatible with the existing eBPF ecosystem is its **control plane support** for userspace eBPF.

Control planes in eBPF are usually responsible for tasks like loading and unloading programs, 
configuring maps, and providing monitoring and debugging interfaces. bpftime can fully supports this in userspace by hooking syscalls using **LD_PRELOAD** or kernel eBPF, and connect to the userspace runtime.

### Slide 24: Performance Benchmarks

**[VISUAL: Bar charts comparing performance metrics between kernel eBPF and bpftime]**

- Network performance improvements:
  - Up to **3x faster** in simple XDP network functions
  - Up to **40% faster** in real-world applications like Katran
- Tracing performance:
  - **10x faster** memory access
  - **10x faster** uprobes
- Demonstrates userspace eBPF can outperform kernel-based solutions while maintaining compatibility

"Now, let's see the **performance benchmarks**. 

We've benchmarked a variety of eBPF programs running with bpftime. bpftime has achieved up to **3x faster performance** in simple XDP network functions. In real-world applications
like Katran, bpftime can acheive up to  **40% faster**.

This shows that userspace eBPF can be fasyer kernel-based solutions, while retaining the flexibility that makes eBPF powerful."

### Slide 25: Conclusion

**[VISUAL: Summary diagram showing bpftime bridging eBPF and Wasm approaches with key benefits highlighted]**

- bpftime: A new approach to userspace extensions
- Combines strengths from both worlds:
  - eBPF's verification and performance
  - Wasm-like interface flexibility
- Enables:
  - High-performance extensions
  - Fine-grained safety controls
  - Compatibility with existing tools
- Available today for experimentation and production use

"In conclusion, bpftime is a new approach to userspace eBPF that combines the safety and performance of kernel-level eBPF with the flexibility of userspace extensions. It's a path forward for developers and administrators alike, enabling highly customized, performant, and secure applications."
