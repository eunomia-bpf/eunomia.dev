# Speech Script for OSDI 2025 bpftime talk

≈ 2 250 words, ~15 minutes

## [Slide 0]

Good morning, everyone. Thank you for joining our session. My name is **Yusheng Zheng** from UC Santa Cruz. Today I will present our OSDI '25 paper titled **"Extending Applications Safely and Efficiently."**

Here's what I'll cover today. First, I'll explain what extensions are and why they matter using a concrete Nginx example. Then I'll show why current extension systems have serious problems, forcing painful tradeoffs between safety, performance, and flexibility. Next, I'll present our two-part solution: EIM, a fine-grained capability model for least-privilege policies, and bpftime, an efficient runtime using offline verification and hardware isolation. I'll demonstrate six real-world use cases from Nginx firewalls to distributed tracing. Our evaluation shows 5-6× better performance than WebAssembly and Lua, and 14× faster than kernel eBPF for uprobes.

## [Slide 1] Extensions: A Concrete Example

Extensions are everywhere in modern software. PostgreSQL has over 100 extensions for everything from geospatial data to time-series analytics. Kong API Gateway relies on Lua plugins for rate limiting and authentication. Emacs users install packages for language support and productivity tools. Vim has thousands of plugins for syntax highlighting and development workflows. Redis uses Lua scripts for custom data processing. Even browsers depend on extensions for ad blocking and developer tools.

Let me start with a concrete example to show you what extensions are and why we need them. Consider Nginx deployed as a reverse proxy. The original Nginx developers write the core server functionality. But different deployments need different behaviors. Some need firewalls to block malicious requests. Others need load balancers to distribute traffic. Many need monitoring for observability. Extensions solve this problem by allowing customization without modifying the original application source code.

Here's how the extension execution model works. A developer defines new logic for the program as a set of extensions, and associates each extension with a specific location in the host application, called an extension entry. When a user runs the application, the system loads both the host application and the user's configured extensions. Each time an application thread reaches an extension entry, the thread jumps to the associated extension. It executes the extension in the extension runtime context. Once the extension completes, the thread returns to the host at the point immediately after the extension entry.

## [Slide 2] Extension Problems and Requirements

However, these extension systems face serious safety and performance challenges. Real-world incidents show the risks: In 2023, a malformed Lua plugin created an infinite loop inside Nginx, causing Bilibili's entire CDN to go down for hours. Apache's Lua module suffered buffer overflows that crashed httpd. Redis Lua scripts enabled remote code execution through stack overflows. These examples show that even mature plugin ecosystems can bring production systems to a halt.

Meanwhile, operators often disable WebAssembly or language-based sandboxes in production due to their persistent 10–15 percent throughput penalty. This creates a painful tension between safety, extensibility, and performance.

Extension frameworks need three key features. First, fine-grained safety and interconnectedness trade-offs. Extensions must interact with the host by reading state and calling functions, but managers need to follow the principle of least privilege, granting only necessary permissions per extension. Second, isolation to protect extensions from host application bugs. Third, efficiency with near-native speed execution, since extensions often run on critical paths like per-request processing.

## [Slide 3] State-of-the-Art Falls Short

Unfortunately, existing approaches cannot satisfy all requirements simultaneously. Dynamic loading achieves speed but provides no isolation or policies. Software Fault Isolation systems like WebAssembly deliver safety but carry 10–15 percent performance penalties. Subprocess isolation ensures separation but has untenable IPC overhead. Kernel eBPF uprobes offer isolation but trap into the kernel on every invocation, costing microseconds each time.

## [Slide 4] Contribution: EIM + bpftime

We present a two-part solution. First, the Extension Interface Model (EIM) treats every extension capability as a named resource. We split the work into development time, where application developers declare possible capabilities, and deployment time, where extension managers choose minimal privilege sets following least privilege principles.

Second, bpftime is a new runtime that efficiently enforces EIM using three key techniques: offline eBPF verification for zero runtime safety checks, Intel Memory Protection Keys for fast domain switching, and concealed extension entries that eliminate overhead for unused hooks. Together, they provide kernel-grade safety with library-grade performance while maintaining 100% eBPF compatibility.

## [Slide 5] EIM: Extension Interface Model

To enable fine-grained control, we introduce the Extension Interface Model, or EIM. EIM treats extension capabilities as named resources with a two-phase specification approach.

Let me explain this using our Nginx example. In the extension ecosystem, we have four key roles. First, Nginx application developers write the core web server code. Second, extension developers create plugins like firewalls, load balancers, and monitoring tools. Third, the extension manager—typically a system administrator or DevOps engineer—decides which extensions to deploy and what privileges each should have. Finally, end users send HTTP requests that trigger both the host application and extensions.

EIM captures this separation of concerns through capabilities as resources. State access capabilities control reading and writing variables like request headers or connection counts. Function call capabilities govern invoking Nginx APIs like `nginx_time()` or `ngx_http_finalize_request()`, complete with pre- and post-conditions. Hardware resource capabilities limit CPU instructions and memory access patterns.

The key insight is splitting specification into two phases. During development time, Nginx developers annotate their code to declare the universe of possible extension behaviors—what state could be accessed, which functions could be called, where extensions could hook. This creates a comprehensive capability manifest embedded in the binary.

At deployment time, the extension manager writes policies that grant minimal privilege sets to specific extensions. A monitoring extension might only read request data and call logging functions. A firewall extension needs both read and write access to modify responses. A load balancer requires network capabilities to contact upstream servers.

This separation means managers can refine security policies in production without touching application source code, enabling true least-privilege extension deployment.

## [Slide 6] EIM Development-Time Specification

Now let me show you how EIM works in practice. During development time, Nginx developers annotate their code to declare what extensions could possibly do. They might add a state capability called `readPid` for accessing the process ID, a function capability `nginxTime()` for getting timestamps, and extension entries like `processBegin` when request processing starts.

These annotations are automatically extracted and compiled into the binary. This happens once during development and creates a complete map of what extensions could ever access. The key insight is that developers only declare possibilities—they don't decide what actually gets used.

## [Slide 7] EIM Deployment-Time Specification

Here's where the magic happens. At deployment time, the system administrator writes simple policies that grant minimal privileges to each extension. An observability extension might only read request data and call logging functions. A firewall extension gets both read and write access to modify responses. A load balancer needs network capabilities to contact upstream servers.

The beauty is that these policies live completely outside the application code. You can refine security settings in production without recompiling anything. This separation enables true least-privilege deployment while keeping the original application unchanged.

## [Slide 8] EIM Summary

To summarize EIM, we've solved the fine-grained control problem through two innovations. First, we model every extension capability as a named resource that can be precisely granted or denied. Second, we separate development time concerns from deployment time policies.

This fills a critical gap. Existing frameworks either give you no control at all, or they bundle everything into coarse-grained categories. EIM lets you say "this monitoring extension can only read request headers and call logging functions" while "this firewall can read and modify response content"—all without changing a single line of application code.

The key breakthrough is treating safety and interconnectedness as independent dimensions that can be balanced precisely for each use case.

## [Slide 9] bpftime: Why We Need a New Runtime

Now you might ask, "Can't we just use existing frameworks to enforce EIM policies?" Unfortunately, no. Current frameworks make painful trade-offs that prevent efficient EIM enforcement. Software fault isolation like WebAssembly adds 10-15% runtime overhead. Subprocess isolation requires expensive context switches. Kernel eBPF uprobes trap into the kernel on every single function call.

We built bpftime specifically to enforce EIM efficiently while maintaining complete eBPF compatibility. This compatibility is crucial—it means existing eBPF tools work immediately with bpftime, and extensions can share data with kernel eBPF programs for comprehensive monitoring that spans both kernel and userspace.

## [Slide 10] bpftime Architecture

Here's how bpftime works at a high level. We intercept eBPF system calls before they reach the kernel. Our loader converts EIM policies into bytecode assertions and feeds everything through the kernel's proven eBPF verifier for safety guarantees. After JIT compilation to native code, we use binary rewriting to patch trampolines into the target application only when extensions are actually loaded. At runtime, we flip memory protection keys to switch security domains and execute the native extension code directly.

The key insight is reusing the existing eBPF ecosystem while adding just the minimal components needed for userspace deployment with EIM enforcement.

## [Slide 11] bpftime: Key Challenges and Design

Now you might ask, "Can't we just expand existing extension frameworks to enforce EIM policies?" Unfortunately, that approach won't work. Existing frameworks provide safety and isolation through heavyweight operating-system isolation or software-fault isolation techniques like WebAssembly. These are already inefficient, imposing 10-15% overhead. Adding EIM enforcement on top would degrade their performance even further, making them unsuitable for production use.

So we designed bpftime as a new extension framework specifically for compiled applications. But ensuring eBPF compatibility presented a major challenge. The Linux eBPF ecosystem consists of tightly coupled components—compilers, runtime libraries, and the kernel—that are nearly impossible to disentangle. Prior user-level eBPF systems tried re-implementing the entire eBPF technology stack and ultimately failed to provide reasonable performance and compatibility.

Instead, bpftime takes a different approach. We identify a narrow waist in the current eBPF ecosystem and interpose at that point. Specifically, we intercept eBPF-related system calls and the shared map mechanism for data sharing between extensions. This lets us reuse the proven eBPF ecosystem while adding just the minimal new components needed for userspace deployment.

bpftime employs two key design constraints that work together. First, we use separate lightweight approaches for EIM enforcement versus isolation—similar to how KFlex uses two separate verification techniques for kernel extensions. We enforce EIM safety without runtime overhead using eBPF-style verification, and provide efficient isolation using ERIM-style intra-process hardware isolation. Second, we introduce concealed extension entries using binary rewriting, so extension entries are zero-cost when not in use.

## [Slide 12] Real-World Use Cases

To prove our approach works, we built six real-world applications. For security, we created an Nginx firewall that blocks malicious URLs in real time. For reliability, we built a Redis extension that bridges the durability gap between losing thousands of writes versus taking a 6× performance hit. For performance, we accelerated FUSE file operations with in-process caching. For observability, we ported existing tools like DeepFlow, syscount, and sslsniff to demonstrate seamless eBPF compatibility.

## [Slide 13] Performance Results: Nginx Firewall

Let me show you the performance impact. For our Nginx firewall, we compared different extension approaches under a realistic workload. Lua and WebAssembly extensions impose 11–12 percent throughput loss—that's significant overhead that many operators can't accept in production. Our bpftime implementation achieves the same security functionality with only 2 percent overhead. That's a 5× to 6× improvement over existing approaches.

This result matters because it crosses the threshold where extension overhead becomes acceptable in production environments.

## [Slide 14] Performance Results: SSL Monitoring

For observability, consider sslsniff, which monitors encrypted TLS traffic—crucial for debugging production microservices. With kernel eBPF, this monitoring costs 28 percent throughput loss. That's prohibitive for production use. With bpftime, the same monitoring functionality costs only 7 percent overhead.

This improvement makes encrypted traffic monitoring practical in performance-sensitive environments where every percentage point of overhead matters.

## [Slide 15] Take-Aways and Future Work

Let me close with three key takeaways. First, EIM provides the missing piece for fine-grained extension control—you can now specify precise least-privilege policies per extension entry without touching application source code. Second, bpftime shows that you don't have to choose between safety and performance. We achieve kernel-grade safety with library-grade performance using offline verification, hardware isolation, and concealed trampolines. Third, maintaining 100% eBPF compatibility means you can adopt our approach immediately without changing your existing workflows.

Looking ahead, we're expanding bpftime to support GPU and ML workloads, broadening the scope of safe, efficient extension deployment beyond traditional systems programming.

## [Slide 16] Thank You & Questions

Thank you for your attention. **bpftime** is open-source under the MIT license at **github.com/eunomia-bpf/bpftime**. You can get started today by running it as a drop-in replacement for eBPF applications. We welcome your issues, pull requests, and collaboration. I'm happy to take your questions.

## **Complete Slide Deck (16 slides, 16:9)**

---

**Slide 0: Title**
**Extending Applications Safely & Efficiently**

Yusheng Zheng¹ • Tong Yu² • Yiwei Yang¹ • Yanpeng Hu³
Xiaozheng Lai⁴ • Dan Williams⁵ • Andi Quinn¹

¹UC Santa Cruz   ²eunomia-bpf Community   ³ShanghaiTech University
⁴South China University of Technology   ⁵Virginia Tech

---

**Slide 1: Extensions - A Concrete Example**

- **Extensions are everywhere:** PostgreSQL (PostGIS), API gateways (Lua plugins), Redis (custom scripts), browsers (ad blockers)

Nginx plugin as an example:​

- **What are extensions?** Customize software without modifying source code
- **Why do we need them?** Different deployments, different needs

- **Extension execution model:** Thread → Extension entry → Jump to extension → Execute → Return to host

---

**Slide 2: Extension Problems & Requirements**

- **Real-world safety violations:** Bilibili CDN outage, Apache buffer overflow, Redis RCE
- **Performance penalty**: WebAssembly/Lua impose 10-15% overhead
- **A painful tension**: Safety vs. Extensibility vs. Performance


---

**Slide 3: State-of-the-Art Falls Short**

| Approach | Safety | Isolation | Efficiency | Fine-Grained Control |
|----------|---------|-----------|------------|---------------------|
| Native Loading | ✗ | ✗ | ✓ | ✗ |
| SFI (Wasm, Lua) | Limited | ✓ | ✗ (10-15% overhead) | ✗ |
| Subprocess | ✓ | ✓ | ✗ (context switches) | Limited |
| eBPF uprobes | ✓ | ✓ | ✗ (kernel traps) | Limited |

**Problem**: No single framework satisfies all requirements

---

**Slide 4: Contribution - EIM + bpftime**

- **Extension Interface Model (EIM)**: Fine-grained capability control
- **bpftime Runtime**: Kernel-grade safety with library-grade performance

---

**Slide 5: EIM - Extension Interface Model**

**Four key roles in Nginx ecosystem:**
  - **Nginx developers**
  - **Extension developers**
  - **Extension manager**
  - **End users**

- **Capabilities as resources:**

- Separate development-time possibilities from deployment-time policies

---

**Slide 6: EIM Development-Time Specification**

**Nginx developers annotate code:**
```c
State_Capability(name="readPid", operation=read(ngx_pid))
Function_Capability(name="nginxTime") 
Extension_Entry(name="processBegin")
```

Automatically extracted into capability manifest

---

**Slide 7: EIM Deployment-Time Specification**

Extension Developer or Manager  write simple policies to explore interconnectedness/safety trade-offs

```yaml
observeProcessBegin:
  entry: "processBegin"
  allowed: [readPid, nginxTime, read(request)]

updateResponse:
  entry: "updateResponseContent"  
  allowed: [read(request), write(response)]
```

Refine security policies in production **without recompiling**

---

**Slide 8: EIM Summary**

Existing frameworks → no control OR coarse-grained bundles

**Two innovations:**
1. **Named capabilities** → Precise control
2. **Separate concerns** → Development ≠ Deployment

Treats safety and interconnectedness as independent dimensions

**Example policies:**
- Monitoring extension: read-only access to specific variables
- Firewall extension: read/write for response modification

---

**Slide 9: bpftime - Why We Need a New Runtime**

Can't existing frameworks enforce EIM efficiently?

- **WebAssembly/SFI**: 10-15% overhead
- **Subprocess isolation**: Expensive switches
- **Kernel eBPF uprobes**: Kernel traps

**bpftime advantages:**
- **eBPF ecosystem compatibility**
- **Work together with kernel eBPF extensions**

---

**Slide 10: bpftime Architecture**

**High-level approach:**
- Intercept eBPF syscalls before kernel
- Convert EIM policies into bytecode assertions
- Use kernel's proven eBPF verifier for safety
- JIT compile to native code
- Binary rewriting for trampolines only when needed
- MPK for fast security domain switching

[use the figure here]

**Key insight**: Reuse existing eBPF ecosystem + minimal new components

---
**Slide 11: bpftime - Key Challenges & Design**

**Why not expand existing frameworks?**
- Heavyweight isolation is inefficient
- Adding EIM would degrade performance further

**The eBPF compatibility challenge:**
- Linux eBPF has tightly coupled components (compilers, runtime, kernel)
- Prior user eBPF failed by re-implementing entire stack
- **bpftime solution**: Interpose on eBPF syscalls only

**Key design principles:**
1. **Lightweight EIM enforcement**
2. **Concealed extension entries**

Reuse proven eBPF ecosystem + minimal new components

---

**Slide 12: Real-World Use Cases**

**Six applications demonstrate breadth:**

**Security**: Nginx firewall (malicious URL blocking)
**Reliability**: Redis durability tuner (bridge everysec/alwayson gap)
**Performance**: FUSE metadata cache (in-process acceleration)
**Observability**: DeepFlow, syscount, sslsniff (eBPF compatibility)

---

**Slide 13: Performance Results - Nginx Firewall**

**Workload**: 8 threads, 64 connections, realistic traffic

**Results**:
- **Lua/WebAssembly**: 11-12% throughput loss
- **bpftime**: 2% overhead
- **Improvement**: 5-6× better performance

**Impact**: Crosses threshold for production acceptability

---

**Slide 14: Performance Results - SSL Monitoring**

**Use case**: sslsniff for encrypted TLS traffic monitoring

**Results**:
- **Kernel eBPF**: 28% throughput loss (prohibitive)
- **bpftime**: 7% overhead (acceptable)
- **Improvement**: 4× reduction in monitoring cost

**Impact**: Makes encrypted monitoring practical in production

---

**Slide 15: Take-Aways & Future Work**

**Three key takeaways:**
1. **EIM** → Fine-grained least-privilege policies without source changes
2. **bpftime** → Don't choose between safety and performance
3. **100% eBPF compatibility** → Immediate adoption possible

**Future work**: GPU and ML workload support

**Get started**: Drop-in eBPF replacement available today

```bash
bpftime load ./example/malloc/malloc
bpftime start nginx -c ./nginx.conf
```

---

**Slide 16: Thank You & Questions**

**GitHub**: github.com/eunomia-bpf/bpftime (MIT license)
**Ready to use**: Drop-in replacement for eBPF applications

We welcome issues, pull requests, and collaboration!

---
