# Speech Script for OSDI 2025 bpftime talk

≈ 2 250 words, ~15 minutes

## [Slide 0]

Good morning! I'm **Yusheng Zheng** from UC Santa Cruz, presenting our OSDI '25 paper on extension systems. I'll show why current extensions force painful tradeoffs between safety and performance, then present our solution: EIM for fine-grained policies and bpftime, a userspace eBPF runtime, for efficient enforcement.

## [Slide 1] Extensions Everywhere

Extensions are everywhere in modern software. PostgreSQL has over 100 extensions that turn a basic database into specialized systems—PostGIS adds geospatial data support, TimescaleDB handles time-series analytics, all without changing PostgreSQL's core code. VS Code works the same way—it starts as a simple text editor but becomes a full development environment through extensions like language servers, debuggers, and productivity tools like GitLens. This pattern repeats across software: Nginx gains new features through modules, Redis uses Lua scripts, and browsers rely on ad blockers and developer tools.

## [Slide 2] Nginx Extension Example

Let me start with a concrete example to show you what extensions are and why we need them. Consider Nginx deployed as a reverse proxy. The original Nginx developers write the core server functionality. But different deployments need different behaviors. Some need firewalls to block malicious requests. Others need load balancers to distribute traffic. Many need monitoring for observability. Extensions solve this problem by allowing customization without modifying the original application source code. This approach preserves maintainability by keeping core code stable, reduces security risks, enables early experiment, and allows different teams to develop functionality independently without coordination overhead.

Here's how the extension execution model works in Nginx. A developer defines new logic as Nginx modules or plugins, and associates each extension with specific locations in Nginx's request processing pipeline, called extension entries. When a user runs Nginx, the system loads both the core Nginx binary and the configured extensions. Each time an Nginx worker thread reaches an extension entry—like when processing an incoming HTTP request—the thread jumps to the associated extension. It executes the extension logic within Nginx's runtime context. Once the extension completes, the thread returns to Nginx's core processing at the point immediately after the extension entry.

## [Slide 3] Extension Problems

However, Nginx extension systems face serious safety and performance challenges. Real-world incidents show the risks: In 2023, a malformed Lua plugin created an infinite loop inside Nginx, causing Bilibili's entire CDN to go down for hours. Similar issues plague other software—Redis scripts can enable remote code execution, and Apache modules suffer buffer overflows. These examples show that even mature plugin ecosystems can bring production services to a halt.

Meanwhile, many people using sandbox technology like WebAssembly for safer extension development, but these approaches impose a persistent 10–15 percent throughput penalty on HTTP request processing—that's simply too much overhead to pay in production environments. Nginx operators often disable these safer extension systems entirely due to performance concerns.

## [Slide 4] Extension Requirements

Nginx extension frameworks need three key features. First, fine-grained safety and interconnectedness trade-offs. Nginx extensions must interact with the web server by reading request headers and calling HTTP processing functions, but managers need to follow the principle of least privilege, granting only necessary permissions per extension. Second, isolation to protect Nginx extensions from core server bugs and vice versa. Third, efficiency with near-native speed execution, since Nginx extensions often run on critical paths like per-request processing where every millisecond matters for user experience.

## [Slide 5] State-of-the-Art Falls Short

Unfortunately, existing approaches cannot satisfy all requirements simultaneously. Dynamic loading achieves speed but provides no isolation or policies. Software Fault Isolation systems like WebAssembly deliver safety but carry 10–15 percent performance penalties. Subprocess isolation ensures separation but has untenable IPC overhead. Kernel eBPF uprobes offer isolation but trap into the kernel on every invocation, costing microseconds each time.

## [Slide 6] Outline

Now that we've established the motivation and challenges, let me outline our two-part solution and walk you through how we'll address these problems.

We present a two-part solution. First, the Extension Interface Model (EIM) treats every extension capability as a named resource. We split the work into development time, where application developers declare possible capabilities, and deployment time, where extension managers choose minimal privilege sets following least privilege principles.

Second, bpftime is a new runtime that efficiently enforces EIM using three key techniques: offline eBPF verification for zero runtime safety checks, Intel Memory Protection Keys for fast domain switching, and concealed extension entries that eliminate overhead for unused hooks. Together, they provide kernel-grade safety with library-grade performance while maintaining eBPF compatibility.

Our evaluation on six real-world applications shows bpftime can reduce overhead by up to 6x compared to solutions like WebAssembly, bringing performance to near-native levels while providing strong safety guarantees.

## [Slide 7] EIM: Extension Interface Model

To enable fine-grained safety-interconnectness trade-offs, we introduce the Extension Interface Model, or EIM. EIM treats extension capabilities as named resources with a two-phase specification approach.

Let me explain this using our Nginx example. In the extension ecosystem, we have four key roles. First, Nginx application developers write the core web server code. Second, extension developers create plugins like firewalls, load balancers, and monitoring tools. Third, the extension manager—typically a system administrator or DevOps engineer—decides which extensions to deploy and what privileges each should have. Finally, end users send HTTP requests that trigger both the host application and extensions.

EIM captures this separation of concerns through capabilities as resources.

The key insight is splitting specification into two phases. During development time, Nginx developers annotate their code to declare possible extension behaviors. At deployment time, the extension manager writes policies that grant minimal privilege sets to specific extensions. This separation means managers can refine security policies in production without touching application source code, enabling true least-privilege extension deployment.

## [Slide 8] EIM Development-Time Specification

Now let me show you how EIM works in practice. During development time, Nginx developers annotate their code to declare what extensions could possibly do. They might add a state capability called `readPid` for accessing the process ID, a function capability `nginxTime()` for getting timestamps, complete with pre- and post-conditions, and extension entries like `processBegin` when request processing starts.

These annotations are automatically extracted and compiled into the binary. This happens once during development and creates a complete map of what extensions could ever access. The key insight is that developers only declare possibilities—they don't decide what actually gets used.

<!-- State access capabilities control reading and writing variables like request headers or connection counts. Function call capabilities govern invoking Nginx APIs like `nginx_time()` or `ngx_http_finalize_request()`, complete with pre- and post-conditions. Hardware resource capabilities limit CPU instructions and memory access patterns. -->

## [Slide 9] EIM Deployment-Time Specification

At deployment time, the system administrator writes simple policies that grant minimal privileges to each extension. 

An observability extension might only read request data and call logging functions. A firewall extension gets both read and write access to modify responses. A load balancer needs network capabilities to contact upstream servers.

These policies live completely outside the application code. You can refine security settings in production without recompiling anything. This separation enables true least-privilege deployment while keeping the original application unchanged.

## [Slide 10] bpftime: userspace eBPF​ extension framework

Now you might ask, "Can't we just use existing frameworks to enforce EIM policies?" Unfortunately, as we discussed in the previous work, current frameworks make painful trade-offs that prevent efficient EIM enforcement.

We built bpftime specifically to enforce EIM efficiently while maintaining complete eBPF compatibility. Why we are using eBPF? It provides proven safety through verification and a rich ecosystem we can reuse. Our efficiency comes from binary rewriting with concealed extension entries, which is similar to eBPF Uprobes, and we achieve isolation using Intel Memory Protection Keys. This compatibility is crucial—existing eBPF tools work immediately with bpftime, and extensions can share data with kernel eBPF programs for full system customization, from user-level to kernel-level.

But ensuring eBPF compatibility presented a major challenge. The Linux eBPF ecosystem consists of tightly coupled components—compilers, runtime libraries, and the kernel—that are nearly impossible to disentangle. Prior user-level eBPF systems tried re-implementing the entire eBPF technology stack and ultimately failed to provide reasonable performance and compatibility.

Instead, bpftime takes a different approach. We identify a narrow waist in the current eBPF ecosystem and interpose at that point. Specifically, we intercept eBPF-related system calls and the shared map mechanism for data sharing between extensions. This lets us reuse the proven eBPF ecosystem while adding just the minimal new components needed for userspace deployment.

So, Here's how bpftime works at a high level. We intercept eBPF system calls before they reach the kernel. Our loader converts EIM policies into bytecode assertions and feeds everything through the kernel's proven eBPF verifier for safety guarantees. After JIT compilation to native code, we use binary rewriting to patch trampolines into the target application only when extensions are actually loaded. At runtime, we use memory protection keys to protect and execute the extension.

## [Slide 11] Real-World Use Cases

To prove our approach works, we built six real-world applications. For security, we created an Nginx firewall that blocks malicious URLs in real time. For reliability, we built a Redis extension that bridges the durability gap between losing thousands of writes versus taking a 6× performance hit. For performance, we accelerated FUSE file operations with in-process caching. For observability, we ported existing tools like DeepFlow, syscount, and sslsniff to demonstrate seamless eBPF compatibility. bpftime is open source on GitHub with an active community, and these applications demonstrate both the versatility of our approach and real-world user adoption.

## [Slide 12] Performance Results: Nginx Firewall

Let me show you the performance impact. For our Nginx firewall, we compared different extension approaches under a realistic workload. In this diagram, the more to the top, the higher throughput, the better. Lua and WebAssembly extensions impose 11–12 percent throughput loss—that's significant overhead that many operators can't accept in production. Our bpftime implementation achieves the same security functionality with only 2 percent overhead. That's a 5× to 6× improvement over existing approaches.

## [Slide 13] Performance Results: SSL Monitoring

For observability, consider sslsniff, which monitors encrypted TLS traffic—crucial for debugging production microservices.The figure shows a clear performance comparison across different data sizes from 1K to 256K bytes. Also, the more to the top, the better. With kernel eBPF, this monitoring costs 28 percent throughput loss. That's prohibitive for production use. With bpftime, the same monitoring functionality costs only 7 percent overhead. 

## [Slide 14] Take-Aways (On Outline, not separate slide)

Let me close with three key takeaways. First, EIM provides the missing piece for fine-grained extension control—you can now specify precise least-privilege policies per extension entry without touching application source code. Second, bpftime shows that you don't have to choose between safety and performance. We achieve safety with near-native performance using offline verification, hardware isolation, and concealed trampolines. Third, maintaining eBPF compatibility means you can adopt our approach immediately without changing your existing workflows.

<!-- 
Looking ahead, we're expanding bpftime to support GPU and ML workloads, broadening the scope of safe, efficient extension deployment beyond traditional systems programming.

However, current bpftime and EIM still have some limitations. First, EIM tools and policies are mainly for compiled applications, and we are working on supporting more languages. Also, you need to write the extension code in eBPF, which is not easy for some users. -->

## Thank You & Questions (On Outline, not separate slide)

Thank you for your attention. **bpftime** is open-source under the MIT license at GitHub. You can get started today by running it as a drop-in replacement for eBPF applications. We welcome your issues, pull requests, and collaboration. I'm happy to take your questions.

## **Complete Slide Deck (15 slides, 16:9)**

---

**Slide 0: Title**
**Extending Applications Safely & Efficiently**

Yusheng Zheng¹ • Tong Yu² • Yiwei Yang¹ • Yanpeng Hu³
Xiaozheng Lai⁴ • Dan Williams⁵ • Andi Quinn¹

¹UC Santa Cruz   ²eunomia-bpf Community   ³ShanghaiTech University
⁴South China University of Technology   ⁵Virginia Tech

---

**Slide 1: Extensions Everywhere**

- **Extensions are everywhere:** PostgreSQL (PostGIS), API gateways (Lua plugins), Redis (custom scripts), browsers (ad blockers)

---

**Slide 2: Nginx Extension Example**

- **Extension execution model:** Thread → Extension entry → Jump to extension → Execute → Return to host

---

**Slide 3: Extension Problems**

- **Real-world safety violations:** Bilibili CDN outage, Apache buffer overflow, Redis RCE
- **Performance penalty**: WebAssembly/Lua impose 10-15% overhead

---

**Slide 4: Extension Requirements**

- **Fine-grained safety** and interconnectedness trade-offs
- **Isolation** to protect extensions from core server bugs and vice-versa
- **Efficiency** with near-native speed execution

---

**Slide 5: State-of-the-Art Falls Short**

No single approach meets all requirement​

Dynamic loading: fast but no isolation or policy
enforcement​

Software Fault Isolation (e.g., WebAssembly): safety
with 10–15 % performance penalty​

Subprocess isolation: strong separation but high IPC
overhead​

Kernel eBPF uprobes: isolation at microsecond-level
trap cost
**Problem**: No single framework satisfies all requirements

---

**Slide 6: Outline**

- **Background & motivation:** Extensions
- **Extension Interface Model (EIM):** Fine-grained Interface
- **bpftime Runtime:** safety & performance
- **Evaluation**

---

**Slide 7: EIM - Extension Interface Model**

**Four key roles in Nginx ecosystem:**
  - **Nginx developers**
  - **Extension developers**
  - **Extension manager**
  - **End users**

- **Capabilities as resources:** Separate development-time possibilities from deployment-time policies

---

**Slide 8: EIM Development-Time Specification**

**Nginx developers annotate code:**
```c
State_Capability(name="readPid", operation=read(ngx_pid))
Function_Capability(name="nginxTime") 
Extension_Entry(name="processBegin")
```

Automatically extracted into capability manifest

---

**Slide 9: EIM Deployment-Time Specification**

Extension Developer or Manager write simple policies to explore interconnectedness/safety trade-offs

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

**Slide 10: bpftime: userspace eBPF extension framework**

**Two innovations:**
1. **Named capabilities** → Precise control
2. **Separate concerns** → Development ≠ Deployment

Treats safety and interconnectedness as independent dimensions

**Example policies:**
- Monitoring extension: read-only access to specific variables
- Firewall extension: read/write for response modification

**bpftime advantages:**
- **eBPF ecosystem compatibility**
- **Work together with kernel eBPF extensions**

---

**High-level approach:**
- Intercept eBPF syscalls before kernel
- Convert EIM policies into bytecode assertions
- Use kernel's proven eBPF verifier for safety
- JIT compile to native code
- Binary rewriting for trampolines only when needed
- MPK for fast security domain switching

[use the figure here]

**The eBPF compatibility challenge:**
- Linux eBPF has tightly coupled components (compilers, runtime, kernel)
- Prior user eBPF failed by re-implementing entire stack
- **bpftime solution**: Interpose on eBPF syscalls only

**Key design principles:**
1. **Lightweight EIM enforcement**
2. **Concealed extension entries**

Reuse proven eBPF ecosystem + minimal new components

---

**Slide 11: Real-World Use Cases**

**Six applications demonstrate breadth:**

**Security**: Nginx firewall (malicious URL blocking)
**Reliability**: Redis durability tuner (bridge everysec/alwayson gap)
**Performance**: FUSE metadata cache (in-process acceleration)
**Observability**: DeepFlow, syscount, sslsniff (eBPF compatibility)

---

**Slide 12: Performance Results - Nginx Firewall**

**Workload**: 8 threads, 64 connections, realistic traffic

**Results**:
- **Lua/WebAssembly**: 11-12% throughput loss
- **bpftime**: 2% overhead
- **Improvement**: 5-6× better performance

**Impact**: Crosses threshold for production acceptability

---

**Slide 13: Performance Results - SSL Monitoring**

**Use case**: sslsniff for encrypted TLS traffic monitoring

**Results**:
- **Kernel eBPF**: 28% throughput loss (prohibitive)
- **bpftime**: 7% overhead (acceptable)
- **Improvement**: 4× reduction in monitoring cost

**Impact**: Makes encrypted monitoring practical in production

---

**Slide 14: Take-Aways & Future Work on outline slide again**

**Three key takeaways:**
1. **EIM** → Fine-grained least-privilege policies without source changes
2. **bpftime** → Don't choose between safety and performance

**Future work**: GPU and ML workload support

**Get started**: Drop-in eBPF replacement available today

```bash
bpftime load ./example/malloc/malloc
bpftime start nginx -c ./nginx.conf
```

---

**Slide 15: Thank You & Questions**

**GitHub**: github.com/eunomia-bpf/bpftime (MIT license)
**Ready to use**: Drop-in replacement for eBPF applications

We welcome issues, pull requests, and collaboration!

---
