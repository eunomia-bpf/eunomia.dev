# Speech Script for OSDI 2025 bpftime talk

≈ 2,250 words, ~15 minutes

## [Slide 0]

Good morning! I'm **Yusheng Zheng** from UC Santa Cruz, presenting our OSDI '25 paper on extension systems. 

## [Slide 0.5] Outline

This is the outline of my talk, I'll first show why current extensions frameworks force painful tradeoffs between safety and performance, then look at our solution: EIM for fine-grained policies and bpftime, a userspace eBPF runtime, for efficient enforcement.

## [Slide 1] Extensions Everywhere

Extensions are everywhere in modern software. PostgreSQL has over 100 extensions that turn a basic database into specialized systems—PostGIS adds geospatial data support, TimescaleDB handles time-series analytics, all without changing PostgreSQL's core code. VS Code works the same way—it starts as a simple text editor but becomes a full development environment through extensions like language servers, debuggers, and productivity tools like GitLens. This pattern repeats across software: Nginx gains new features through modules, Redis uses Lua scripts, and browsers rely on ad blockers and developer tools.

> too much text

different deployments need different behaviors. Extensions like Nginx modules solve this problem by allowing customization without modifying the original application source code. This approach preserves maintainability by keeping core code stable, reduces security risks, enables early experimentation, and allows different teams to develop functionality independently.

> use general description for the 2 questions What are extensions? Why do we needthem? 

> 2-3 sentences

people extension nginx Some need firewalls to block malicious requests. 
Others need load balancers to distribute traffic. 
Many need monitoring for observability.

## [Slide 2] Nginx Extension Example

conside User wants to have a firewall to block malicious requests​ 

> make them more around firewalls.

Here's how the extension execution model works in Nginx. A developer defines new logic as Nginx modules or plugins, and associates each extension with specific locations in Nginx's request processing pipeline, called extension entries. When a user runs Nginx, the extension runtime system loads both the core Nginx binary and the configured extensions. Each time an Nginx worker thread reaches an extension entry—like when processing an incoming HTTP request url—the thread jumps to the associated extension. It executes the extension logic within Nginx's runtime context. Once the extension completes, the thread returns to Nginx's core processing at the point immediately after the extension entry.

> simplify the image for only one extension entry
> flow chart

## [Slide 3] Extension Problems

However, Nginx extension systems face serious safety and performance challenges. Real-world incidents show the risks: In 2023, a malformed Lua plugin created an infinite loop inside Nginx, causing Bilibili's entire CDN to go down for hours. Similar issues plague other software—Redis scripts can enable remote code execution, and Apache modules suffer buffer overflows. These examples show that even mature plugin ecosystems can bring production services to a halt.

Meanwhile, many people are using sandbox technology like WebAssembly for safer extension development, but these approaches impose a persistent 10–15 percent throughput penalty on HTTP request processing—that's simply too much overhead to pay in production environments. Nginx operators often disable these safer extension systems entirely due to performance concerns.

> fix the image for only one extension entry

## [Slide 4] Extension Requirements

These real-world problems highlight what extension frameworks actually need. Extension frameworks need three key features. 

First, fine-grained safety and interconnectedness trade-offs. For example, Nginx extensions must interact with the web server by reading request headers and calling HTTP processing functions, but systems managers need to follow the principle of least privilege, granting only necessary permissions per extension. like in firewall.... but in observability...

> more context 

Second, isolation to protect Nginx extensions from core server bugs and vice versa. Third, efficiency with near-native speed execution, since Nginx extensions often run on critical paths like per-request processing where every millisecond matters for user experience.

## [Slide 5] State-of-the-Art Falls Short

Unfortunately, existing approaches cannot satisfy all requirements simultaneously. Dynamic loading achieves speed but provides no isolation or policies. Software Fault Isolation systems like WebAssembly deliver safety but carry 10–15 percent performance penalties. Subprocess isolation ensures separation but has untenable IPC overhead. Kernel eBPF uprobes offer isolation but trap into the kernel on every invocation, costing microseconds each time.

> include citations and papers in the slide 

## [Slide 5.5] Contribution

> To help nav tradoffs....

We present a two-part solution. First, the Extension Interface Model (EIM) treats every extension capability as a named resource. We split the work into development time, where application developers declare possible capabilities, and deployment time, where extension managers choose minimal privilege sets following least privilege principles.

Second, bpftime is a new runtime that efficiently enforces EIM using three key techniques: offline eBPF verification for zero runtime safety checks, Intel Memory Protection Keys for fast domain switching, and concealed extension entries that eliminate overhead for unused hooks. Together, they provide kernel-grade safety with library-grade performance while maintaining eBPF compatibility.

Our evaluation on six real-world applications shows bpftime can reduce overhead by up to 6x compared to solutions like WebAssembly, bringing performance to near-native levels while providing strong safety guarantees.

> make them to figtures

## [Slide 6] Outline and transition to EIM

Now that we've established the motivation and challenges, let me outline our two-part solution and walk you through how we'll address these problems. I'll first explain our Extension Interface Model for fine-grained policy specification, then demonstrate how bpftime efficiently enforces these policies while maintaining eBPF compatibility.

## [Slide 7] EIM: Extension Interface Model

> high level overview.

So how do we achieve this? To enable fine-grained safety-interconnectedness trade-offs, we introduce the Extension Interface Model, or EIM.


> use only 2 roles

> 1. Developer ()
> 2. Extension manager

> the key insight is manager spec fine-grained safety-interconnectedness trade-offs, but them don't know .... those need to be make by develoepr.... the fine-grained safety-interconnectedness trade-offs are deployment time feature.

> general description for the 2 phases, not nginx

During development time, Nginx developers annotate their code to declare possible extension behaviors. , the extension manager writes policies that grant minimal privilege sets to specific extensions. deployment time EIM.

EIM treats extension capabilities as named resources with a two-phase specification approach.

> add one like workflow chart , 2 phase, develoment time, deployment time, as two box.

## [Slide 8] EIM Development-Time Specification

> Let us use nginx as an example....

Now let me show you how EIM works in practice. During development time, Nginx developers annotate their code to declare what extensions could possibly do. They might add a state capability called `readPid` for accessing the process ID, a function capability `nginxTime()` for getting timestamps, complete with pre- and post-conditions, and extension entries like `processBegin` when request processing starts.

These annotations are automatically extracted and compiled into the binary. This happens once during development and creates a complete map of what extensions could ever access. The key insight is that developers only declare possibilities—they don't decide what actually gets used.

> white background for the code

> keep only the developer, and make the figure around host application.

## [Slide 9] EIM Deployment-Time Specification

> more around fine-grained safety-interconnectedness trade-offs

At deployment time, the system administrator writes simple policies that grant minimal privileges to each extension. 

An observability extension might only read request data and call logging functions. A firewall extension gets both read and write access to modify responses. A load balancer needs network capabilities to contact upstream servers.

These policies live completely outside the application code. You can refine security settings in production without recompiling anything. This separation enables true least-privilege deployment while keeping the original application unchanged.

> use the same figure 

## [Slide 9.5] Transition to bpftime

Now you might ask, "Can't we just use existing frameworks to enforce EIM policies?" Unfortunately, as we discussed in the previous work, current frameworks make painful trade-offs that prevent efficient EIM enforcement. So, we built a new userspace eBPF extension framework called bpftime.

## [Slide 10] bpftime: userspace eBPF extension framework

We built bpftime specifically to enforce EIM efficiently while maintaining complete eBPF compatibility. Why we are using eBPF? It provides proven safety through verification and a rich ecosystem we can reuse. Our efficiency comes from binary rewriting with concealed extension entries, which is similar to eBPF Uprobes, and we achieve isolation using Intel Memory Protection Keys. This compatibility is crucial—existing eBPF tools work immediately with bpftime, and extensions can share data with kernel eBPF programs for full system customization, from user-level to kernel-level.

> add popup or diagram near the verifier, to  

## [Slide 10.5] bpftime: userspace eBPF extension framework

But ensuring eBPF compatibility and efficiency presented a major challenge. The Linux eBPF ecosystem consists of tightly coupled components, like compilers, runtime libraries, and the kernel, they are nearly impossible to disentangle. Prior user-level eBPF systems tried re-implementing the entire eBPF technology stack and ultimately failed to provide reasonable performance and compatibility.

Instead, bpftime takes a different approach. We identify a narrow waist in the current eBPF ecosystem and interpose at that point. Specifically, we intercept eBPF-related system call, and use the shared map mechanism for data sharing between extensions, applications, and kernel eBPF. This lets us reuse the proven eBPF ecosystem while adding just the minimal new components needed for userspace deployment.

So, Here's how bpftime works at a high level. We intercept eBPF application system calls before they reach the kernel. Our loader converts EIM policies into bytecode assertions and feeds everything through the kernel's proven eBPF verifier for safety guarantees. After JIT compilation to native code, we use dynamic binary rewriting to patch trampolines into the target application only when extensions are actually loaded. At runtime, we use memory protection keys to protect and execute the extension.


## [Slide 10.75] transition to real-world use cases

Now, let me show you how bpftime works in practice, and our evaluation results.

## [Slide 11] Real-World Use Cases

bpftime is open source on GitHub with an active community, and these applications demonstrate both the versatility of our approach and real-world user adoption.
... we design the 6 useca
we built six real-world applications. For security, we created an Nginx firewall that blocks malicious URLs in real time. For reliability, we built a Redis extension that bridges the durability gap between losing thousands of writes versus taking a 6× performance hit. For performance, we accelerated FUSE file operations with in-process caching. For observability, we ported existing tools like DeepFlow, syscount, and sslsniff to demonstrate seamless eBPF compatibility.

For more detail, you can check the paper.... because of time limit, I will not go into the details...

we will only look at ....

## [Slide 12] Customization: Nginx Firewall

Let me show you the performance impact for customization purposes plugins. For our Nginx firewall, we compared different extension approaches under a realistic workload. In this diagram, the more to the top, the higher throughput, the better. Lua and WebAssembly extensions impose 11–12 percent throughput loss—that's significant overhead that many operators can't accept in production. Our bpftime implementation achieves the same security functionality with only 2 percent overhead. That's a 5× to 6× improvement over existing approaches.

> add lua, wasm not proviing safety/interconnectedness trade-offs...

> add a axis for better direction.
> y axis request per second 
> put bpftime to the right

> label and group them no provideing xxx, 

## [Slide 13] Performance Results: SSL Monitoring

Beyond web servers, let's look at observability. Consider sslsniff, which monitors encrypted TLS traffic—crucial for debugging production microservices. 
> we run sslsniff on nginx deployment...
> have more reference for what the things are
> label the y axis...
 The figure shows a clear performance comparison across different data sizes from 1K to 256K bytes. Also, the more to the top, the better. With kernel eBPF, this monitoring costs 28 percent throughput loss. That's prohibitive for production use. With bpftime, the same monitoring functionality costs only 7 percent overhead. 

> add a axis for better direction.
> only have largest and smallest.

## [Slide 14] Take-Aways (On Outline, not separate slide)

Let me close with three key takeaways. First, EIM provides the missing piece for fine-grained extension control—you can now specify precise least-privilege policies per extension entry without touching application source code. Second, bpftime shows that you don't have to choose between safety and performance. We achieve safety with near-native performance using offline verification, hardware isolation, and concealed trampolines. Third, maintaining eBPF compatibility means you can adopt our approach immediately without changing your existing workflows.
> 

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

- **Extensions are everywhere:** PostgreSQL (PostGIS, TimescaleDB), VS Code (language servers, debuggers, GitLens), Nginx (modules), Redis (Lua scripts), browsers (ad blockers, developer tools)

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

**No single approach meets all requirements:**

- **Dynamic loading**: fast but no isolation or policy enforcement
- **Software Fault Isolation (e.g., WebAssembly)**: safety with 10–15% performance penalty
- **Subprocess isolation**: strong separation but high IPC overhead  
- **Kernel eBPF uprobes**: isolation at microsecond-level trap cost

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

**System administrator writes simple policies that grant minimal privileges to each extension:**

```yaml
observeProcessBegin:
  entry: "processBegin"
  allowed: [readPid, nginxTime, read(request)]

updateResponse:
  entry: "updateResponseContent"  
  allowed: [read(request), write(response)]
```

**Benefits:**
- Policies live completely outside application code
- Refine security settings in production **without recompiling**
- True least-privilege deployment

---

**Slide 10: bpftime: userspace eBPF extension framework**

**Why eBPF?**
- **Proven safety** through verification
- **Rich ecosystem** we can reuse
- **Compatibility** - existing eBPF tools work immediately

**The eBPF compatibility challenge:**
- Linux eBPF has tightly coupled components (compilers, runtime, kernel)
- Prior user eBPF failed by re-implementing entire stack
- **bpftime solution**: Interpose on eBPF syscalls only

**High-level approach:**
- Intercept eBPF syscalls before kernel
- Convert EIM policies into bytecode assertions
- Use kernel's proven eBPF verifier for safety
- JIT compile to native code
- Binary rewriting for trampolines only when needed
- MPK for fast security domain switching

**Key efficiency techniques:**
1. **Offline eBPF verification** for zero runtime safety checks
2. **Intel Memory Protection Keys** for fast domain switching
3. **Concealed extension entries** that eliminate overhead for unused hooks

[use the figure here]

---

**Slide 11: Real-World Use Cases**

**Six applications demonstrate breadth:**

**Security**: Nginx firewall (malicious URL blocking)
**Reliability**: Redis durability tuner (bridge everysec/alwayson gap)
**Performance**: FUSE metadata cache (in-process acceleration)
**Observability**: DeepFlow, syscount, sslsniff (eBPF compatibility)

---

**Slide 12: Performance Results - Nginx Firewall**

**Workload**: 8 threads, 256 connections, realistic traffic

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

**Slide 14: Take-Aways**

**Key takeaways:**
1. **EIM** → Fine-grained least-privilege policies without source changes
2. **bpftime** → Don't choose between safety and performance 

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


<!-- 
Let me explain this using our Nginx example. In the extension ecosystem, we have 2 key roles. First, Nginx application developers write the core web server code. Second, extension developers create plugins like firewalls, load balancers, and monitoring tools. Third, the extension manager—typically a system administrator or DevOps engineer—decides which extensions to deploy and what privileges each should have. Finally, end users send HTTP requests that trigger both the host application and extensions. -->