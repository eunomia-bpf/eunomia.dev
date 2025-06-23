# Speech Script for OSDI 2025 bpftime talk

≈ 2 250 words, ~15 minutes

## [Slide 0]

Good morning, everyone. Thank you for joining our session. My name is **Yusheng Zheng** from UC Santa Cruz. Today I will present our OSDI '25 paper titled **"Extending Applications Safely and Efficiently."**

## [Slide 0.5] Roadmap

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

| Approach | Safety | Isolation | Efficiency | Fine-Grained Control |
|----------|---------|-----------|------------|---------------------|
| Native Loading | ✗ | ✗ | ✓ | ✗ |
| SFI (Wasm, Lua) | Limited | ✓ | ✗ (10-15% overhead) | ✗ |
| Subprocess | ✓ | ✓ | ✗ (context switches) | Limited |
| eBPF uprobes | ✓ | ✓ | ✗ (kernel traps) | Limited |

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

## [Slide 5] EIM Development-Time Specification

During development time, application developers annotate their code to declare what capabilities extensions could possibly use. For our Nginx example, developers might declare a state capability called `readPid` for accessing the process ID, a function capability `nginxTime()` with postconditions ensuring positive return values, and extension entries like `processBegin` at request start and `updateResponseContent` before sending responses.

These annotations are extracted through static analysis and compiled into a binary-embedded manifest. This happens once during development and covers the full universe of possible extension behaviors the application could support.

## [Slide 6] EIM Deployment-Time Specification

At deployment time, the trusted extension manager writes human-readable policies that create extension classes mapping entries to minimal capability sets. For instance, an "observeProcessBegin" class might allow infinite instructions, `readPid`, `nginxTime()`, and reading request arguments, but no writes. A separate "updateResponse" class allows both reading and writing request data.

Crucially, these policies live outside the host application, so managers can refine privileges in production without changing binaries. This enables fine-grained least-privilege policies per extension entry without touching application source code.

## [Slide 7] EIM Summary

To summarize, EIM enables fine-grained extension control through two key innovations. First, it models all extension capabilities—from memory access patterns to function calls—as named resources that can be precisely granted or denied. Second, it separates concerns between development time (where application developers declare possible capabilities) and deployment time (where managers craft least-privilege policies).

This approach solves a critical gap: existing frameworks either provide no fine-grained control (like native loading) or bundle capabilities into coarse-grained program types (like eBPF's networking vs. tracing classifications). EIM allows managers to grant an observability extension only read access to specific variables and approved logging functions, while giving a firewall extension both read and write permissions for request modification—all without touching application source code.

The key insight is treating interconnectedness and safety as orthogonal dimensions that can be balanced precisely for each extension entry point, enabling true least-privilege deployment in production environments.


## [Slide 7] bpftime: Why We Need a New Runtime

While EIM specifications are platform-independent, existing frameworks cannot efficiently enforce them. Software fault isolation adds runtime overhead. Subprocess isolation requires expensive context switches. Kernel eBPF uprobes trap on every invocation.

We built bpftime to efficiently enforce EIM while maintaining eBPF compatibility. This allows existing eBPF tools to seamlessly adopt bpftime, and enables extensions to share state with kernel eBPF programs for comprehensive observability use cases spanning both kernel and userspace.

## [Slide 8] bpftime Architecture

Our architecture interposes at the narrow waist of the eBPF ecosystem. The loader intercepts `bpf()` syscalls, converts EIM policies into bytecode assertions, and uses the kernel's eBPF verifier for safety proofs. After JIT compilation, the binary rewriter patches trampolines at extension entries only when extensions are loaded. The user-runtime switches memory protection domains and executes native code, while bpftime maps eliminate syscalls for map operations.

This design reuses the proven eBPF verifier and toolchain while adding the minimal components needed for userspace extension deployment with EIM enforcement.
## [Slide 9] bpftime: Key Techniques and Challenges

Now let me explain the core challenge that bpftime solves and how we achieve our performance advantages. Current extension frameworks face a fundamental three-way tension between safety, isolation, and efficiency that existing solutions cannot resolve simultaneously.

Extension safety requires preventing extension failures from harming the host application by restricting system resources and host interactions, but this often conflicts with the interconnectedness that extensions need to be useful. Extension isolation means protecting extensions from host application interference, which is essential for security monitoring, but traditional OS-level abstractions require expensive context switches. Extension efficiency demands near-native speed execution, yet current frameworks impose significant overhead through heavyweight isolation or software fault isolation techniques.

The core problem is that existing frameworks make painful trade-offs. Process-based isolation is safe but slow due to context switching costs. Software fault isolation like WebAssembly or NaCl sacrifices 10-15% performance for runtime safety checks. Kernel eBPF uprobes trap on every function call, adding microseconds of overhead that kills performance on hot paths.

bpftime resolves this fundamental tension through three synergistic techniques that work together to achieve kernel-grade safety with library-grade performance. First, offline verification uses the eBPF verifier at load time with EIM constraints, guaranteeing safety upfront so the hot path carries no extra runtime checks whatsoever. Second, Intel Memory Protection Keys enable intra-process isolation with just two fast WRPKRU instructions, completely avoiding expensive mprotect calls or context switches. Third, concealed extension entries dynamically erase unused trampolines entirely, adding zero overhead for dormant hooks while active extensions pay only 84 nanoseconds for domain switching. Together, these techniques deliver the efficiency of native code with the safety guarantees of sandboxed execution.

## [Slide 10] Real-World Use Cases

We validated our design with six real-world use cases. An inline Nginx firewall filters malicious URLs at line rate. A Redis durability tuner batches fsync calls to bridge the gap between "everysec" and "alwayson" configurations. A FUSE metadata cache accelerates repeated stat calls with in-process lookups. DeepFlow provides distributed tracing for microservices. Syscount profiles syscalls per-process. Sslsniff decrypts TLS traffic for end-to-end observability. Each exercises different requirements around throughput, latency, and policy control.

## [Slide 11] Performance Results: Customization

For throughput-critical scenarios, bpftime significantly outperforms alternatives. Our Nginx firewall implemented with bpftime incurs only 2 percent overhead compared to 11–12 percent for Lua and WebAssembly—a 5×–6× improvement. Redis durability tuning with our delayed-fsync extension achieves 65,000 requests per second, five times faster than "alwayson" while risking at most two lost writes compared to tens of thousands with "everysec". FUSE metadata caching accelerates 100,000 stat calls from 3.65 seconds to 0.176 seconds—a 20× speedup.

## [Slide 12] Performance Results: Observability

For observability workloads, bpftime substantially reduces monitoring overhead. DeepFlow tracing with kernel uprobes cuts throughput by 54 percent on large responses; bpftime shrinks that to 20 percent. Sslsniff TLS interception costs 28 percent under kernel eBPF but only 7 percent with bpftime, making encrypted tracing feasible in performance-sensitive servers. Syscount traditionally imposes 9–10 percent machine-wide overhead; bpftime confines impact to target processes, enabling fair multi-tenant monitoring.

Our microbenchmarks show that uprobe dispatch takes 190 nanoseconds versus 2.5 microseconds for kernel eBPF—a 14× improvement. Map operations run 2× faster since they operate on in-process data structures rather than kernel objects.

## [Slide 13] Take-Aways and Future Work

Three key takeaways from our work. First, EIM enables extension managers to specify precise, least-privilege resource policies at individual entry granularity without modifying application source code. Second, bpftime enforces these policies efficiently using offline eBPF verification, Intel MPK domain switching, and dynamic trampolines—delivering kernel-grade safety with library-grade performance. Third, we maintain 100 percent eBPF compatibility, allowing seamless adoption of existing tools and workflows.

Looking ahead, we're extending bpftime to support GPU and ML workloads, broadening the scope of safe, efficient userspace extension deployment.

## [Slide 14] Thank You & Questions

Thank you for your attention. **bpftime** is open-source under the MIT license at **github.com/eunomia-bpf/bpftime**. We welcome your issues, pull requests, and collaboration. I'm happy to take your questions.

## **Complete Slide Deck (16 slides, 16:9)**

---

**Slide**
**Extending Applications Safely & Efficiently**
Yusheng Zheng¹ • Tong Yu² • Yiwei Yang¹ • Yanpeng Hu³
Xiaozheng Lai⁴ • Dan Williams⁵ • Andi Quinn¹
¹UC Santa Cruz   ²eunomia-bpf Community   ³ShanghaiTech University
⁴South China University of Technology   ⁵Virginia Tech

---

**Slide**
**Extensions Are Everywhere**

- **Web browsers**: AdBlock, password managers, developer tools
- **HTTP servers**: Nginx Lua scripts, Apache modules, WebAssembly plugins  
- **Databases**: PostgreSQL functions, MySQL plugins, custom operators
- **Text editors**: Vim/VS Code extensions, language servers, formatters
- **Operating systems**: eBPF programs for monitoring, networking, security

**Common theme**: Customize software without modifying application source code

> There are too many examples. you should illustrate what are the extensions/plugins? Why we want them?

you can reorganize it as: 

- Popular / high level exampl​e of extensions and plugins
- What is extension?
- Why we care about extensions?

Slide

---

**Slide**
**Nginx plugin example**

!\[Figure 1 from paper]

a simple example can go first​ and help people understand the extension and plugin background.

3 min to understand the background of it. no need to introduce role here, introduce role in EIM part

=== some text for reference

System extensions augment an application without modifying
its source code to customize behavior, enhance security, add
custom features, and observe behavior. By supporting appli-
cation modifications without requiring source code changes,
extensions allow a customized deployment to integrate main-
tenance updates from upstream repositories easily and can
provide assurances of security and safety. The rest of this sec-
tion discusses the principal roles of system extensions (2.1),
provides an example web-server use-case (2.2), articulates
the key properties of extension frameworks (2.3), discusses
limitations of the current state-of-the-art (2.4), and articulates
the threat model (2.5).
2.1 Roles
The system extension usage model considers four key prin-
cipals. The application developers are a group of trusted
developers who write the original application, while the ex-
tension developers are a group of trusted developers who
create the extensions. System extensions assume that both
the application developer(s) and extension developer(s) are
trusted but fallible, so applications and extensions might be
exploitable but are not intentionally malicious. Next, the sys-
tem extension model includes an extension manager, a trusted
individual that installs and manages the extensions; the model
relies on the manager to be both trusted and infallible. Finally,
users are untrusted individuals who interact with the extended
application; users can be malicious and may try to craft inputs
that would trigger vulnerabilities in otherwise benign code.
Figure 1 provides a representation of an extended appli-
cation and shows the role of each principal. The application
developers write the host application. The extension devel-
oper creates the extension program, which can read and write
application state and execute application-defined functions.
The extension manager is responsible for deciding which ex-
tensions to use at each extension entry. Finally, users produce
input that interacts with the host application and, indirectly,
the extension program.
2.2 Web-Server Example
Consider an instance of Nginx deployed as a reverse proxy.
The application developers write the server, while the ex-
tension developers provide a suite of possible extensions to
deploy on the system for monitoring, firewalls, and load bal-
ancing. The extension manager determines the extensions for
the deployment and the privileges to provide each extension.
First, the manager uses an extension program that monitors
traffic to detect reliability issues [27]. Second, the manager
deploys an extension program that implements a firewall that
returns a 404 response for URLs that are indicative of SQL
injection and cross-site scripting attack. Finally, the man-
ager deploys an extension program to perform load balancing
across the possible servers downstream from the proxy by pe-
riodically contacting downstream servers to measure system
load [26]

=== you should simplify and make this into 3 mins talk

---


**Slide**
**Extensions Have Serious Problems**

- Real-world extension safety violations:

| Bug                  | Software   | Summary                                                |
| -------------------- | ---------- | ------------------------------------------------------ |
| Bilibili \[73]       | Nginx      | Livelock in a Lua extension caused a multi-hour outage |
| CVE-2021-44790 \[47] | Apache Lua | Buffer overflow crashed httpd under load               |
| CVE-2024-31449 \[42] | Redis Lua  | Stack overflow enabled remote code execution           |

(No need to have a full table.)

- **Performance penalty**: Wasm/language sandboxes impose 10–15% overhead
- **Painful tension**: Safety vs. Extensibility vs. Performance

[merge this slide and the next slide, merge the issues and key requirements]

**Key Extension Framework Features**

> short version
>
> 1. fine-grained safety/interconnectedness trade-offs
> 2. isolation
> 3. efficiency



1. **Fine-Grained Safety/Interconnectedness Trade-offs**
   – Extensions must be interconnected (read/write host state, call host functions)
   – Extensions must be safe (follow principle of least privilege)
   – Manager configures per-extension capabilities without changing host application
2. **Isolation**
   – Extensions protected from buggy or compromised host applications
   – Prevents attackers from circumventing extension-based security via host exploits
3. **Efficiency**
   – Near-native speed execution on production hot paths
   – Critical for per-request, per-operation extension deployments

> merge it with previous one.

---

**Slide 5**
**State-of-the-Art Falls Short**

* **Native Loading (LD\_PRELOAD, DBI)**
  Fast, but offers no isolation or least-privilege controls.
* **Software Fault Isolation (NaCl, WebAssembly, RLBox)**
  Enforces safety via runtime checks, incur 10–15 % overhead.
* **Subprocess Isolation (lwC, Wedge, Orbit)**
  Context-switch costs are prohibitive for low-latency paths.
* **Kernel-space eBPF Uprobes**
  Isolated, but software breakpoints trap into the kernel on every hit.
* **Aspect-Oriented Languages**
  No built-in model for specifying or enforcing per-entry policies.

[tight this up with a table to make it more concise]


**Slide**
**Contribution**

1. **Motivation** → Extension safety vs. performance
2. **EIM + bpftime** → Our two-part solution  
3. **Use Cases** → Six real-world applications
4. **Evaluation** → 5-14× performance improvements

> move the  contribution after the motivation
> Like roadmap​
> Say what is eim and what is bpftime​
> See the slide and understand the struct​
> Roadmap can be after​

Roadmap is a signal for back to abstractions


**Slide 5.5**
**EIM: Extension Interface Model**

- Capabilities as Resources​
  - State access (e.g., read/write variables)​
  - Function calls (with pre/post-conditions)​
  - Hardware resources (instructions, memory)​

- Two-Phase Specification​
  - Development-Time (by Developer)​
  - Deployment-Time (by Manager)

> Can cite something here​
> modify it, Reintroduce it using nginx example, make everything around nginx

---

**Slide**
**EIM: Development-Time Specification**

- Developed by application developer
- Annotations in source code or YAML configs


*(Figure 2: Example development-time EIM for Nginx observability)*

> a picture, no need to detail​, Show they are short and what people are doing on it. Tell people to look at paper and we are not talking about some detail​. make it around the nginx example


---

**Slide**
**EIM: Deployment-Time Specification**

- Developed by extension developer or manager
- YAML configs to explore the interconnectedness/safety trade-offs

> make it around the nginx example.


*(Figure 3: Example deployment-time EIM for Nginx observability)*

**Slide**
> add a slide tosummary of eim

>  Do need some abstract concept here. to summary the eim.

**Slide**
**High level overview​ for bpftime**

> add a slide 

- Motivate bpftime​: Some tools that can use eim, but they fail short​
- Why bpftime​
- challenge





---

**Slide**
**bpftime Architecture at a Glance**

- Compatible with eBPF
- Verification for safety and efficiency

!\[Figure 4 from paper]

* **Loader** intercepts `bpf()` syscalls; enforces EIM via kernel's verifier + extra assertions.
* **Binary Rewriter** inserts trampolines at extension entries only when needed.
* **User-Runtime** in target process flips MPK keys and executes JIT-compiled code.
* **bpftime Maps** mirror eBPF maps in user space to avoid syscalls.

> A little too complex ​ in the figture. Replace with High level summay

---

**Slide 9**
**Efficient Safety & Isolation**

* **Offline Verification**
  All pointer and type checks resolved at load time by eBPF verifier ⇒ zero runtime checks.
* **Intel MPK Isolation**
  Two instructions (`WRPKRU`) switch domains in ≈ 80 ns, no syscalls.
* **Concealed Entries**
  Unused trampolines erased at load time ⇒ dormant cost per hook only ≈ 1.3 ns.

> Replace the name and content  with key tech and challenge 1. mpk/ verify 2. hook

---

**Slide 11**
**Six Real-World Use Cases**

* **Nginx Firewall**: A lightweight request filtering and logging Nginx module.
* **Redis Durability**: An extension that bridges Redis's durability gap between "everysec" and "alwayson" configs.
* **FUSE Metadata Cache**: A caching mechanism that accelerates file system metadata operations.
* **DeepFlow**: A distributed tracing agent that instruments both kernel and user APIs.
* **syscount**: A per-process syscall analysis eBPF tool from bcc.
* **sslsniff**: A TLS traffic analysis eBPF tool from bcc.

> a little less detail here.

---

**Slide 12**
**Customization: Nginx, Redis, FUSE**

* **Nginx (Figure 6)**
  - Lua/Wasm: 11–12 % throughput loss
  - bpftime: **2 %** loss ⇒ 5–6× improvement

> around nginx, and show the figture here 

---

**Observability: DeepFlow, syscount, sslsniff**

* **sslsniff (Figure 9)**
  - kernel eBPF: 28 % overhead
  - bpftime: **7 %** overhead

make it around sslsniff

---

**Slide 13**
**Micro-Benchmark**

Compare with eBPF:

* **Uprobe Dispatch**: 2.56 µs → 190 ns (14× faster)
* **Syscall Tracepoint**: 151 ns → 232 ns (1.5× slower)
* **Memory access** (Table 3): user-space read/write <2 ns vs 23 ns (10× faster)
* **Overall**: average 1.5× faster than ubpf/rbpf (Figure 11)

> Remove microbench ​and Take it as one sentence inline into design to say why it's much more faster

---

**Slide 14**
**Take-Aways**

* **EIM** gives per-entry least-privilege policies without host changes.
* **bpftime** enforces EIM via eBPF verifier, MPK, and concealed trampolines.
* **eBPF-ABI compatible**: run eBPF applications seamlessly without recompilation and modification.

**Roadmap**

- GPU and ML workloadssupport
- Keep compatibility with eBPF upstream

get started: you can run bpftime just as ebpf pogram and please visit the github page for more details.

---

**Slide 15**
**Thank You & Q \&A**

**GitHub:** github.com/eunomia-bpf/bpftime (MIT license)
We welcome issues, pull requests, and collaboration.

---

---
