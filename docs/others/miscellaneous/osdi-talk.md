# Speech Script for OSDI 2025 bpftime talk

≈ 2 250 words, ~15 minutes

## [Slide 0]

Good morning, everyone. Thank you for joining our session. My name is **Yusheng Zheng** from UC santa cruz. Today I will present our OSDI ’25 paper titled **“Extending Applications Safely and Efficiently.”** 

## [Slide 0.5] outline of the talk

In this talk, I will describe our new **Extension Interface Model**, or EIM, and **bpftime**, a userspace extension framework. Together, they enable extension code to run inside production applications with **kernel-grade safety**, **per-entry least-privilege policies**, and **near-native performance**, all without requiring kernel patches or restructuring existing toolchains.

## [Slide 1]

Extensions are everywhere in modern software systems. Web browsers use extensions like AdBlock and password managers. HTTP servers like Nginx and Apache support Lua scripts and WebAssembly modules. Databases like PostgreSQL and MySQL enable custom functions and plugins. Text editors from Vim to VS Code rely heavily on extensions for functionality. Even operating systems use extensions—eBPF programs extend kernel capabilities for monitoring, networking, and security. The common theme is that extensions allow software to be customized without modifying the original application source code.

## [Slide 2]

To clarify the problem, Figure 1 depicts the four principal actors in any extension deployment. First, **application developers** build and ship the host application, instrumenting it with named extension entries that mark safe hook points. Second, **extension developers** write modules targeting those entries, potentially invoking host-provided functions or reading and writing application state. Third, the trusted **extension manager**—often a DevOps or security engineer—selects which modules to deploy, configures their privileges, and oversees policy updates. Finally, **end users** generate traffic or inputs that drive both the host and extension code. A robust framework must empower the extension manager to set per-entry policies, must protect the extension’s integrity even if the host misbehaves, and must introduce zero or near-zero steady-state overhead on the critical path.

## [Slide 2.5]

However, these extension systems face serious safety and performance challenges. Let me show you some real-world incidents. In 2023, a malformed Lua plug-in accidentally created an infinite loop inside Nginx, causing Bilibili's entire content delivery network to go down for hours. Soon thereafter, Apache's Lua module suffered a buffer-overflow bug that silently crashed httpd under a heavy workload. Early this year, a stack overflow in a Redis Lua script was weaponized to achieve arbitrary remote code execution. These examples, summarized in Table 1, show that even mature plug-in ecosystems can introduce vulnerabilities that bring production systems to a halt. Equally important, operators often disable Wasm or language-based sandboxes in production due to their persistent **10–15 percent** throughput tax on the hot path. In short, we still face a painful tension between safety, extensibility, and performance.

## [Slide 3]
Extension use-cases require three key framework features. The first is **fine-grained safety/interconnectedness trade-offs**. Extensions must be interconnected—able to interact with the host application by reading/writing host state and executing host-defined functions. However, extension managers wish to ensure extensions are safe and cannot harm deployment reliability or security. Since safety and interconnectedness are in tension, managers should follow the principle of least privilege, allowing each extension to perform only the actions necessary for its specific task. For example, Nginx observability extensions only need read access to specific host states, while firewall extensions need read/write access to different host states. The second requirement is **isolation**: extensions must be protected from harm by the host application. This ensures that attackers cannot circumvent extension-based security by exploiting bugs in the host. The third requirement is **efficiency**: extensions should execute at near-native speed since they may be deployed on the hot path of production systems, such as per HTTP request processing.

## [Slide 4]

Unfortunately, existing approaches cannot satisfy all three simultaneously. Mechanisms based on **dynamic loading**—for example, LD\_PRELOAD or binary instrumentation—achieve great speed but provide neither isolation nor fine-grained policies. **Software Fault Isolation** systems, including Native Client, Lua sandboxes, or WebAssembly runtimes, do deliver safety, but they do so by inserting runtime instrumentation that carries a **10–15 percent** performance penalty. **Subprocess isolation** frameworks such as lwC, Wedge, or Orbit ensure strong separation at the OS level, but the overhead of process boundaries and IPC is untenable on low-latency paths. **Kernel eBPF uprobes** offer isolation and moderately rich policies, but they trap into the kernel on every invocation, costing several microseconds each time. And **aspect-oriented programming** tools have no built-in model for per-entry privilege constraints. The bottom line is that no single framework today can express extension-specific, per-entry policies, isolate code from host faults, and still perform like native code.

## [Slide 5]

Our first contribution, the **Extension Interface Model**, or **EIM**, addresses the policy gap. EIM treats every extension capability—be it concrete, like memory allocation or CPU instructions, or abstract, like calling a host function or reading a global variable—as a **resource**. During **development time**, the application developer annotates the host code to declare precisely which resources each extension entry could possibly consume. For example, as shown in Figure 2, a developer might declare a **state capability** named `readPid` that allows reading the global `ngx_pid` field, a **function capability** named `nginxTime()` with a post-condition ensuring a positive return value, and two **extension entries**—`processBegin` at the start of request processing, and `updateResponseContent` right before sending the response. These annotations are pulled from C attributes or kfunc annotations, fed into a static analysis tool that extracts symbol and DWARF debug information, and compiled into a binary-embedded manifest representing the full set of possible capabilities.

## [Slide 6]

Then, at **deployment time**, the trusted extension manager writes a small, human-readable **deployment-time EIM**. As shown in Figure 3, the manager creates one or more **extension classes** per entry point that bundle the exact capabilities to grant. For instance, an “observeProcessBegin” class might allow infinite instructions, `readPid`, `nginxTime()`, and reading the `Request *r` argument, but no writes or additional helpers. A separate “updateResponse” class might allow both `read(r)` and `write(r)` to modify response buffers. Crucially, these policies live outside the host code, so the manager can refine or revoke privileges in production without changing application binaries.

## [Slide 7]

Having defined a flexible, fine-grained policy model, we built **bpftime**, a new extension **runtime** that enforces EIM and delivers native-class performance, and keep it compatible with eBPF.

Figure 4 shows the high-level architecture, White components are from eBPF; orange components are new to bpftime. Blue arrows show execution flow when compiling and loading an eBPF application. Green arrows show execution flow when an eBPF extension executes. White arrows with black outline indicate components that interact with eBPF maps

We start with the standard Linux eBPF toolchain—compiler, loader, verifier, and JIT—shown in white. We then insert three orange components: a **loader** in userspace that intercepts `bpf()` syscalls, feeds the bytecode through the kernel’s eBPF verifier with extra EIM-derived assertions, and JIT-compiles the result; a **binary rewriter** that patches five-byte trampolines at each extension entry only when an extension is loaded; and a lean **user-runtime** inside the target process that flips memory protection keys and executes the JIT-compiled native code. We also provide **bpftime maps**, user-space equivalents of eBPF maps, to eliminate syscalls for map operations.

## [Slide 8]

The **loader and runtime workflow** is as follows. First, the bpftime loader intercepts the standard `bpf()` syscalls from libbpf or bcc and parses the embedded EIM manifests alongside DWARF/BTF information. It converts the deployment-time policy into bytecode assertions, then invokes the kernel’s eBPF verifier to prove compliance. Next, it JIT-compiles the verified bytecode to native x86, attaches the resulting code and data into the target process via `ptrace` and helper libraries like Frida and Capstone, and writes trampolines at the designated entry offsets. At runtime, when the host process executes one of those patched instructions, it jumps into the user-runtime, executes two `WRPKRU` instructions to switch to the extension domain, jumps to the extension code, and upon return resets the key and continues execution. Meanwhile, any shared state lives in bpftime maps in user space, avoiding repeated kernel traps.


## [Slide 9]

bpftime’s performance advantage rests on three synergistic techniques. First, we leverage **offline verification**: by using the eBPF verifier at load time with added EIM constraints, we guarantee pointer safety, type safety, and resource limits upfront—so the hot path carries no extra checks. Second, we employ **Intel Memory Protection Keys (MPK)** for **intra-process isolation**. A single pair of `WRPKRU` instructions flips the protection domain, obviating expensive `mprotect` calls or context switches. Third, we introduce **concealed extension entries**: if no extension attaches to a given hook, our rewriter erases the trampoline entirely, no overhead for per potential hook. In combination, these techniques yield near-native performance even under heavy load.

## [Slide 10]

We validated our design with **six** real-world use cases drawn from observability, security, and performance tuning. First, an **inline firewall** module for Nginx that filters malicious URLs at line rate. Second, a **durability tuner** for Redis’s Append-Only File that batches fsync calls to tune the throughput-safety trade-off. Third, a **metadata cache** for FUSE that collapses repeated `stat` calls into a fast in-process lookup. Fourth, **DeepFlow**, an open-source distributed tracing platform that instruments both kernel and user APIs. Fifth, **syscount**, the classic per-process syscall profiler from the bcc toolkit. And sixth, **sslsniff**, which decrypts and logs SSL/TLS traffic in userspace for end-to-end observability. Each use case exercises a different corner of our requirement triangle—some demand throughput, some demand low latency, some demand precise policy control.

## [Slide 11]

Let us begin with **throughput-critical** scenarios. In Figure 6 we show Nginx performance under an eight-thread, 64-connection `wrk` workload. The same firewall logic implemented in Lua or WebAssembly incurs an **11–12 percent** throughput loss. Rewriting it as a bpftime plug-in governed by an EIM policy reduces that penalty to only **2 percent**, a **5×–6×** improvement. Next, in Figure 8 we explore Redis durability. Redis offers “always-on” fsync, which blocks on every write and cuts throughput sixfold, or “every-second” fsync, which is faster but risks losing tens of thousands of writes on crash. With a 20-line bpftime extension we implement **delayed-fsync**, batching at most two writes per flush and consulting a shared kernel counter to avoid redundant calls. This delivers **65 000 req/s**—five times faster than always-on—while risking at most two lost writes. Finally, Table 2 reports FUSE caching. A simple passthrough FUSE filesystem takes **3.65 s** to issue 100 000 `stat` calls; with bpftime’s metadata cache that drops to **0.176 s**, a **20×** speed-up and dramatically improved responsiveness for file-system–heavy workloads.

## [Slide 12]

Turning to **observability**, we examine DeepFlow, syscount, and sslsniff. Figure 7 plots the throughput of a TLS-protected Go microservice under DeepFlow instrumentation. Kernel uprobes alone cut throughput by **54 percent** on large responses; swapping in bpftime uprobes with identical tracing code shrinks that drop to **20 percent**. Next, syscount traditionally hooks every syscall in every process, imposing a **9–10 percent** overhead machine-wide. Figure 10 shows that bpftime confines syscount’s overhead to the target PID—other processes run at native speed, enhancing multi-tenant fairness. Finally, Figure 9 shows that sslsniff’s TLS interception costs **28 percent** under kernel eBPF; under bpftime it costs only **7 percent**, making end-to-end encrypted tracing feasible in performance-sensitive servers.

## [Slide 13]

To peel back the layers, we ran **micro-benchmarks** shown in Figure 11 and Table 3. Dispatching a user-space uprobe via kernel eBPF takes **2.5 µs**; bpftime’s trampoline and MPK switch complete in **190 ns**, a **14×** improvement. Syscall tracepoints under bpftime finish in **232 ns**, compared to **151 ns** in kernel mode—only a **1.5×** overhead for argument marshaling. Map operations like hash update, delete, and lookup run **2×** faster or better, since they operate on in-process data structures rather than kernel objects. Overall, we observe an average **1.5–1.7×** speed-up over existing userspace eBPF VMs such as ubpf and rbpf, confirming that our verifier-offline, MPK-protected design delivers near-native performance.

## [Slide 14]
I will close with three **take-aways** and a glance at the **roadmap**. First, **EIM** gives extension managers a precise, declarative policy language to grant least-privilege resource sets at the granularity of individual entry points, all without touching application source code at deployment time. Second, **bpftime** enforces those policies with three lightweight primitives—offline eBPF verification, Intel MPK for domain switching, and dynamic trampolines—yielding kernel-grade safety and library-grade speed. Third, we preserve **100 percent compatibility** with the existing eBPF syscall ABI, so you can run eBPF applications seamlessly without recompilation and modification. Looking ahead, we are also extending bpftime to support GPU and ML workloads.

## [Slide 15]
Thank you for your attention. **bpftime** is open-source under the MIT license at **github.com/eunomia-bpf/bpftime**. We welcome your issues, pull requests, and collaboration. I’m happy to take your questions.

## **Complete Slide Deck (16 slides, 16:9)**

---

**Slide 0**
**Extending Applications Safely & Efficiently**
Yusheng Zheng¹ • Tong Yu² • Yiwei Yang¹ • Yanpeng Hu³
Xiaozheng Lai⁴ • Dan Williams⁵ • Andi Quinn¹
¹UC Santa Cruz   ²eunomia-bpf Community   ³ShanghaiTech University
⁴South China University of Technology   ⁵Virginia Tech

**Slide 0.5**

Outline of the talk

---

**Slide 1**
**Extensions Are Everywhere**

- **Web browsers**: AdBlock, password managers, developer tools
- **HTTP servers**: Nginx Lua scripts, Apache modules, WebAssembly plugins  
- **Databases**: PostgreSQL functions, MySQL plugins, custom operators
- **Text editors**: Vim/VS Code extensions, language servers, formatters
- **Operating systems**: eBPF programs for monitoring, networking, security

**Common theme**: Customize software without modifying application source code

---

**Slide 2**
**Four Roles in an Extension Ecosystem**

!\[Figure 1 from paper]

– **Application Developers** write and ship the host binary with named extension entries.
– **Extension Developers** implement modules against those entries.
– **Extension Manager** chooses which module runs where and with what privileges.
– **Users** generate inputs that exercise both host and extensions.

---


**Slide 3**
**Extensions Have Serious Problems**

- Real-world extension safety violations:

| Bug                  | Software   | Summary                                                |
| -------------------- | ---------- | ------------------------------------------------------ |
| Bilibili \[73]       | Nginx      | Livelock in a Lua extension caused a multi-hour outage |
| CVE-2021-44790 \[47] | Apache Lua | Buffer overflow crashed httpd under load               |
| CVE-2024-31449 \[42] | Redis Lua  | Stack overflow enabled remote code execution           |

- **Performance penalty**: Wasm/language sandboxes impose 10–15% overhead
- **Painful tension**: Safety vs. Extensibility vs. Performance

---

**Slide 4**
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

---

**Slide 6**
**EIM: Development-Time Specification**

- Developed by application developer
- Annotations in source code or YAML configs


*(Figure 2: Example development-time EIM for Nginx observability)*

---

**Slide 7**
**EIM: Deployment-Time Specification**

- Developed by extension developer or manager
- YAML configs to explore the interconnectedness/safety trade-offs


*(Figure 3: Example deployment-time EIM for Nginx observability)*

---

**Slide 8**
**bpftime Architecture at a Glance**

!\[Figure 4 from paper]

* **Loader** intercepts `bpf()` syscalls; enforces EIM via kernel's verifier + extra assertions.
* **Binary Rewriter** inserts trampolines at extension entries only when needed.
* **User-Runtime** in target process flips MPK keys and executes JIT-compiled code.
* **bpftime Maps** mirror eBPF maps in user space to avoid syscalls.

---

**Slide 9**
**Efficient Safety & Isolation**

* **Offline Verification**
  All pointer and type checks resolved at load time by eBPF verifier ⇒ zero runtime checks.
* **Intel MPK Isolation**
  Two instructions (`WRPKRU`) switch domains in ≈ 80 ns, no syscalls.
* **Concealed Entries**
  Unused trampolines erased at load time ⇒ dormant cost per hook only ≈ 1.3 ns.

---

**Slide 10**
**Loader & Runtime Workflow**

1. **Intercept** standard eBPF syscalls from libbpf/bcc.
2. **Parse** EIM manifests and DWARF/BTF to generate constraints.
3. **Verify** byte-code via kernel's eBPF verifier with added assertions.
4. **JIT-Compile** verified byte-code into native x86.
5. **Inject** user-runtime via `ptrace` + Frida + Capstone trampolines.
6. **Execute** extension: flip MPK key → jump to code → flip back → resume.

---

**Slide 11**
**Six Real-World Use Cases**

* **Nginx Firewall**: A lightweight request filtering and logging Nginx module.
* **Redis Durability**: An extension that bridges Redis's durability gap between "everysec" and "alwayson" configs.
* **FUSE Metadata Cache**: A caching mechanism that accelerates file system metadata operations.
* **DeepFlow**: A distributed tracing agent that instruments both kernel and user APIs.
* **syscount**: A per-process syscall analysis eBPF tool from bcc.
* **sslsniff**: A TLS traffic analysis eBPF tool from bcc.

---

**Slide 12**
**Customization: Nginx, Redis, FUSE**

* **Nginx (Figure 6)**
  - Lua/Wasm: 11–12 % throughput loss
  - bpftime: **2 %** loss ⇒ 5–6× improvement
* **Redis (Figure 8)**
  - Always-on: 13 k req/s
  - Delayed-fsync: **65 k req/s**, risk ≤2 lost writes
* **FUSE (Table 2)**
  - Passthrough `fstat`: 3.65 s native → 0.176 s cached

---

**Observability: DeepFlow, syscount, sslsniff**

* **DeepFlow (Figure 7)**
  - Kernel uprobes: 54 % throughput drop
  - bpftime uprobes: **20 %** drop
* **syscount (Figure 10)**
  - kernel eBPF: 10 % slowdown on all processes
  - bpftime: **3.36 %** on target only
* **sslsniff (Figure 9)**
  - kernel eBPF: 28 % overhead
  - bpftime: **7 %** overhead

---

**Slide 13**
**Micro-Benchmark**

Compare with eBPF:

* **Uprobe Dispatch**: 2.56 µs → 190 ns (14× faster)
* **Syscall Tracepoint**: 151 ns → 232 ns (1.5× slower)
* **Memory access** (Table 3): user-space read/write <2 ns vs 23 ns (10× faster)
* **Overall**: average 1.5× faster than ubpf/rbpf (Figure 11)

---

**Slide 14**
**Take-Aways**

* **EIM** gives per-entry least-privilege policies without host changes.
* **bpftime** enforces EIM via eBPF verifier, MPK, and concealed trampolines.
* **eBPF-ABI compatible**: run eBPF applications seamlessly without recompilation and modification.

**Roadmap**

- GPU and ML workloadssupport
- Keep compatibility with eBPF upstream

---

**Slide 15**
**Thank You & Q \&A**

**GitHub:** github.com/eunomia-bpf/bpftime (MIT license)
We welcome issues, pull requests, and collaboration.

---

---
