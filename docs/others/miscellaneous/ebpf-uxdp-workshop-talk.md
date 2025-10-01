# uXDP: Frictionless XDP Deployments in Userspace (12-min talk)

**Venue:** eBPF Workshop (Coimbra, Sep 8–11, 2025)
**Speaker:** Panayiotis Gavriil (with Yusheng Zheng, Marios Kogias)

---

## Goals (for a 12‑min slot)

- Land the problem: today’s NF trade‑off (kernel eBPF vs kernel‑bypass).
- State the idea in one line: *run verified XDP programs in userspace without recompiling*.
- Show the shape of the system: control plane + runtime, DPDK & AF\_XDP modes.
- One slide of profiling insight (why these optimizations matter).
- Three slides of results: throughput, latency, ablations.
- Close with limits + what’s next.

**Pacing:** \~55–60 s/slide × 12 slides ≈ 11–12 min.

**Script length target:** 1,200–1,400 words (spoken). On‑slide text ≤30–40 words/slide.

---

## Timeline (minute‑by‑minute)

0:00 Title & hook (0:45)
0:45 What is an NF? (0:50)
1:35 Problem trade‑off (0:50)
2:25 Our idea (0:55)
3:20 Our Motivation: A Profiling Insight (0:55)
4:15 The Challenge: The Compatibility Problem (0:55)
5:10 uXDP Architecture (1:00)
6:10 How We Optimize (1:00)
7:10 Implementation highlights (0:50)
8:00 Eval setup (0:40)
8:40 Results: Throughput & Latency (1:30)
10:10 Results: Optimization Ablations (0:55)
11:05 Conclusion (0:55)
12:00 End.

---

## Slide‑by‑slide outline with speaker notes

### 1) Title & Hook

**On slide**

- uXDP: Frictionless XDP Deployments in Userspace
- Yusheng Zheng, Panayiotis Gavriil, Marios Kogias
- eBPF Workshop (Coimbra, Sep 8–11, 2025)

**Speaker notes (~90–110 words)**
Hi everyone, I’m xxx. Today I'm going to talk about uXDP. The basic idea is that we take the eBPF/XDP development model you're already familiar with, but we run the verified program in userspace. This means you don't have to maintain a separate codebase for userspace network functions. You get all the benefits you're used to, like the verifier's safety, maps, and control-plane workflows, while also opening the door to powerful userspace optimizations like JIT/AOT compilation and SIMD.  With the exact same eBPF binary, we've seen up to a **3.3x** throughput increase over in-kernel execution for simple network functions, and a **40%** boost for a complex one like Katran. In this talk, I’ll explain how this is possible, what the system looks like, and what our results show.

---

### 2) Background: Network Function (NF)

**On slide**

- A program that processes network packets at line rate.
- The "middle-boxes" of the modern network.
- Examples: Load balancers, firewalls, DDoS mitigation, gateways.
- Used by: Cloud providers, CDNs, Telcos, large enterprises.

**Speaker notes (~80–100 words)**
Before we dive into the details of uXDP, let's quickly define what a network function is. At its core, an NF is a piece of software that processes network packets as they fly by, often at very high speeds. Think of them as the specialized middle-boxes of the internet. They perform critical tasks that keep networks secure, reliable, and fast. Common examples include firewalls that block malicious traffic, load balancers that distribute requests across servers, and systems that protect against denial-of-service attacks. These are fundamental building blocks used by cloud providers, telcos, and any large-scale service.

---

### 3) Problem: today’s NF deployment trade‑off

**On slide**

Kernel eBPF(XDP)​
- safe, portable, easy ops → performance ceiling​
- E.g. lack of SIMD instructions​
Kernel‑bypass(DPDK, AF_XDP)​
- fast → ecosystem fragmentation, lacks verifier safety

**Speaker notes (~80–100 words)**
In production network functions deployment, you typically have to choose between two worlds. On one hand, you have kernel eBPF/XDP, which integrates beautifully with existing systems. You get the verifier, maps, great tooling, and easy rollouts. But running inside the kernel has performance ceilings. It constrains optimizations, so you can't use things like SIMD and LLVM features, and you have costs from interrupts and helper call overhead. On the other hand, you have kernel-bypass frameworks like DPDK or VPP. And while newer kernel features like AF_XDP make it easier to get packets to userspace, they don't solve the core problem. You still have to build the entire processing logic yourself, and without the safety net of the eBPF verifier. For example, a major outage at a streaming service company by Bilibili was traced back to an infinite loop, exactly the kind of bug the verifier is designed to prevent. The performance gap isn't trivial, especially for complex NFs like load balancers. Our goal with uXDP is to bridge this gap, combining the safety and workflows of eBPF with the performance potential of userspace.

---

### 4) Idea

**On slide**

- Run **unmodified** verified XDP in userspace
- Keep verifier, maps, control‑plane workflows
- Two modes: **DPDK** and **AF\_XDP**
- Optimizations: **inline helpers/maps**, LLVM‑IR path
- Results: up to **3.3×**; Katran **+40%**

**Speaker notes (~80–100 words)**
Our core idea is that we take a verified XDP program and move its execution to userspace, without touching the source code or bypassing any safety checks. We support two popular modes, DPDK and AF_XDP. We also introduce optimizations that are difficult to do in the kernel, like inlining common helpers and map lookups, and using an LLVM IR-based compilation path that preserves type information for better register allocation and vectorization. The same eBPF binary is still verified by the kernel; we just run a highly optimized native version in userspace.

---

### 5) Motivation: A Profiling Insight

**On slide**

- **Show Figure 1: Katran Flamegraph**
- Key takeaway: ~50% of program time is in map/helper calls.
- This is the key opportunity for optimization.

**Speaker notes (~85–100 words)**
So, what was the technical motivation for our approach? We started by profiling Katran, a complex, real-world load balancer from Meta. This flamegraph shows what we found. About two-thirds of the CPU time is spent inside the XDP program itself. And of that time, about half is spent in helper and map calls. This was our key insight; it told us that the biggest performance gains would come from reducing call overhead and exposing more of the program to the compiler's optimizer. This data is what drove our focus on aggressive, userspace-only optimizations like inlining.

---

### 6) Challenge: The Compatibility Problem

**On slide**

- Real-world NFs have: complex control planes (e.g., Katran)​
  - Maps to interact with kernel eBPF programs​
  - Multiple Syscalls​
  - Complex userspace Libraries and logic​
- Moving to userspace means rewriting all of it.

**Speaker notes (~80–90 words)**
But raw performance isn't the only problem. Moving eBPF to userspace introduces a major compatibility challenge. It's not just about running the eBPF bytecode; it's about recreating the entire environment the program expects. The biggest part of this is the **control plane**. Most real-world eBPF deployments have a complex userspace control plane for loading programs, updating maps to reflect topology changes, and reading stats. For instance, even a basic tutorial XDP program can require hundreds of system calls to manage eBPF programs and maps. Replicating this, along with the semantics of map types and helper calls, is a huge effort. When teams move to a userspace dataplane, they often have to re-implement everything from map management to the program loader library in userspace. uXDP solves this compatibility problem by preserving the original, verified eBPF program and its familiar control plane, so you don't have to rewrite those critical pieces.

---

### 7) Architecture: How it Works

**On slide**

- **Show Figure 2: Deployment Modes Diagram** (Kernel vs. uXDP DPDK vs. uXDP AF_XDP)
- Control Plane + Data Plane architecture.
- Shared memory for maps.

**Speaker notes (~95–110 words)**
Here's a high-level look at uXDP's design, which is shown in the diagram. It has two main processes: a control process that loads and verifies the eBPF program, and a data process that executes it. These processes share maps and metadata using shared memory. The runtime is built on `bpftime`, which gives us fast JIT/AOT compilation and map management. We extended it to support XDP helpers and more complex map types like LPM_TRIE and DEVMAP. As you can see, we support two main deployment modes. You can run fully in userspace with DPDK for maximum performance. Or, you can use AF_XDP, where a small XDP program in the kernel redirects packets to userspace queues. In both modes, the original eBPF program and its control plane remain completely unchanged.

---

### 8) Optimization: Inlining & LLVM IR Path

**On slide**

- **Show Figure 3: Compilation Pipeline Diagram**
- **Path 1 (Black)**: Lift from bytecode -> Inline -> Native Code
- **Path 2 (Orange)**: Use original LLVM IR -> Inline -> Better Native Code (SIMD)

**Speaker notes (~100–120 words)**
This diagram shows our two main optimization strategies. The first path, in black, works even without the original source code. We lift the verified eBPF bytecode to LLVM IR, and then we can do something the kernel can't. we aggressively inline our own IR implementations of common helpers and map lookups. This allows the compiler to see across call boundaries and generate much more efficient native code.

The second path, in orange, is even more powerful. When we have access to the original LLVM IR from the compiler, we package it with the bytecode. We still have the kernel verify the bytecode for safety, but we generate the final native code from the much richer IR. This preserves type information and data layout, leading to better register allocation and unlocking advanced optimizations like SIMD.

---

### 9) Implementation

**On slide**

- ~3.9K LOC runtime/loader (C/C++), Python tools, IR libs
- bpftime extensions: XDP helpers, map types, bpf\_link
- CO‑RE/BTF to fix xdp\_md pointer width

**Speaker notes (~80–95 words)**
Just a few implementation details. We extended the `bpftime` runtime to support attaching XDP programs via `bpf_link` and to handle the full set of XDP helpers and various map types. One small but critical detail was handling pointer widths. The `xdp_md` struct uses 32-bit offsets in the kernel, but userspace needs 64-bit pointers. We use CO-RE and a custom BTF definition to bridge this gap, allowing the same bytecode to run correctly in both environments. The loader handles verification, IR processing, and JIT compilation, while maps are placed in shared memory so the control plane can access them just like it always does.

---

### 10) Evaluation

**On slide**

- **Key Questions:**
  - Is there any development cost in porting existing XDP programs to uXDP?
  - What is the performance benefit of running XDP in userspace on DPDK or AF_XDP?
  - What is the impact of the proposed optimization techniques on the performance of eBPF XDP-based NFs?
- **Setup:** 2x 100G NICs, Xeon CPU, connected back-to-back
- **Workloads:** Linux samples + Katran + open-source NFs

**Speaker notes (~90–100 words)**
Our evaluation seeks to answer three key questions, taken directly from our paper. First: Is there any development cost in porting existing XDP programs to uXDP? Second: What is the performance benefit of running XDP in userspace on DPDK or AF_XDP? And third: What is the impact of the proposed optimization techniques on the performance of eBPF XDP-based NFs? To answer these, we used a testbed with 100-gigabit NICs to test workloads from simple packet forwarders to Katran. The next slides will detail our findings.

---

### 11) Results: Throughput & Latency

**On slide**

- **Show Figure 4 (Throughput) & 5 (Latency)**
- **Throughput:** DPDK is fastest; AF_XDP > Kernel for complex NFs.
- **Latency:** DPDK is lowest; AF_XDP is reasonable.
- Katran: **+40%**; Simple NFs: up to **3.3x**.

**Speaker notes (~110–120 words)**
Our main results are shown here. The graph on the left shows throughput. As expected, DPDK is the top performer across all workloads. But the key finding is that for complex NFs like Katran, our AF_XDP mode is faster than the native kernel driver. This is significant because it shows our userspace optimizations can overcome the cost of the kernel-userspace transition. We see a 40% gain for Katran and up to 3.3x for simpler functions.

The graph on the right shows unloaded latency for a simple echo NF. Again, DPDK is the lowest. AF_XDP has slightly higher latency than the kernel, but it remains in a very reasonable range, making it a viable option for latency-sensitive workloads that also demand high throughput.

---

### 12) Results: Optimization Ablations

**On slide**

- **Show Figure 6 (Ablations)**
- Where does the speedup come from?
- Inlining + LLVM IR path are key.
- Together: **+83%** boost for Katran over baseline AOT.

**Speaker notes (~90–100 words)**
So where does this performance boost actually come from? Our ablation study on Katran, shown here, breaks it down. We started with a baseline AOT-compiled version, which already beats the kernel, and then enabled our optimizations one by one. The results are clear: the combination of inlining helper functions and compiling from the original LLVM IR are the biggest contributors to the speedup. Together, these two techniques gave us an 83% throughput increase over the baseline. This confirms that our core optimization strategy is effective and is the key to unlocking this new level of performance.

---

### 13) Conclusion

**On slide**

- **Problem:** NFs face a trade-off: kernel safety & ecosystem vs. userspace speed.
- **uXDP:** Runs verified XDP in userspace, no code changes.
- **Combines:** eBPF's safety & workflows with userspace performance (DPDK/AF_XDP).
- **Results:** Up to 3.3x speedup; +40% for Katran.

**Speaker notes (~90–110 words)**
To summarize, today's network functions force a difficult choice between the safety and ecosystem of kernel eBPF and the raw performance of userspace frameworks. uXDP bridges this gap. We take your existing, verified XDP programs and run them in userspace without any code changes. This approach allows you to keep the entire eBPF control plane and safety guarantees you rely on, while unlocking significant performance gains through userspace-specific optimizations like JIT compilation and aggressive inlining. We've shown this can boost throughput by up to 3.3x for simple functions and provides a 40% improvement for a complex load balancer like Katran. The key takeaway is that with uXDP, you no longer have to choose. You can have the best of both worlds: the trusted eBPF development model and the performance of a dedicated userspace solution. We encourage you to try it out on your own network functions; we’d love to get your feedback.

---

## One‑liner and closing

- **One‑liner:** *uXDP runs your verified XDP programs in userspace, no code changes, with DPDK or AF\_XDP, and speeds up real NFs.*
- **Ask:** try it on your NF; we’d love feedback on reinjection and multi‑core scaling.

---

## Q\&A cheat‑sheet (expected questions)

**Safety if it’s userspace?** The bytecode is still verified by the kernel. Our runtime mirrors the exact semantics of kernel helpers and maps. And if the userspace process crashes, it doesn't take down the kernel.
**Map semantics: per‑CPU, LRU, etc.?** We implement them to match the kernel's behavior. For example, per-CPU arrays become thread-local storage. Everything is designed to work as you'd expect, including access from the control plane via shared memory.
**xdp\_md pointer widths?** We use a CO-RE/BTF shim that translates between the kernel's 32-bit offsets and userspace's 64-bit pointers, so the same bytecode works everywhere.
**Why not AF\_XDP alone?** AF_XDP is just a transport for getting packets to userspace. uXDP provides the runtime to actually execute the verified eBPF program, preserving your existing tools and workflows, while adding significant performance gains from our compiler optimizations.
**Why not write DPDK NFs directly?** You could, but you'd lose the safety of the eBPF verifier and the rich tooling of the ecosystem. You'd have to maintain a completely separate codebase and safety model. uXDP gives you one source of truth.
**Can it run without kernel eBPF?** You still need the kernel for verification. However, our offline AOT compiler can produce a native binary that you could run, once you've had the eBPF version verified.
**CPU/power?** Poll-mode drivers like DPDK do consume more power. AF_XDP provides a good middle ground, offering better performance than the kernel without the high CPU cost of constant polling.
**Multi‑core and NUMA?** Per-CPU maps are fully supported. Scaling across multiple cores is straightforward, and we recommend NUMA pinning for best performance.
**Zero‑copy vs copy in AF\_XDP?** We support both. Zero-copy performance depends on the NIC and driver, but our performance gains are present in both modes.
**SIMD correctness?** The verifier-visible semantics of the program are always preserved. We can use tools like Alive2 to formally check that the optimized IR is equivalent to the original, ensuring correctness.

---

## Slide building tips

* Max 4 bullets, 1–2 lines each; 32–36 pt text.
* Plots: large labels; annotate 1 takeaway per plot.
* Dark theme, high contrast; avoid putting code on slides (use keywords).
* Show only two numbers: *3.3×* and *+40%*.

---

## Rehearsal plan & word budget

* Aim for **~1,300 words** spoken. Print the notes, but don’t read them directly; use the bolded phrases as anchors to guide you.
* Do a 10‑min dry run first to get the timing down, then add the missing 2 min with more detail on results & takeaways.
* Time checkpoints: slide 7 at ~5:10, slide 12 at ~9:40.

---

## Backup slides (if allowed)

- Flamegraph details.
- Loader/JIT pipeline with IR snippets.
- Map layout and shared‑memory schema.
- Extra results: performance by packet size, zero‑copy vs copy.

---

## Optional closing script (20–25 s)

“So, uXDP keeps what’s great about XDP, things like its safety, maps, and workflows, and gives you userspace performance without needing a second codebase. You use the same binary and get better codegen. We’re planning to release packet reinjection soon, so if you run large-scale network functions, we’d love to work with you and get your feedback.”
