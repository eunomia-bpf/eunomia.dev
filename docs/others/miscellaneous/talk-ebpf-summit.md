# eBPF Summit 2025 bpftime talk

### Slide 1: bpftime: Userspace eBPF Runtime

“Hello, everyone. I’m Yusheng Zheng, and today, I’m excited to talk about **bpftime**, a userspace eBPF runtime for network and observability. 

You may remember bpftime from last year's Linux Plumber, but we’ve made a lot of progress since then. 

So, what is bpftime? It’s a userspace eBPF runtime that supports tracing features like Uprobe, USDT, syscall tracepoints, and even network features like XDP, all in userspace. It supports more than 10 map types and 30 helpers, so it’s highly compatible with the kernel eBPF ecosystem. 

you can use your fammiliar way to develop and deploy eBPF programs, but in userspace.
bpftime can run alongside kernel eBPF, using kernel eBPF maps and working together with kprobe.
It can now using either ubpf, which already used by eBPF for windoes, or a new eBPF VM called llvmbpf as its virtual machine for execution.

---

### Slide 2: bpftime for Observability

“So why do we want to do observability with eBPF in userspace?

It’s simple: userspace tracing is **faster and more flexible**. For example, Uprobes in the kernel take about **1000 nanoseconds**, but in userspace, we’ve brought that down to just **100 nanoseconds**. Similarly, memory access in userspace is about **10 times faster**—**4 nanoseconds** versus **40 nanoseconds** in the kernel. This speed difference happens because the kernel often has to translate memory addresses or run additional checks to access userspace memory.

On top of that, there’s less overhead on untraced processes, especially when dealing with syscall tracepoints.

What can we run in userspace?

With userspace tracing, tools like **bcc** and **bpftrace**  can run completely in userspace where kernel eBPF is not available. And you can run more complex observability agents that combine kprobes and uprobes, improving performance by shifting part of the workload to userspace.

---

### Slide 3: bpftime for Userspace Network

“Now, let’s talk about bpftime in the **networking** context. Why using userspace eBPF instead of running ebpf in kernel?

We’ve seen kernel-bypass solutions like **DPDK** and **AF_XDP**. They can offer faster packet processing by bypassing the kernel. But with bpftime, you can combine the performance benefits of these kernel-bypass technologies with the extensive eBPF ecosystem. So, you get the best of both low-latency packet processing and the ability to use eBPF’s safety and existing tools.

We can  also use **LLVM optimizations** to further boost performance in userspace.”

---

### Slide 4: Use bpftime with Userspace Network

“Using **bpftime**, you can seamlessly integrate the **eBPF XDP ecosystem** into kernel-bypass applications. 

For instance, solutions like **Katran**, a high-performance load balancer, can benefit from the optimizations we’ve made in bpftime for userspace. 

bpftime can work with both **AF_XDP** and **DPDK**. You can run your XDP eBPF programs as if they were in the kernel, just load them with bpftime, and they’ll work like normal, while a DPDK app handles the network processing.

Right now, there are some limitations with **XDP_TX** and **XDP_DROP** in userspace, but we’re actively working on solutions. We’re exploring ways to reinject packets into the kernel to support **XDP_PASS**.”

---

### Slide 5: Control Plane Support for Userspace eBPF

“One of the core features that allows bpftime to remain compatible with the existing eBPF ecosystem is its **control plane support** for userspace eBPF.

Control planes in eBPF are usually responsible for tasks like loading and unloading programs, 
configuring maps, and providing monitoring and debugging interfaces. bpftime can fully supports this in userspace by hooking syscalls using **LD_PRELOAD** or kernel eBPF, and connect to the userspace runtime.

---

### Slide 6: Benchmark

“Now, let’s see the **performance benchmarks**. 

We’ve benchmarked a variety of eBPF programs running with bpftime. bpftime has achieved up to **3x faster performance** in simple XDP network functions. In real-world applications
like Katran, bpftime can acheive up to  **40% faster**.

This shows that userspace eBPF can be fasyer kernel-based solutions, while retaining the flexibility that makes eBPF powerful.”

---

### Slide 7: llvmbpf: eBPF VM with LLVM JIT/AOT

“Let’s talk about **llvmbpf**. Originally, llvmbpf was part of the bpftime repo, 
but we’ve now separated it into a standalone project. llvmbpf is an eBPF virtual machine and compiler tool specifically optimized for userspace.

We’ve tested it with **bpf_conformance** for ompatibility, you can see it in CI. 
it can have better performance, and we can easily experiment different Optimization such as Inline helpers.
The  **llvmbpf** also supports both **JIT** and **AOT compilation**, depending on your needs.”

---

### Slide 8: llvmbpf: Build into Standalone Binary

**llvmbpf** also aupports building eBPF programs into **standalone binaries**, where you can call eBPF module like a simple C code, and Using eBPF bytecode as IR for verification. Maps and helpers are also supported. you don’t need any external dependencies

This can makes it easyer to **deploy eBPF programs** on any machine, including embedded systems and microcontrollers, which often don’t have an OS or runtime support for eBPF.

---

### Conclusion

In conclusion, **bpftime** are more than just userspace tracing and network performance. 
we hope it can bring the power of the eBPF ecosystem into userspace, without compromising on speed or flexibility.

I encourage all of you to explore bpftime for your observability and network needs. Thanks for professor marios from imperial college london, and
Thank you for your time, and feel free to check out the repo on GitHub or ask me any questions. 

