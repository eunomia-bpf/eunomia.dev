---
date: 2024-08-11
---

# The Past, Present, and Future of eBPF and Its Path to Revolutionizing Systems

> This blog post mainly references Alexei Starovoitov's presentation "[Modernize BPF for the Next 10 Years](http://oldvger.kernel.org/bpfconf2024_material/modernize_bpf.pdf)" at BPFConf 2024.

Imagine having a Swiss Army knife for your computer's core operations—something that lets you peek inside how data moves, tweak processes on the fly, and monitor everything in real-time. That’s exactly what [eBPF](https://en.wikipedia.org/wiki/EBPF) (Extended Berkeley Packet Filter) offers. Over the past decade, eBPF has transformed from a simple packet filtering tool into a powerhouse for networking, observability, and security. So, what’s next for eBPF? Let’s dive into its journey, explore where it’s headed in the next ten years, and discuss the challenges and opportunities that lie ahead. This exploration will help us understand how eBPF is shaping the future of modern systems.
<!-- more -->

## A Decade in Review: eBPF's Journey So Far

### How Did Programmable Networking Begin with eBPF?

Back in 2014, the networking world was facing some serious limitations. Traditional networking stacks were rigid, making it tough to implement custom packet processing logic tailored to specific needs. Enter [eBPF](https://isovalent.com/blog/post/ebpf-documentary-creation-story/)—a game-changer that allowed developers to write small programs running directly in the kernel whenever a network packet arrived. This innovation meant more control over data handling, leading to better performance and flexibility without the hassle of cumbersome network drivers.

With eBPF, developers could create solutions that fit their exact networking requirements, paving the way for more efficient data processing and innovative networking applications. This shift marked the beginning of programmable networking, where customization and performance could go hand in hand.

### What Made XDP a Game-Changer for High-Speed Networking?

As eBPF gained traction in networking, a significant hurdle emerged: the `sk_buff` structure in the Linux kernel introduced too much overhead, making it difficult to achieve high speeds like 10 Gbps. While user-space networking solutions could reach these speeds, eBPF running in the kernel struggled to keep up.

The breakthrough came with [eXpress Data Path (XDP)](https://www.iovisor.org/technology/xdp). By running eBPF programs directly within the Network Interface Card (NIC) driver, XDP significantly reduced processing overhead. This allowed for much faster packet handling, enabling high-speed networking applications that were previously out of reach.

Tools like **Katran** and **Cilium** leveraged XDP to deliver lightning-fast networking solutions, showcasing eBPF's ability to handle high-throughput data effortlessly. XDP turned eBPF into a viable option for environments demanding top-tier network performance, solidifying its role in modern networking.

### How Did BTF and CO-RE Make Tracing Smarter?

As eBPF expanded into tracing and observability, developers encountered a new challenge: kernel data structures varied across different kernel versions. This inconsistency meant BPF programs had to include specific kernel headers and be recompiled for each system, complicating deployment and maintenance.

Enter [BPF Type Format (BTF)](https://docs.kernel.org/bpf/btf.html) and [Compile Once - Run Everywhere (CO-RE)](https://nakryiko.com/posts/bpf-portability-and-co-re/). BTF adds type information to the kernel binary (vmlinux), allowing BPF programs to understand kernel data structures without needing recompilation for each version. CO-RE, facilitated by [libbpf](https://libbpf.readthedocs.io/en/latest/libbpf_overview.html), lets BPF programs be compiled once and run on any system, dynamically adapting to different kernel versions at load time.

These advancements made tracing tools more robust and portable, easing the maintenance burden on developers and encouraging broader adoption of eBPF-based observability solutions. Developers could now deploy tracing tools across diverse environments without worrying about kernel version mismatches, greatly enhancing productivity and reliability.

### How Did Skeletons and Global Variables Simplify eBPF Development?

Writing eBPF programs in C often meant dealing with global variables that were tricky to manage from user space. These variables lived in `.data` and `.bss` sections, making interactions between user-space applications and BPF programs cumbersome and error-prone.

The introduction of skeleton generation, powered by BTF and libbpf, changed the game. Skeletons allow developers to generate type-safe code that bridges the gap between user space and BPF programs. No more wrestling with opaque global variables! Instead, developers can interact with BPF variables in a structured and safe manner, significantly simplifying the development process.

This not only reduces bugs but also speeds up the creation of feature-rich eBPF applications. Additionally, tools like [GPTtrace](https://github.com/eunomia-bpf/GPTtrace) leverage large language models (LLMs) to further simplify the development of eBPF programs, lowering the barrier for developers without deep OS expertise. This combination of skeletons and AI-driven tools makes eBPF development more accessible and efficient than ever before.

### From No Loops to Powerful Iterators: How Did Control Flow in eBPF Evolve?

Early eBPF programs were limited in their control flow capabilities—loops weren’t supported to keep programs simple and verifiable. While this approach ensured safety, it also restricted what developers could achieve, limiting the complexity of eBPF applications.

Over the years, eBPF has gradually introduced more advanced looping mechanisms:

- **2014:** No support for loops, maintaining simplicity and safety.
- **2019:** Introduction of bounded loops, allowing loops with a fixed number of iterations.
- **2021:** Addition of the [`bpf_loop()`](https://docs.ebpf.io/linux/helper-function/bpf_loop/) helper function, providing more flexibility for looping constructs.
- **2023:** Implementation of open-coded [iterators](https://lwn.net/Articles/926041/), offering even more powerful and efficient looping mechanisms.
- **2024 (Planned):** Introduction of [`cond_break`](https://lwn.net/Articles/964641/) to allow breaking out of loops based on specific conditions.

These enhancements enable developers to write more complex and efficient eBPF programs. With support for loops and advanced iterators, eBPF can handle sophisticated data processing tasks and perform real-time analytics directly within the kernel. This evolution in control flow capabilities has unlocked new possibilities for what eBPF can achieve, making it a more versatile tool for developers.

---

## Shaping the Future: What’s Next for eBPF?

As we look to the future, eBPF continues to evolve with cutting-edge features and enhancements that promise to revolutionize how we interact with system internals. Let’s explore some of the exciting developments on the horizon and the opportunities and challenges they present.

### How Will kfuncs Make Kernel Interfaces More Flexible?

Traditionally, BPF helper functions had fixed user-space APIs (UAPIs) with hard-coded IDs, limiting eBPF's flexibility and extensibility. The introduction of the **kfunc** mechanism changes this dynamic. [Kfuncs](https://docs.kernel.org/bpf/kfuncs.html) allow kernel modules to define their own helper functions for BPF, providing a more flexible and extensible interface.

This means developers can extend eBPF's capabilities without waiting for kernel updates. Custom helper functions can be tailored to specific needs, fostering innovation and enabling new use cases that were previously out of reach. For more details, see the [Tutorial on kfuncs](https://eunomia.dev/tutorials/43-kfuncs/).

By allowing kernel modules to define their own helpers, kfuncs make the eBPF ecosystem more adaptable and responsive to emerging requirements, ensuring that eBPF remains relevant and powerful in a rapidly changing technological landscape.

### What Are Struct-Ops and How Do They Enhance eBPF?

Adding new eBPF attach types for kernel subsystems has been challenging due to the lack of a stable interface. The **struct-ops** mechanism addresses this by allowing sets of BPF programs to act as callbacks for stable kernel APIs like TCP congestion control.

This opens the door for eBPF to integrate deeply with various kernel subsystems, such as:

- **Schedulers:** Creating custom [eBPF task scheduling policies](https://www.kernel.org/doc/html/next/scheduler/sched-ext.html) to optimize CPU usage.
- **HID (Human Interface Devices):** Developing unique [eBPF input device handling mechanisms](https://docs.kernel.org/hid/hid-bpf.html).
- **FUSE (Filesystem in Userspace):** Implementing flexible and efficient [eBPF fuse solutions](https://lpc.events/event/16/contributions/1339/attachments/945/1861/LPC2022%20Fuse-bpf.pdf).
- **Queuing Disciplines:** Managing network traffic more effectively, reducing latency, and improving throughput. See the [link](https://netdevconf.info/0x17/sessions/talk/ebpf-qdisc-a-generic-building-block-for-traffic-control.html) for details.

[Struct-ops](https://docs.ebpf.io/linux/program-type/BPF_PROG_TYPE_STRUCT_OPS/) enable eBPF to enhance performance and flexibility across these subsystems, making it a versatile tool for a wide range of kernel-level customizations. By providing a stable interface, struct-ops simplify the integration process, encouraging more widespread adoption and innovative applications of eBPF in system management and optimization.

### How Does bpf_arena Enhance Data Structures in eBPF?

As eBPF use cases expand, there's a growing need for more complex data structures like trees and graphs. The introduction of **bpf_arena** addresses this by providing a shared memory space between BPF and user space. [bpf_arena](https://lwn.net/Articles/961594/) allows developers to implement intricate algorithms and data structures directly within eBPF programs.

With bpf_arena, developers can handle more sophisticated data processing tasks, optimize memory usage, and improve access patterns. This enhancement paves the way for eBPF to support advanced applications that require robust data management capabilities. Detailed capabilities are outlined in the [eBPF documentation](https://docs.ebpf.io/linux/kfuncs/bpf_arena_free_pages/).

By facilitating the creation of complex data structures, bpf_arena significantly broadens the scope of what eBPF can achieve, enabling more advanced analytics, monitoring, and optimization tasks within the kernel.

### Why Are BPF Libraries Important for a Rich Ecosystem?

Sharing code across BPF programs has historically been challenging due to dependency management issues. Inspired by languages like Rust and Python, the future of eBPF lies in robust library support. By distributing libraries as source code, developers can simplify dependencies and encourage code reuse.

This approach fosters a community-driven ecosystem where developers can build on each other's work, reducing duplication and accelerating development. A rich library ecosystem will make it easier to create feature-rich eBPF applications, driving broader adoption and innovation.

Robust BPF libraries provide standardized tools and functions that developers can leverage, enhancing productivity and ensuring consistency across different eBPF projects. This collective effort not only speeds up development but also improves the overall quality and reliability of eBPF applications.

### How Will Arbitrary Locks Improve Concurrency in eBPF?

Current locking mechanisms in eBPF, such as `bpf_spin_lock()`, are limited and prone to deadlocks. This restricts the development of more complex, concurrent BPF applications. The proposed solution is a new locking system that supports multiple locks and prevents deadlocks.

This upgrade will allow more sophisticated concurrency patterns within BPF programs, enabling developers to build more reliable and efficient applications. With better concurrency support, eBPF can handle more demanding tasks without compromising system stability. Learn more from [LWN.net](https://lwn.net/Articles/779120/) and the [eBPF Documentation](https://docs.ebpf.io/linux/concepts/concurrency/).

Improved concurrency mechanisms will enhance the performance and scalability of eBPF applications, making them more suitable for high-performance environments where multiple operations need to run simultaneously without interference.

### What Does Embracing Turing Completeness Mean for eBPF?

eBPF is already [Turing complete](https://isovalent.com/blog/post/ebpf-yes-its-turing-complete/), meaning it can perform any computation given enough resources. However, to fully leverage this potential, additional features like jump tables and indirect goto instructions are needed. These enhancements will enable more dynamic and flexible control flow within eBPF programs.

With these improvements, eBPF can support even more powerful and flexible programming models within the kernel. This will push the boundaries of what eBPF can achieve, opening up new possibilities for developers. Embracing Turing completeness fully will allow eBPF to handle more complex algorithms and processes, making it an even more indispensable tool for system programming and optimization.

## Making eBPF Even Better: Instruction Set and Registers

### How Will Evolving the BPF Instruction Set (ISA) Improve eBPF?

Certain operations in eBPF remain clunky or inefficient. Enhancing the instruction set can make a significant difference in both performance and ease of use. Proposed enhancements include:

- **Indirect Calls:** Introducing new opcodes to simplify and speed up function calls. See [LPC talk](https://lpc.events/event/18/contributions/1941/).
- **Bit Manipulation:** Adding instructions for common bit operations, such as finding and counting bits, can optimize frequent tasks. See the [LPC talk](https://lpc.events/event/18/contributions/1949/).

These additions will make eBPF programs more efficient and easier to write, expanding their usability and performance. Detailed specifications can be found in the [Kernel Docs](https://docs.kernel.org/bpf/standardization/instruction-set.html).

By refining the instruction set, eBPF becomes more powerful and versatile, allowing developers to write more optimized and feature-rich programs without unnecessary complexity.

### What Optimizations Can Be Made to eBPF Registers?

Different architectures offer varying numbers of registers, and eBPF can sometimes be inefficient in how it uses them. Potential improvements include:

- **Virtual Registers:** Abstracting away hardware limitations to maximize efficiency.
- **Register Spilling/Filling:** Optimizing how registers are used and managed to prevent bottlenecks.
- **More Hardware Registers:** Allowing compilers to take advantage of additional registers when available.

Better register management means faster and more efficient eBPF programs, enhancing overall performance and making eBPF a more powerful tool for developers. Optimizing register usage is crucial for ensuring that eBPF can handle increasingly complex tasks without running into resource limitations.

### How Can eBPF Handle More Function Arguments?

Currently, eBPF functions are limited to passing five arguments due to register constraints. To overcome this limitation, two solutions are proposed:

- **Additional Registers:** Utilizing more registers where possible to pass extra arguments.
- **Stack Space:** Passing extra arguments via the stack, carefully managing performance and safety.

These solutions will provide more flexibility in function calls, allowing for more complex and capable eBPF programs. For more information, see the [instruction set](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-04.html) and [Stack Overflow](https://stackoverflow.com/questions/70905815/how-to-read-all-parameters-from-a-function-ebpf).

Enhancing the ability to handle more function arguments will enable developers to write more comprehensive and feature-rich eBPF programs, expanding the scope of applications that can be efficiently managed within the kernel.

---

## Ambitious Goals: Compiling the Kernel to BPF ISA

Imagine if significant parts of the Linux kernel could be compiled to the BPF instruction set. This vision would revolutionize kernel development and analysis, offering several exciting benefits:

- **Enhanced Analysis:** Monitoring and verifying kernel behavior becomes easier with BPF’s flexible programmability.
- **Flexibility:** Quickly adapting and updating kernel components without needing a full recompilation.

This ambitious goal envisions a more dynamic and adaptable kernel, driven by the power and flexibility of eBPF. It could lead to more efficient kernel development cycles and a more resilient operating system overall. Discussions on this can be found in [LWN.net](https://lwn.net/Articles/975830/) and the [IETF Draft on BPF ISA](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-04.html).

Compiling the kernel to BPF ISA would allow developers to write and deploy kernel modules with the same ease and flexibility that eBPF already provides, streamlining development and enhancing system reliability.

---

## Memory Management Upgrades: Dynamic Stacks and More

### How Can We Break Free from the 512-Byte Stack Limit?

eBPF programs currently face a strict 512-byte stack limit, which restricts their complexity and the types of computations they can perform. To overcome this limitation, introducing `alloca()` will allow dynamic memory allocation within eBPF programs.

With `alloca()`, the stack can grow as needed, enabling more complex functions and data structures. This enhancement will allow developers to create more sophisticated and feature-rich eBPF programs, expanding the range of possible applications. Details are available in the [bpf_arena_alloc_pages](https://lwn.net/Articles/961594/).

Breaking free from the stack limit will empower developers to implement more intricate logic and handle larger datasets within eBPF programs, enhancing their capabilities and applications.

---

## Safer Programs: Cancellable eBPF Scripts

### How Can We Ensure eBPF Programs Remain Safe and Efficient?

Long-running eBPF programs can consume significant CPU resources, potentially leading to system instability. To address this, new mechanisms for safely canceling these programs are proposed.

Implementing timeouts will automatically terminate programs that run too long, while watchdogs can monitor and manage program execution. Additionally, providing safe cancellation points ensures that programs can be stopped without causing system issues.

These safeguards will make eBPF programs more reliable and stable, even when handling complex tasks, ensuring that the system remains responsive and secure. By introducing these safety mechanisms, eBPF can be used more confidently in critical environments where system stability is paramount.

---

## Expanding Observability to User Space

### How Is eBPF Making User-Space Monitoring Easier?

Observing what's happening in user-space applications is inherently more complex than monitoring the kernel. Diverse programming languages and runtime environments add to the challenge. However, eBPF is evolving to bridge this gap.

Innovations like Fast Uprobes offer efficient user-space probes with minimal performance impact. User-Space Statically Defined Tracing (USDT) allows applications to define their own tracing points, providing more granular monitoring. Additionally, language-specific stack walkers for languages like C++, Python, and Java can interpret their specific stack frames, offering meaningful trace information.

These advancements enable more comprehensive and detailed monitoring of user-space applications, providing better insights and debugging capabilities for developers and system administrators. By extending observability into user space, eBPF ensures that every aspect of system performance can be meticulously tracked and optimized.

---

## Rethinking Limits: The 1 Million Instruction Cap

### Should We Relax the 1 Million Instruction Limit in eBPF?

Currently, eBPF programs are limited to 1 million instructions to ensure they remain verifiable and terminate correctly. While this safeguard maintains system stability, it also limits the complexity of what eBPF programs can achieve.

There's an ongoing debate about relaxing this limit for programs that can demonstrate forward verification progress. Balancing the need for more complex programs with system safety and performance is crucial. If successfully implemented, this change could allow more sophisticated eBPF applications, expanding their usefulness without compromising security or stability. For more insights, refer to [LWN.net](https://lwn.net/Articles/975830/).

Relaxing the instruction limit could unlock new possibilities for eBPF, allowing it to handle more extensive and intricate tasks while still maintaining the necessary safeguards to protect system integrity.

---

## Modularizing eBPF: Independent Kernel Modules

### What Are the Benefits of Making eBPF a Separate Kernel Module?

There has been significant discussion about making eBPF an independent kernel module. Imagine being able to update the eBPF subsystem without needing to update the entire kernel. This vision is becoming a reality by making BPF a separate kernel module.

This modular approach offers several advantages:

- **Faster Updates:** New features and fixes can be rolled out more quickly without waiting for full kernel releases.
- **Reduced Dependency:** Developers and users don’t have to wait for kernel updates to leverage the latest eBPF capabilities.
- **Increased Flexibility:** Experimentation and innovation can proceed without being tied to kernel release cycles.

This shift will lead to a more agile and responsive eBPF ecosystem, keeping pace with rapid technological advancements and developer needs. By decoupling eBPF from the kernel, updates and improvements can be deployed more efficiently, enhancing the overall user experience and system performance.

---

## Expanding eBPF to Other Platforms

Beyond Linux, eBPF's capabilities are extending to other platforms, broadening its impact and utility across different environments.

Furthermore, [eBPF for Windows](https://github.com/microsoft/ebpf-for-windows) extends eBPF's capabilities beyond Linux, enabling developers to utilize eBPF's powerful features on Windows systems. This cross-platform support opens up new avenues for developers who work in heterogeneous environments, allowing them to apply eBPF's benefits regardless of the operating system.

Additionally, Userspace eBPF runtime such as [bpftime](https://github.com/eunomia-bpf/bpftime) overcomes kernel-space limitations, unlocking even more potential for eBPF applications. By enabling user-space execution of eBPF application such as bcc-tools or bpftrace, bpftime allows for greater flexibility and experimentation, making eBPF accessible to a wider range of use cases and developers.

Expanding eBPF to other platforms ensures that its powerful features are available to a broader audience, promoting innovation and enhancing system performance across diverse operating systems.

---

## What's Next for eBPF?

eBPF has already tackled major challenges in tracing, observability, and programmable networking. But the journey doesn’t stop here. The future holds even more exciting possibilities:

### How Will bpf-lsm Expand eBPF into Security?

**bpf-lsm** ([BPF Linux Security Modules](https://docs.kernel.org/bpf/prog_lsm.html)) allows eBPF to enforce custom security policies. This means developers can tailor security measures to specific needs, leveraging eBPF’s power to monitor and control system behavior in real-time. With bpf-lsm, eBPF can play a crucial role in enhancing system security, offering more granular and dynamic protection mechanisms.

By integrating security directly into the kernel via eBPF, bpf-lsm provides a flexible and powerful way to implement and manage security policies, making systems more resilient against threats and vulnerabilities.

### Can eBPF Optimize Scheduling for Better Performance?

Applying eBPF to task and packet scheduling can lead to better performance and resource management. For task scheduling, eBPF can create custom scheduling policies that optimize CPU usage based on specific workloads. For packet scheduling, eBPF can manage network traffic more effectively, reducing latency and improving throughput. These optimizations will result in more efficient and responsive systems capable of handling diverse workloads with ease. Check out the [sched-ext/scx](https://github.com/sched-ext/scx) repository for more details.

Optimizing scheduling with eBPF ensures that system resources are utilized more effectively, enhancing overall performance and user experience, especially in environments with varying and demanding workloads.

---

## eBPF's Guiding Principles

At the heart of eBPF’s evolution are three core principles:

1. **To Innovate:** Continuously pushing the boundaries of what's possible in both kernel and user-space programming.
2. **To Enable Others to Innovate:** Providing tools and frameworks that empower developers to build new and exciting solutions.
3. **To Challenge What is Possible:** Breaking through existing limitations and redefining what operating systems can do.

These principles ensure that eBPF remains a cutting-edge tool, driving forward the future of computing by fostering creativity and overcoming challenges. By adhering to these guiding tenets, eBPF continues to evolve and adapt, maintaining its position at the forefront of system development and optimization.

---

## Conclusion

Looking back, it's clear that eBPF has come a long way—from its humble beginnings in programmable networking to becoming a powerhouse for tracing, observability, and beyond. The future is even brighter, with exciting features on the horizon that promise to make eBPF even more powerful, flexible, and user-friendly.

These developments, along with tools like LLMs such as [GPTtrace](https://github.com/eunomia-bpf/GPTtrace), [eBPF for Windows](https://github.com/microsoft/ebpf-for-windows), and [bpftime](https://github.com/eunomia-bpf/bpftime), are making eBPF more accessible and versatile across different environments, further cementing its role as a critical tool in modern system development. Additionally, we are leveraging large language models to better understand eBPF code in the kernel through projects like [code-survey](https://github.com/eunomia-bpf/code-survey), enhancing our ability to analyze and optimize eBPF programs.

As we move into the next decade, eBPF stands ready to tackle new challenges and unlock new possibilities. Whether you're a developer looking to optimize your applications, a system administrator striving for better performance, or just a tech enthusiast eager to explore the latest innovations, eBPF has something to offer.

For more details and interesting topics, please visit [BPFConf 2024](http://oldvger.kernel.org/bpfconf2024.html).

---

## References

1. [Wikipedia](https://en.wikipedia.org/wiki/EBPF) - An overview of eBPF’s journey and its capabilities to run programs securely in the kernel.
2. [Isovalent Blog](https://isovalent.com/blog/post/ebpf-documentary-creation-story/) - Chronicles the story behind eBPF’s creation and its impact on the tech industry.
3. [IO Visor](https://www.iovisor.org/technology/xdp) - XDP allows for efficient packet processing directly in the kernel, offering significant performance enhancements.
4. [XDP](https://www.tigera.io/learn/guides/ebpf/ebpf-xdp/) - A framework for fast packet processing that illustrates the advantages of integrating XDP with eBPF applications.
5. [BPF Type Format](https://docs.kernel.org/bpf/btf.html) - BTF provides essential type information that enhances the verifiability and portability of BPF applications.
6. [libbpf Documentation](https://libbpf.readthedocs.io/en/latest/libbpf_overview.html) - Skeleton files ease the interaction between user space and BPF programs, optimizing the management of global variables.
7. [GitHub Discussion](https://github.com/cilium/ebpf/discussions/943) - Discusses how global variables can be accessed and managed within eBPF applications.
8. [Speaker Deck](https://speakerdeck.com/f1ko/ebpf-vienna-bpf-evolution-of-a-loop) - Analyzes the functioning and verification of control flows within eBPF.
9. [Kernel Docs](https://docs.kernel.org/bpf/kfuncs.html) - Provides insights on kernel functions that enhance the flexibility and extensibility of BPF applications.
10. [Kfunc Tutorial](https://eunomia.dev/tutorials/43-kfuncs/) - Describes how custom kfuncs enable more powerful interactions between kernel functions and eBPF programs.
11. [eBPF Docs](https://docs.ebpf.io/linux/program-type/BPF_PROG_TYPE_STRUCT_OPS/) - Explains how struct-ops improve performance and allow for more elaborate interfaces between BPF programs and kernel subsystems.
12. [LWN.net](https://lwn.net/Articles/961594/) - Discusses bpf_arena as a memory region that supports custom data structures shared between BPF programs and user space.
13. [eBPF Docs](https://docs.ebpf.io/linux/kfuncs/bpf_arena_free_pages/) - Details the capabilities of bpf_arena in managing complex data structures.
14. [Red Hat Developers](https://developers.redhat.com/articles/2023/10/19/ebpf-application-development-beyond-basics) - Discusses the role of libbpf in streamlining interactions and enhancing program development.
15. [LWN.net](https://lwn.net/Articles/779120/) - Explains the significance of arbitrary locks in managing concurrency in eBPF, enhancing process integrity.
16. [eBPF Documentation](https://docs.ebpf.io/linux/concepts/concurrency/) - Offers a comprehensive overview of concurrency management techniques within eBPF programs.
17. [Isovalent](https://isovalent.com/blog/post/ebpf-yes-its-turing-complete/) - Confirms that eBPF is Turing complete, capable of solving any computable problem with potential applications spanning various domains.
18. [Kernel Docs](https://docs.kernel.org/bpf/standardization/instruction-set.html) - Outlines the specifications and historical context of BPF ISA, along with recent enhancements for better performance.
19. [Stack Overflow](https://stackoverflow.com/questions/70905815/how-to-read-all-parameters-from-a-function-ebpf) - Offers insights into how parameters can be accessed and managed within eBPF functions.
20. [Standardizing the BPF ISA - LWN.net](https://lwn.net/Articles/975830/) - Discusses the broader implications of compiling kernel modules to use BPF ISA and the benefits it brings.
21. [IETF Draft on BPF ISA](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-04.html) - Examines the details surrounding BPF ISA and its roadmap for future enhancements.
23. [bpf_arena_alloc_pages](https://lwn.net/Articles/961941/) - Details the introduction of `alloca()` for dynamic memory allocation within eBPF programs.
24. [sched-ext/scx](https://github.com/sched-ext/scx) - Repository for task and packet scheduling optimizations using eBPF.

