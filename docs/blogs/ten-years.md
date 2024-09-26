# Modernizing eBPF for the Next Decade: Past, Present, and Future

> This blog post is based on the presentation "[Modernize BPF for the Next 10 Years](http://oldvger.kernel.org/bpfconf2024_material/modernize_bpf.pdf)" by Alexei Starovoitov at BPFConf 2024.

Imagine having a Swiss Army knife for your computer's core operations—something that lets you peek inside how data moves, tweak processes on the fly, and monitor everything in real-time. That’s exactly what eBPF (Extended Berkeley Packet Filter) offers. Over the past decade, eBPF has evolved from a simple packet filtering tool into a powerhouse for networking, observability, and security. As we look forward to the next ten years, it's exciting to explore how eBPF has grown and where it's headed.

This blog delves into eBPF's journey, highlighting its key developments and discussing the innovative features on the horizon. Whether you're a developer, system administrator, or tech enthusiast, understanding eBPF's evolution and future will help you harness its full potential.

---

## A Decade in Review: eBPF's Journey So Far

### 2014: The Birth of Programmable Networking

Back in 2014, networking technology faced significant challenges. Traditional networking stacks were rigid, making it difficult to implement custom packet processing logic. Enter eBPF—a revolutionary tool that allowed developers to write small programs running directly in the kernel whenever a network packet arrived. This programmability meant more control over how data was handled, leading to better performance and flexibility without the need for cumbersome network drivers.

With eBPF, developers could create custom solutions tailored to specific networking needs. This shift opened the door to more efficient data processing and set the stage for innovative networking applications.

### Overcoming Networking Hurdles with XDP

As eBPF gained traction in networking, a major obstacle emerged: the `sk_buff` structure in the Linux kernel introduced too much overhead, making it challenging to achieve high speeds like 10 Gbps. While user-space networking solutions could reach these speeds, eBPF in the kernel struggled to keep up.

The breakthrough came with the introduction of eXpress Data Path (XDP). By running eBPF programs directly within the Network Interface Card (NIC) driver, XDP significantly reduced processing overhead. This allowed for faster packet handling and opened the door to high-speed networking applications. Tools like **Katran** and **Cilium** leveraged XDP to deliver blazing-fast networking solutions, showcasing eBPF's capability to handle high-throughput data with ease.

### Tracing Gets Smarter with BTF and CO-RE

As eBPF expanded into tracing and observability, developers encountered a new challenge: kernel data structures varied across different kernel versions. This inconsistency meant that BPF programs had to include specific kernel headers and be recompiled for each system, complicating deployment and maintenance.

To address this, the BPF Type Format (BTF) and Compile Once - Run Everywhere (CO-RE) were introduced. BTF adds type information to the kernel binary (`vmlinux`), allowing BPF programs to understand kernel data structures without needing to be recompiled for each version. CO-RE, facilitated by `libbpf`, enables BPF programs to be compiled once and run on any system, dynamically adapting to different kernel versions at load time.

These advancements made tracing tools more robust and portable, reducing the maintenance burden on developers and allowing for broader adoption of eBPF-based observability solutions.

### Making eBPF Easier with Skeletons and Global Variables

Writing eBPF programs in C often involved dealing with global variables that were difficult to manage from user space. These variables resided in `.data` and `.bss` sections, making interactions between user-space applications and BPF programs cumbersome.

The introduction of skeleton generation, powered by BTF and `libbpf`, changed the game. Skeletons allow developers to generate type-safe code that bridges the gap between user space and BPF programs. This means developers no longer have to wrestle with opaque global variables. Instead, they can interact with BPF variables in a structured and safe manner, significantly simplifying the development process. This enhancement not only reduces bugs but also accelerates the creation of feature-rich eBPF applications.

### From No Loops to Powerful Iterators

Early eBPF programs were limited in their control flow capabilities—loops were not supported to ensure that programs remained simple and verifiable. While this approach kept programs safe, it also restricted what developers could achieve.

Over the years, eBPF has gradually introduced more advanced looping mechanisms:

- **2014:** No support for loops, maintaining simplicity and safety.
- **2019:** Introduction of bounded loops, allowing loops with a fixed number of iterations.
- **2021:** Addition of the `bpf_loop()` helper function, providing more flexibility for looping constructs.
- **2023:** Implementation of open-coded iterators, offering even more powerful and efficient looping mechanisms.
- **2024 (Planned):** Introduction of `cond_break` to allow breaking out of loops based on specific conditions.

These enhancements enable developers to write more complex and efficient eBPF programs. With support for loops and advanced iterators, eBPF can handle sophisticated data processing tasks and perform real-time analytics directly within the kernel.

---

## Shaping the Future: What’s Next for eBPF

### Flexible Kernel Interfaces with kfuncs

Traditionally, BPF helper functions had fixed user-space APIs (UAPIs) with hard-coded IDs, which limited eBPF's flexibility and extensibility. The introduction of the **kfunc** mechanism changes this dynamic. Kfuncs allow kernel modules to define their own helper functions for BPF, providing a more flexible and extensible interface.

This innovation means developers can extend eBPF's capabilities without waiting for kernel updates. Custom helper functions can be tailored to specific needs, fostering innovation and enabling new use cases that were previously unattainable.

### Struct-Ops: Registering Callback Sets

The **struct-ops** mechanism allows sets of BPF programs to act as callbacks for stable kernel APIs like TCP congestion control. This opens the door for eBPF to integrate deeply with various kernel subsystems, such as:

- **Schedulers:** Creating custom task scheduling policies to optimize CPU usage.
- **HID (Human Interface Devices):** Developing unique input device handling mechanisms.
- **FUSE (Filesystem in Userspace):** Implementing flexible and efficient filesystem solutions.
- **Queuing Disciplines:** Managing network traffic more effectively, reducing latency, and improving throughput.

Struct-ops enable eBPF to enhance performance and flexibility across these subsystems, making it a versatile tool for a wide range of kernel-level customizations.

### Advanced Algorithms and Data Structures with bpf_arena

As eBPF use cases expand, there's a growing need for more complex data structures like trees and graphs. The introduction of **bpf_arena** addresses this by providing a shared memory space between BPF and user space. This allows developers to implement intricate algorithms and data structures directly within eBPF programs.

With bpf_arena, developers can handle more sophisticated data processing tasks, optimize memory usage, and improve access patterns. This enhancement paves the way for eBPF to support advanced applications that require robust data management capabilities.

### Building a Rich Ecosystem with BPF Libraries

Sharing code across BPF programs has historically been challenging due to dependency management issues. Inspired by languages like Rust and Python, the future of eBPF lies in robust library support. By distributing libraries as source code, developers can simplify dependencies and encourage code reuse.

This approach fosters a community-driven ecosystem where developers can build on each other's work, reducing duplication and accelerating development. A rich library ecosystem will make it easier to create feature-rich eBPF applications, driving broader adoption and innovation.

### Enhancing Concurrency with Arbitrary Locks

Current locking mechanisms in eBPF, such as `bpf_spin_lock()`, are limited and prone to deadlocks. This restricts the development of more complex, concurrent BPF applications. The proposed solution is a new locking system that supports multiple locks and prevents deadlocks.

This upgrade will allow more sophisticated concurrency patterns within BPF programs, enabling developers to build more reliable and efficient applications. With better concurrency support, eBPF can handle more demanding tasks without compromising system stability.

### Embracing Turing Completeness

eBPF is already Turing complete, meaning it can perform any computation given enough resources. However, to fully leverage this potential, additional features like jump tables and indirect `goto` instructions are needed. These enhancements will enable more dynamic and flexible control flow within eBPF programs.

With these improvements, eBPF can support even more powerful and flexible programming models within the kernel. This will push the boundaries of what eBPF can achieve, opening up new possibilities for developers.

---

## Making eBPF Even Better: Instruction Set and Registers

### Evolving the BPF Instruction Set (ISA)

Certain operations in eBPF remain clunky or inefficient. Enhancing the instruction set can make a significant difference in both performance and ease of use. Proposed enhancements include:

- **Indirect Calls:** Introducing new opcodes to simplify and speed up function calls.
- **Bit Manipulation:** Adding instructions for common bit operations, such as finding and counting bits, can optimize frequent tasks.

These additions will make eBPF programs more efficient and easier to write, expanding their usability and performance.

### Optimizing Registers

Different architectures offer varying numbers of registers, and eBPF can sometimes be inefficient in how it uses them. Potential improvements include:

- **Virtual Registers:** Abstracting away hardware limitations to maximize efficiency.
- **Register Spilling/Filling:** Optimizing how registers are used and managed to prevent bottlenecks.
- **More Hardware Registers:** Allowing compilers to take advantage of additional registers when available.

Better register management means faster and more efficient eBPF programs, enhancing overall performance and making eBPF a more powerful tool for developers.

### Handling More Function Arguments

Currently, eBPF functions are limited to passing five arguments due to register constraints. To overcome this limitation, two solutions are proposed:

- **Additional Registers:** Utilizing more registers where possible to pass extra arguments.
- **Stack Space:** Passing extra arguments via the stack, carefully managing performance and safety.

These solutions will provide more flexibility in function calls, allowing for more complex and capable eBPF programs.

---

## Ambitious Goals: Compiling the Kernel to BPF ISA

Imagine if significant parts of the Linux kernel could be compiled to the BPF instruction set. This vision would revolutionize kernel development and analysis, offering several exciting benefits:

- **Enhanced Analysis:** Monitoring and verifying kernel behavior becomes easier with BPF’s flexible programmability.
- **Flexibility:** Quickly adapting and updating kernel components without needing a full recompilation.

This ambitious goal envisions a more dynamic and adaptable kernel, driven by the power and flexibility of eBPF. It could lead to more efficient kernel development cycles and a more resilient operating system overall.

---

## Memory Management Upgrades: Dynamic Stacks and More

### Breaking Free from the 512-Byte Stack Limit

eBPF programs currently face a strict 512-byte stack limit, which restricts their complexity and the types of computations they can perform. To overcome this limitation, introducing `alloca()` will allow dynamic memory allocation within eBPF programs.

With `alloca()`, the stack can grow as needed, enabling more complex functions and data structures. This enhancement will allow developers to create more sophisticated and feature-rich eBPF programs, expanding the range of possible applications.

---

## Safer Programs: Cancellable eBPF Scripts

Long-running eBPF programs can consume significant CPU resources, potentially leading to system instability. To address this, new mechanisms for safely canceling these programs are proposed.

Implementing timeouts will automatically terminate programs that run too long, while watchdogs can monitor and manage program execution. Additionally, providing safe cancellation points ensures that programs can be stopped without causing system issues.

These safeguards will make eBPF programs more reliable and stable, even when handling complex tasks, ensuring that the system remains responsive and secure.

---

## Expanding Observability to User Space

### Making User-Space Monitoring Easier

Observing what's happening in user-space applications is inherently more complex than monitoring the kernel. Diverse programming languages and runtime environments add to the challenge. However, eBPF is evolving to bridge this gap.

Innovations like Fast Uprobes offer efficient user-space probes with minimal performance impact. User-Space Statically Defined Tracing (USDT) allows applications to define their own tracing points, providing more granular monitoring. Additionally, language-specific stack walkers for languages like C++, Python, and Java can interpret their specific stack frames, offering meaningful trace information.

These advancements enable more comprehensive and detailed monitoring of user-space applications, providing better insights and debugging capabilities for developers and system administrators.

---

## Rethinking Limits: The 1 Million Instruction Cap

Currently, eBPF programs are limited to 1 million instructions to ensure they remain verifiable and terminate correctly. While this safeguard maintains system stability, it also limits the complexity of what eBPF programs can achieve.

There's an ongoing debate about relaxing this limit for programs that can demonstrate forward verification progress. Balancing the need for more complex programs with system safety and performance is crucial. If successfully implemented, this change could allow more sophisticated eBPF applications, expanding their usefulness without compromising security or stability.

---

## Modularizing eBPF: Independent Kernel Modules

Imagine being able to update the eBPF subsystem without needing to update the entire kernel. This vision is becoming a reality by making BPF a separate kernel module.

This modular approach offers several advantages:

- **Faster Updates:** New features and fixes can be rolled out more quickly without waiting for full kernel releases.
- **Reduced Dependency:** Developers and users don’t have to wait for kernel updates to leverage the latest eBPF capabilities.
- **Increased Flexibility:** Experimentation and innovation can proceed without being tied to kernel release cycles.

This shift will lead to a more agile and responsive eBPF ecosystem, keeping pace with rapid technological advancements and developer needs.

---

## What's Next for eBPF?

eBPF has already tackled major challenges in tracing, observability, and programmable networking. But the journey doesn’t stop here. The future holds even more exciting possibilities:

### Expanding into Security with bpf-lsm

**bpf-lsm** (BPF Linux Security Modules) allows eBPF to enforce custom security policies. This means developers can tailor security measures to specific needs, leveraging eBPF’s power to monitor and control system behavior in real-time. With bpf-lsm, eBPF can play a crucial role in enhancing system security, offering more granular and dynamic protection mechanisms.

### Optimizing Scheduling

Applying eBPF to task and packet scheduling can lead to better performance and resource management. For task scheduling, eBPF can create custom scheduling policies that optimize CPU usage based on specific workloads. For packet scheduling, eBPF can manage network traffic more effectively, reducing latency and improving throughput. These optimizations will result in more efficient and responsive systems capable of handling diverse workloads with ease.

---

## eBPF's Guiding Principles

At the heart of eBPF’s evolution are three core principles:

1. **To Innovate:** Continuously pushing the boundaries of what's possible in both kernel and user-space programming.
2. **To Enable Others to Innovate:** Providing tools and frameworks that empower developers to build new and exciting solutions.
3. **To Challenge What is Possible:** Breaking through existing limitations and redefining what operating systems can do.

These principles ensure that eBPF remains a cutting-edge tool, driving forward the future of computing by fostering creativity and overcoming challenges.

---

## Conclusion

Looking back, it's clear that eBPF has come a long way—from its humble beginnings in programmable networking to becoming a powerhouse for tracing, observability, and beyond. The future is even brighter, with exciting features on the horizon that promise to make eBPF even more powerful, flexible, and user-friendly.

As we move into the next decade, eBPF stands ready to tackle new challenges and unlock new possibilities. Whether you're a developer looking to optimize your applications, a system administrator striving for better performance, or just a tech enthusiast eager to explore the latest innovations, eBPF has something to offer.

For more details and intsresting topics, please see <http://oldvger.kernel.org/bpfconf2024.html>
