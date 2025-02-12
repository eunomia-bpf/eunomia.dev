# bpftime talk at linux plumbers 2025

## Title Slide: Introducing bpftime - A New Frontier in Userspace eBPF Runtime （30s）

Hello everyone, and thank you for joining me today.
I'm excited to talk about our latest project - bpftime, a innovative new userspace eBPF runtime.

My name is Yusheng Zheng, currently maintaining a small community called eunomia-bpf, building open source projects related to eBPF. In fact, this is the first time I've spoken at linux plumbers, so I'm very excited to be here.

## Agenda (1m)

This is the topic we are going to discuss about today. I will start with a brief introduction to Why a new userspace eBPF runtime, there may be kernel Uprobe performance issues, kernel eBPF security concerns and limited configurable, other userspace eBPF runtime limitations, and some existing Non-kernel eBPF Usecases.

After that, I will introduce bpftime, how it works, the benchmarks and examples, and how we can run existing eBPF tools and projects on it.

Finally, I will talk about some open problems and our future plans.

## Why bpftime: uprobe (2m)

Let's start with why we felt the need to create a new userspace eBPF runtime. eBPF has change the way we approach observability, networking, and security in Linux environments.

However, its integration at the kernel level has not been without challenges.

Uprobe is short for User-level dynamic tracing, it can attach to a user-level function and run a BPF program when the function is executed. It is widely used in production, such as tracing user-space protocols: SSL, TLS, HTTP2, monitoring memory allocation and detecting leaks, tracking bash scripts, and more.

However, We've seen performance issues with Kernel Uprobe, where current implementations require two kernel context copies, leading to significant overhead. You can see that uprobe have a ten times overhead compared to the kernel kprobe.

Also, kernel Syscall tracepoints will hook all syscalls and require filter for specific process, which is not very flexible.

## Why bpftime: security and flexibility (2m)

Let's move on to the security implications of kernel eBPF. Running eBPF in the kernel demands root access, and this, naturally, enlarges the attack surface.

Now, let's glance at this chart [refer to Figure 1]. It shows a tally of eBPF-related security vulnerabilities, known as CVEs, from the past decade. Notice how the verifier, which is supposed to be the gatekeeper ensuring only safe eBPF programs run, is actually where most CVEs were found. It's a sobering reminder that complexity can breed security gaps.

And when we talk about configurability, there's a sticking point. The verifier restricts eBPF operations quite a bit. To make eBPF fully Turing-complete, which means allowing it to perform any computation given enough resources, requires changing the kernel. And adding new helpers or features? You guessed it - that needs a kernel change too.

What this all boils down to is a need for a runtime that can offer the power of eBPF without these constraints, and that's where bpftime steps in.

## [Slide 6: Current Userspace eBPF Runtime Limitations]

"Let's dive into the third reason for bpftime's inception: the limitations of current userspace eBPF runtimes. Userspace eBPF has some fantastic potential applications like observability within user programs, managing network operations, and handling configurations and plugins. But there's a catch.

The eBPF we have today in userspace can't quite keep up with all the workloads we want it to.

For instance, take Ubpft and Rbpft, both existing userspace eBPF solutions available on GitHub. They've got some neat features like ELF parsing and just-in-time compilation for specific architectures. But they fall short in some areas – they're tough to integrate with, can't use the kernel's eBPF loader or toolchains like libbpf/clang, and they lack certain types of attach support. Plus, they don't support interprocess communication or kernel maps, and they're limited to just a couple of architectures for JIT compilation.

So, in essence, while existing userspace eBPF frameworks lay down a solid foundation, they don't quite offer the broad functionality and ease of use that we're aiming for with bpftime."

## [Slide 7: Existing Non-kernel eBPF Usecases]

"Now, let's look at the innovative ways eBPF is being used outside the kernel. We're not just talking about small tweaks here and there; we're talking about some real game-changing applications.

First up, we've got Qemu+uBPF, which is like giving Qemu a superpower to understand eBP.

Then there's Oko, which gives Open vSwitch-DPDK a serious boost with BPF capabilities..

We also have Solana, which is taking smart contracts to the next level with userspace eBPF. 

For the network folks, DPDK eBPF is where it's at for fast packet processing.

And let's not overlook eBPF for Windows. Yes, Windows! Bringing eBPF toolchains and runtime over to the Windows kernel is no small feat, and it's opening doors for a whole new set of developers.

Diving into research, we see papers like Rapidpatch, which is all about quick fixes for firmware in real-time systems, and Femto-Containers, which are all about making tiny, efficient virtual spaces for IoT devices.

Put it all together, and you've got networks, plugins, edge computing, smart contracts, quick patches, and even Windows environments all benefiting from the power of eBPF. This landscape shows us just how versatile and impactful eBPF technology can be."

## [Slide 8: Bpftime - Userspace eBPF Runtime]

"Now, let's talk about bpftime itself — our userspace eBPF runtime that's all about speed and functionality.

Here's the deal: with bpftime, Uprobes. Our userspace uprobe can be spped up to 10 times faster than the traditional kernel uprobe. And just like kernel uprobe, you don't need any manual instrumentation or restart processes.

We're not just fast; bpftime is compatible with kernel eBPF toolchains and libraries, so there's no need to rewrite your eBPF apps.

Then there's the versatility. Bpftime supports interprocess and kernel maps, allowing it to work in harmony with kernel eBPF. Plus, we've got this cool feature called 'offload to userspace,' which means you can run your checks with the kernel verifier, and then the actual execution happens in userspace.

And for the tech-savvy, we've introduced a new LLVM JIT compiler for eBPF. This is for those who crave that cutting-edge performance.

So, to sum it up: bpftime brings you the speed, compatibility, and advanced features to make your eBPF experience in userspace better.

## [Slide 9: Current Support Features of bpftime]

"Let's get into the specifics of what bpftime currently supports:

On the table for bpftime's features, we've got a variety of shared memory map types that are key to userspace eBPF. We're talking about the classics here: HASH, ARRAY, RINGBUF, PERF_EVENT_ARRAY, and PERCPU varieties. These are your building blocks for creating shared data structures that are crucial for eBPF programs to communicate efficiently and store data.

Now, for the bridge between user space and kernel space, we've got shared maps too. This means that whether you're working above or below the kernel line, you can expect seamless integration and data flow. It's like having a bilingual friend who can translate on the fly in a conversation.

Moving on to probe types, which are essentially the hooks you can attach to in userspace. We cover the whole gamut from syscall tracepoints to uprobes and uretprobes. These allow you to monitor and interact with system calls or user-level functions – a bit like having a spyglass into the inner workings of your programs.

But there's more – bpftime is flexible. You're not limited to what we provide out of the box; you can define other static tracepoints and program types in userspace to suit your needs.

And for those who like the technicalities, we support 22 kernel helper functions and ensure compatibility with both kernel and userspace verifiers. 

We've even put our JIT through the wringer with bpf_conformance tests to make sure it's up to snuff.

## [Slide 10: Uprobe and Kprobe Mix: 2 Modes]

"Alright, let's break down the two modes of operation that bpftime offers for running eBPF in userspace.

Mode 1 is what I like to call the 'lone ranger' mode. It's all about running eBPF solely in userspace. This means you can use it even on systems that aren't running Linux — pretty cool, right? 

However, it's not the go-to for the heavyweight eBPF applications because the maps created in shared memory can't be utilized by kernel eBPF programs.

Then we have Mode 2, which is like a tag team. Here, bpftime works in tandem with kernel eBPF, courtesy of the bpftime-daemon. It mimics the behavior of kernel uprobes, which means it's pretty savvy at attaching to processes, whether they're just starting or already running. You get the full ensemble here: uprobes, kprobes, and even socket support.

Think of Mode 2 like a fusion restaurant — you get the best of both worlds. You don't need to modify the kernel; instead, bpftime uses the eBPF module to keep an eye on or tweak the behavior of BPF syscalls.

In a nutshell, whether you're operating solo in userspace or partnering with the kernel, bpftime has got you covered."

## [Slide 11: Examples - Monitoring Memory Allocation with bpftime]

"Let's move into the practical world and see bpftime in action with a real-life example. Say you want to keep an eye on memory allocation by monitoring the 'malloc' function in libc – that's a function in C that grabs a chunk of memory for you to use.

Here's how simple it is: First, you compile and load your eBPF program with bpftime's command line interface. Just a couple of commands and you're set.

Now, if you want to trace a program from the start, you run it with bpftime, and it'll print out messages every time 'malloc' is called, showing you the process ID. It's like having a tracking device on 'malloc' – you know exactly when and where it's being called.

But what if your program is already up and running, and you decide you want to start monitoring? No problem. You can dynamically attach bpftime to the running process. Just find out the process ID, and with a single command, bpftime jumps into action, no restart needed. It's like deciding to stream your favorite show and instantly getting the feed – smooth and hassle-free.

And there you have it, right on your screen, the output from your original program, showing how many times 'malloc' was called. It's that easy to get insights into your program's behavior with bpftime.

## [Slide 12: Examples - Detailed Monitoring with bpftime]

"Continuing with our hands-on examples, let's walk through the steps to get our bpftime daemon up and tracing those 'malloc' calls in libc.

First things first, kick off the bpftime daemon with a simple sudo command. This starts the behind-the-scenes magic that makes monitoring possible, setting up the shared memory space that bpftime will use.

Now, run the 'malloc' example. A few more commands in your terminal, and you'll load the eBPF object from the buffer. It's like loading a game cartridge into a console; you're just moments away from action.

The next part is where things get really interesting. You can trace 'malloc' calls in your target process. Fire up the example victim program, and with the help of bpftime, you'll see real-time output every time 'malloc' is called, along with the process ID. It's like having a live feed of what's happening under the hood of your application.

And there it is, right in another console window, you'll see the tally of 'malloc' calls for the target process. It's a straightforward way to get actionable insights into how your application is behaving, memory-wise.

In essence, with bpftime, monitoring crucial functions like 'malloc' becomes as easy as following a recipe. You're in control, with all the data you need at your fingertips."

## [slice 13-15]

As we delve further into bpftime's capabilities, let's focus on Mode 1, where eBPF runs entirely in userspace. This mode allows for a seamless experience when using tools like bcc and bpftrace, requiring no modifications to the tools themselves. It's like having an app on your phone that works perfectly without needing any updates.

Now, let's bridge the gap between this userspace operation and how bpftime contrasts with the kernel eBPF. Typically, eBPF toolchains like clang, bpf tool, and bpftrace are used to develop eBPF programs that run in the kernel space, requiring a series of steps including loading, verification, and JIT compilation, which ensures that the program is safe and efficient to run.

With bpftime, these eBPF programs live and run in userspace, bypassing the kernel. 

They are loaded and managed by a userspace library, leveraging a syscall interface provided by bpftime, which still goes through a verification process. 

The difference here is the final execution location. Instead of running within the kernel, bpftime allows these programs to execute in userspace, interacting with target processes through uprobes or tracepoints. This provides a layer of isolation and security, as it limits the potential impact of the eBPF program to the userspace environment only.

The visualization we see in these slides [referring to slides 14 and 15] clarifies the journey an eBPF program takes when run in userspace via bpftime.

It moves from source code through the toolchain, and then within the userspace, it's managed by bpftime components, which handle the verification and JIT compilation. Ultimately, the program can observe and interact with target processes, all within the safety and confines of userspace.

This contrast highlights bpftime's unique approach, offering the power and versatility of eBPF with the added benefits of userspace execution, such as increased safety and flexibility. 

By running in userspace, bpftime opens the door for eBPF's use in environments where modifying the kernel is not possible or desirable.

## How it works: injection

"Moving on, let's talk about how bpftime takes the concept of eBPF and applies it in a practical, user-friendly manner through the process of injection. Injection here simply means how we insert, or 'inject', the bpftime capabilities into the programs we want to monitor or manipulate.

bpftime simplifies this process by supporting two types of injection, tailored to the state of your application. If you've got a running process and you decide you want to start monitoring it with eBPF, bpftime uses the Ptrace method, which is based on the powerful Frida instrumentation toolkit.

For those situations where you're just about to start a new process, bpftime uses the LD_PRELOAD method. This is like giving your application a pair of glasses before it starts reading — it enhances the process from the get-go.

## How it Works: trampoline

"Now, let's get into the how bpftime actually gets the job done with trampoline hooking.

Here's how it looks in practice. For functions running in userspace, bpftime uses something called frida-gum. This is a toolkit that helps us to neatly insert our own code into the function we're interested in. It's like being able to add an extra step in a dance without missing a beat.

For system calls, which are the ways a program requests a service from the kernel, we use two methods: zpoline and pmem/syscall_intercept. These are different techniques for hooking into system calls efficiently.

The beauty of this approach is that it's not rigid; we can add new trampoline methods as needed. This means bpftime is adaptable and can evolve as new requirements or better methods are discovered.

The graphics here [referring to the slide] give you a clear visualization of how the trampoline code is inserted into the process's flow. 

You can see that we're not disrupting the original program; we're just adding our own layer that allows us to observe or modify behavior before passing control back to the program.

## How it Works: work with kernel

In Mode 2, bpftime showcases its versatility by blending userspace eBPF with kernel operations. This is where things get robust, allowing you to run complex observability tools like deepflow, which require a mix of userspace agility and kernel-level depth.

Picture this: bpftime in Mode 2 uses userspace eBPF for quick and safe execution while still interacting transparently with kernel eBPF. It's like having a pass to both VIP and backstage areas at a concert; you get the best of both worlds without compromise.

How does it do this? By using kernel eBPF maps, bpftime can 'offload' certain tasks to userspace, allowing for efficient processing without overloading the kernel. It's akin to cloud computing, where heavy tasks are offloaded to the cloud, keeping your local resources free.

The workflow [refer to slide 19] is intuitive. bpftime's daemon loads the eBPF program, which then hooks into the system calls or functions you're interested in, all verified and JIT-compiled. 

## benchmark llvm jit

Let's shift gears and look at another critical aspect of bpftime's performance: the JIT, or Just-In-Time compilation benchmarking. JIT is like the sprinter of the programming world—it translates eBPF bytecode to native machine code on the fly, aiming for maximum performance.

Now, the execution time graph here [referring to slide 21] shows us a lineup from various JIT implementations, including ubpf, rbpf, and LLVM's jit/aot, to wasm and native execution. And what's the takeaway? LLVM's JIT is leading the pack—it's the Usain Bolt here, potentially the fastest out there.

But, there's a catch. Sometimes LLVM can feel a bit heavy, like it's carrying extra weight on the track. That's where AOT, or Ahead-Of-Time compilation, comes in. It's like preparing for the race ahead of time, so when the starting gun goes off, you're ready to bolt.

The charts present a clear visual representation of execution times across different runtimes for various functions like prime number calculation or memory copying. They make it evident that LLVM's JIT can significantly reduce execution times, bringing it close to native performance levels.

## introduce to Evaluation & Cases

Let's explore some real-world evaluations and use cases. Existing eBPF use cases can be run without or with minor fixes, and we've tested bpftime with a variety of tools and applications, including bcc, bpftrace, and ebpf_exporter.

With bpftime, bcc tools for userspace traceing, like Bash, Memory Allocation, SSL/TLS, and tools for system call tracing, like Opensnoop, Sigsnoop, and Syscount, can be easily deployed in userspace without kernel support and without any modification to the tools themselves.

We've also put bpftime to the test with complex observability projects like Deepflow, which is a combination of userspace tracing and kernel tracing.

## bpftrace and bcc

let's turn our attention to how it enhances the utility of bpftrace and BCC, the tool sets of the eBPF ecosystem.

With bpftime, we've taken bpftrace to new heights—it can now run entirely in userspace. Imagine having a powerful telescope that works just as well from your backyard as it does mounted on a high-altitude observatory; that's bpftrace with bpftime. It can trace system calls or uprobes without leaning on kernel support.

As for BCC, the toolset is vast and versatile, covering everything from applications and runtimes to the system call interface. We've successfully brought these tools, which traditionally operated at the kernel level, into the realm of userspace. 

This transition is akin to moving from studio-based recording to live streaming; the performance is live, direct, and without the constraints of a studio setting.

We haven't just stopped at bpftrace and BCC. We've ported and tested a suite of bcc/libbpf-tools to work seamlessly with bpftime. And for those who are metrics-driven, the Prometheus ebpf_exporter is fully operational under bpftime's wing, ensuring your observability pipelines remain uninterrupted.

The visual here [referring to the slide] provides an expansive view of the eBPF tracing tools landscape, showcasing the breadth of tooling that bpftime supports, from monitoring file systems to network activity, and even CPU performance.

## sslsniff

"Let's focus on a head-to-head comparison that really illustrates the power of userspace eBPF with bpftime: Kernel versus User SSL Sniff on Nginx.

We're using sslsniff, a tool provided by the bcc suite to capture SSL/TLS data.

The impact on performance is our primary concern here. The metrics on the slide reveal a stark contrast. When kernel-based SSL Sniff is at play, the overhead is significant. We're seeing a reduction in requests per second by nearly 58% and a similar figure for the transfer rate. That's a heavy toll on performance.

Switch to userspace, and the picture changes dramatically. The overhead drops to just over 12% for both requests and transfer rates. This is like comparing a fuel-efficient car to a gas-guzzler—both get you to your destination, but one does it with much less resource consumption.

To put this into context, the benchmark uses wrk, a modern HTTP benchmarking tool, against a local Nginx server, all running on a Linux kernel version 6.2.0, with Nginx at version 1.22.0.

What we draw from this is clear: userspace SSL Sniff, enabled by bpftime, significantly lessens the performance impact compared to its kernel counterpart, offering a more efficient solution for SSL/TLS data interception in a live server environment."

## deepflow

Deepflow is another realworld and complex workload that using the power of eBPF. With over 5000 lines of kernel eBPF code, this project integrates uprobes, kprobes, sockets, and tracepoints to monitor and manage the intricacies of application behavior.

Deepflow has been battle-tested in production environments and its findings have been shared with the wider community in SIGCOMM 23, showcasing its success and robustness.

When we zoom in on Layer 7 observability, we encounter a common bottleneck: performance slowdowns due to probing. However, with bpftime's userspace uprobes, this slow down is minimized—reducing request and transfer rates by just over 15%. On the other hand, kernel uprobes exhibit a heavier impact, with reductions close to 22%.

The bar charts visually break down these performance impacts. They show us, in no uncertain terms, the advantage of userspace uprobes enabled by bpftime compared to their kernel counterparts, with a clear lesser impact on request and transfer rates.

This is a testament to the efficiency of userspace eBPF implementations. By keeping more of the observation workload in userspace, Deepflow demonstrates that it's possible to maintain a high level of observability with minimal performance sacrifice.

## Roadmap 1 and Future Enhancements

Here is the roadmap for bpftime's development.

First on the is the network domain. We're examining how userspace eBPF can interplay with DPDK, which is a set of libraries for fast packet processing. The goal? To establish a programmable network stack that enhances the performance without the need for a traditional control plane.

We're also exploring how userspace eBPF can accelerate file systems, particularly fuse — a mechanism widely used for Android and cloud storage. Imagine streamlining the filter process, making it more efficient and less resource-intensive. That's what we're targeting.

And then there's the hotpatching of userspace functions. This is a game-changer for live systems, enabling real-time updates and bug fixes without pausing or rebooting applications.

So, what's next?

## Roadmap 2

There are also more improvements on the horizon for bpftime.

Firstly, we're committed to grounding our progress in data, which is why more benchmarks and evaluations are on the docket. This will help us understand and showcase the true capabilities and enhancements of bpftime in various scenarios.

A key focus is to enhance the synergy between bpftime and kernel eBPF. By improving compatibility through expanded support for maps and helpers, we will bridge the gap between userspace and kernel, ensuring a more seamless experience.

Performance is at the heart of our endeavor. We aim to optimize the LLVM JIT and runtime, ensuring that bpftime is not just powerful, but also efficient. And for environments where resources are at a premium, LLVM AOT compilation for eBPF will be a game-changer, offering the benefits of eBPF even in the most constrained environments.

Security is non-negotiable, and we will be vigilant to ensure that the eBPF is robust against potential attacks.

Lastly, our commitment to quality is unwavering. More tests, continuous integration, and cleaner code are the pillars that will support the reliable growth of bpftime.

## Open Problems and Discussion

As we navigate the intricate path of eBPF development, particularly in bridging the gap between kernel and userspace, we also encounter a set of open problems that remains.

One of the key challenges is the BPF_F_MMAP flag's limitation to arrays. The quest is to devise a high-performance hash map that can be shared efficiently between the kernel and userspace. Could introducing new hash map types be the answer? Or perhaps we should consider overlaying a basic hash map atop the existing array map structure?

Furthermore, there’s the question of facilitating kernel eBPF programs' access to userspace maps. This opens the door to potential cache mechanisms and synchronization with syscalls, merging the speed of kernel operations with the flexibility of userspace.

Error propagation presents another conundrum: How can we design kernel eBPF processes to wait for userspace operations effectively and handle errors gracefully?

We're also pondering the implications of an unprivileged eBPF type that could democratize eBPF usage, reducing the entry barriers for developers and ensuring a broader adoption.

And finally, the security models around eBPF remain a priority. We must continue to innovate to protect against vulnerabilities and ensure that eBPF remains a robust tool for system introspection.

## Closing Slide: Conclusion and Q&A

In conclusion, we've seen how the userspace uprobes offer a tenfold speed increase over kernel uprobes, which is a significant advancement in our performance optimization efforts, and The ability to use shared memory (shm) maps and dynamically inject code into running processes without stopping.

Our tooling can be compatible with existing eBPF toolchains, libraries, and applications, ensuring a seamless integration into your current workflows.

By working together with kernel eBPF, we're not just improving performance but also expanding the scope and capabilities of what we can monitor and how we can intervene.

Thank you for your attention, and I'm looking forward to our discussion.
