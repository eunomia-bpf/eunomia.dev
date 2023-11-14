# bpftime talk

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

Let's move on to the security implications of kernel eBPF. Running eBPF in the kernel demands root access, and this, naturally, enlarges the attack surface. It's a bit like giving someone the keys to your house; there's always a risk, no matter how much you trust the locks. This risk includes nasty stuff like container escapes, where malicious code breaks out of its confined space into other parts of the system.

Now, let's glance at this chart [refer to Figure 1]. It shows a tally of eBPF-related security vulnerabilities, known as CVEs, from the past decade. Notice how the verifier, which is supposed to be the gatekeeper ensuring only safe eBPF programs run, is actually where most CVEs were found. It's a sobering reminder that complexity can breed security gaps.

And when we talk about configurability, there's a sticking point. The verifier restricts eBPF operations quite a bit. To make eBPF fully Turing-complete, which means allowing it to perform any computation given enough resources, requires changing the kernel. And adding new helpers or features? You guessed it - that needs a kernel change too.

What this all boils down to is a need for a runtime that can offer the power of eBPF without these constraints, and that's where bpftime steps in.

## [Slide 6: Current Userspace eBPF Runtime Limitations]

"Let's dive into the third reason for bpftime's inception: the limitations of current userspace eBPF runtimes. Userspace eBPF has some fantastic potential applications like observability within user programs, managing network operations, and handling configurations and plugins. But there's a catch.

The eBPF we have today in userspace can't quite keep up with all the workloads we want it to. It's like having a sports car and being told you can't take it out on the highway. You've got power under the hood, but you're stuck on side streets.

For instance, take Ubpft and Rbpft, both existing userspace eBPF solutions available on GitHub. They've got some neat features like ELF parsing and just-in-time compilation for specific architectures. But they fall short in some areas – they're tough to integrate with, can't use the kernel's eBPF loader or toolchains like libbpf/clang, and they lack certain types of attach support. Plus, they don't support interprocess communication or kernel maps, and they're limited to just a couple of architectures for JIT compilation.

So, in essence, while existing userspace eBPF frameworks lay down a solid foundation, they don't quite offer the broad functionality and ease of use that we're aiming for with bpftime."

## [Slide 7: Existing Non-kernel eBPF Usecases]

"Now, let's look at the innovative ways eBPF is being used outside the kernel. We're not just talking about small tweaks here and there; we're talking about some real game-changing applications.

First up, we've got Qemu+uBPF, which is like giving Qemu a superpower to understand eBPF. This combination allows for all kinds of new functionalities in virtualized environments. There's even a video out there showing it in action.

Then there's Oko, which gives Open vSwitch-DPDK a serious boost with BPF capabilities. It's like upgrading your toolkit; it just makes everything work better and smoother.

We also have Solana, which is taking smart contracts to the next level with userspace eBPF. Think high-speed transactions without the bloat of traditional systems.

For the network folks, DPDK eBPF is where it's at for fast packet processing. We're talking lightning-fast speeds enhanced by the flexibility of userspace eBPF.

And let's not overlook eBPF for Windows. Yes, Windows! Bringing eBPF toolchains and runtime over to the Windows kernel is no small feat, and it's opening doors for a whole new set of developers.

Diving into research, we see papers like Rapidpatch, which is all about quick fixes for firmware in real-time systems, and Femto-Containers, which are all about making tiny, efficient virtual spaces for IoT devices.

Put it all together, and you've got networks, plugins, edge computing, smart contracts, quick patches, and even Windows environments all benefiting from the power of eBPF. This landscape shows us just how versatile and impactful eBPF technology can be."

## [Slide 8: Bpftime - Userspace eBPF Runtime]

"Now, let's talk about bpftime itself — our userspace eBPF runtime that's all about speed and functionality. Imagine having the agility of a cat and the power of an elephant; that's bpftime in the eBPF universe.

Here's the deal: with bpftime, we've supercharged Uprobes. Our userspace uprobe is a real speed demon, up to 10 times faster than the traditional kernel uprobe. And the best part? You can forget about the tedious manual instrumentation or needing to restart processes. It's like having a pit crew in Formula 1; everything happens so fast, you barely notice the changes.

We're not just fast; we play well with others, too. bpftime is compatible with kernel eBPF toolchains and libraries, so there's no need to rewrite your eBPF apps. It's like having a universal remote for all your gadgets.

Then there's the versatility. Bpftime supports interprocess and kernel maps, allowing it to work in harmony with kernel eBPF. It's like a duet where both singers are in perfect harmony. Plus, we've got this cool feature called 'offload to userspace,' which means you can run your checks with the kernel verifier for that extra peace of mind.

And for the tech-savvy, we've introduced a new LLVM JIT compiler for eBPF. This is for those who crave that cutting-edge performance.

So, to sum it up: bpftime brings you the speed, compatibility, and advanced features to make your eBPF experience in userspace not just better, but rather exceptional."

## [Slide 9: Current Support Features of bpftime]

"Let's drill down into the specifics of what bpftime currently supports, and I promise to keep it as straightforward as possible.

On the table for bpftime's features, we've got a variety of shared memory map types that are key to userspace eBPF. We're talking about the classics here: HASH, ARRAY, RINGBUF, PERF_EVENT_ARRAY, and PERCPU varieties. These are your building blocks for creating shared data structures that are crucial for eBPF programs to communicate efficiently and store data.

Now, for the bridge between user space and kernel space, we've got shared maps too. This means that whether you're working above or below the kernel line, you can expect seamless integration and data flow. It's like having a bilingual friend who can translate on the fly in a conversation.

Moving on to probe types, which are essentially the hooks you can attach to in userspace. We cover the whole gamut from syscall tracepoints to uprobes and uretprobes. These allow you to monitor and interact with system calls or user-level functions – a bit like having a spyglass into the inner workings of your programs.

But there's more – bpftime is flexible. You're not limited to what we provide out of the box; you can define other static tracepoints and program types in userspace to suit your needs.

And for those who like the technicalities, we support 22 kernel helper functions and ensure compatibility with both kernel and userspace verifiers. We've even put our JIT through the wringer with bpf_conformance tests to make sure it's up to snuff.

In a nutshell, bpftime is equipped with a robust set of features to make sure your userspace eBPF experience is as powerful and versatile as it gets."

## [Slide 10: Uprobe and Kprobe Mix: 2 Modes]

"Alright, let's break down the two modes of operation that bpftime offers for running eBPF in userspace.

Mode 1 is what I like to call the 'lone ranger' mode. It's all about running eBPF solely in userspace. This means you can use it even on systems that aren't running Linux — pretty cool, right? However, it's not the go-to for the heavyweight eBPF applications because the maps created in shared memory can't be utilized by kernel eBPF programs.

Then we have Mode 2, which is like a tag team. Here, bpftime works in tandem with kernel eBPF, courtesy of the bpftime-daemon. It mimics the behavior of kernel uprobes, which means it's pretty savvy at attaching to processes, whether they're just starting or already running. You get the full ensemble here: uprobes, kprobes, and even socket support.

Think of Mode 2 like a fusion restaurant — you get the best of both worlds. You don't need to modify the kernel; instead, bpftime uses the eBPF module to keep an eye on or tweak the behavior of BPF syscalls.

In a nutshell, whether you're operating solo in userspace or partnering with the kernel, bpftime has got you covered."

## 

## How bpftime Works

Delving into how bpftime works, we've adopted a two-mode approach. Mode 1 allows running eBPF entirely in userspace, ideal for non-Linux systems or where kernel modifications are not feasible. Mode 2, on the other hand, enables bpftime to work alongside kernel eBPF, offering a blend of userspace efficiency and kernel-level capabilities.

## Benchmarks and Examples

"Our benchmarks are truly promising. In terms of Uprobe performance, userspace implementation in bpftime can be up to 10 times faster than kernel Uprobe. We've also successfully run and tested tools like bcc, bpftrace, and ebpf_exporter with minor or no modifications."

## Roadmap and Future Enhancements

"Looking ahead, we're focused on expanding bpftime's capabilities. This includes more benchmarks, enhanced kernel eBPF compatibility, performance optimizations, and ensuring robust security models. We're also exploring exciting new use cases like programmable userspace network stacks and hotpatching functions."

## Open Problems and Discussion

However, we still face challenges, such as improving hash map performance and error propagation between user and kernel space. We're eager to hear your thoughts, ideas, and suggestions on these topics.

## Closing Slide: Conclusion and Q&A

"In conclusion, bpftime represents a significant step forward in the eBPF ecosystem, offering speed, compatibility, and flexibility. It opens up new possibilities for userspace applications and bridges some critical gaps in the current eBPF landscape. Thank you for your attention. I'm now open to any questions or comments you may have."
