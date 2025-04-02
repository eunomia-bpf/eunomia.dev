# eBPF and Wasm: Unifying Userspace Extensions With Bpftime - Yusheng Zheng, eunomia-bpf

Thursday April 3, 2025 14:15 - 14:45 BST
Level 1 | Hall Entrance S10 | Room D

## Abstract

In cloud-native systems, extending and customizing applications is key to improving development, deployment, and observability. eBPF is powerful for kernel-level enhancements, and WebAssembly brings extension to userspace. Yet, both face challenges when userspace extensions need to interact deeply with host applications. eBPF's kernel-focused design struggles in diverse userspace environments, and Wasm's sandboxing introduces overhead and complexity due to extra checks and data copying. Enter bpftime, a framework that extends eBPF's capabilities into userspace. Using dynamic binary instrumentation, bytecode verification, and hardware isolation, bpftime allows secure, high-performance extensions without the overhead of Wasm's sandboxing. This talk explores how bpftime works with the eBPF Interface to simplify userspace extensions, compares the evolution of eBPF and Wasm, and shows how bpftime can power observability, networking, and other cloud-native extensions.

## Slides and texts

### Slide 1: Title

eBPF and Wasm: Unifying Userspace Extensions With Bpftime

> Good afternoon, everyone! My name is Yusheng Zheng, I'm a PhD student at the UC santa cruz and I'm maintaining some eBPF related open source projects in eunomia-bpf comminity.
> Today, I'm going to talk about something that has been around in the software industry for a really long time—software extensions.

### Slide 2: Agenda

- Introduction to Application Extensions
- Challenges: Safety vs. Interconnectedness
- Limitations of Existing Solutions
- EIM: A Fine-Grained Interface Model
- bpftime: Efficient Runtime Design
- Use Cases & Evaluation
- Conclusion

> Specifically, I want to talk about why we need extensions, what makes them challenging to handle correctly, and how our current approaches to managing extensions might not be good enough. And then I'll introduce a new approach to managing extensions called the Extension Interface Model (EIM) and our experimental userspace eBPF runtime called bpftime. This is also a research project that has been accepted by the OSDI 2025.

### Slide 3

So, first, let's step back a little bit. Software extensions aren't new—they have a very long history. If you've been around software for a while, you might remember Apache HTTP Server modules, browser plugins, or even IDE extensions like VSCode plugins. Extensions are everywhere because we, as engineers, really love flexibility. We love building a core application and then letting other developers or even users customize or add extra features later, without needing to rewrite the original software.

Think about web servers. In nginx or Apache, you can use modules or plugins to add extra functionality like authentication, caching, compression, or even application-level firewalls. Database systems like Redis and PostgreSQL have extensions to support new query types, custom data formats, or security audits. Our editors—Vim, Emacs, VSCode—all thrive thanks to the flexibility offered by extensions or plugins. In cloud-native systems like Kubernetes, you can use extensions to add observability features like custom metrics collection and tracing. You can extend networking with custom CNI plugins or service mesh sidecars. Security extensions can add policy enforcement, vulnerability scanning, or runtime threat detection. The flexibility of extensions is what makes Kubernetes such a powerful platform for building modern applications. In kernel level, for example, you can use eBPF programs or kernel modules to add custom kernel behaviors.

### Slide 4

But here's a big question: why don't we just integrate everything into the main codebase directly? Why bother with extensions at all?

The short answer is—flexibility and isolation.

We want flexibility because it makes our software adaptable. Users and administrators want to tweak things to meet their specific requirements without waiting for the core developers to implement changes. But flexibility without isolation is risky. Extensions, by definition, are third-party or at least externally-developed code. You might trust your core engineering team, but trusting external code is a different story. Even if it's not malicious, external code can have bugs. And we all know how easily bugs can creep into our systems, causing all sorts of problems—crashes, performance degradation, or even security vulnerabilities.

### Slide 5

Let me give you some real-world examples. A few years back, the popular video streaming site Bilibili suffered a serious production outage because one of their nginx extensions got stuck in an infinite loop. Apache HTTP Server had similar issues where buffer overflow bugs in a Lua-based module caused crashes and security holes. Even Redis had cases where improperly sandboxed Lua scripts resulted in remote code execution vulnerabilities. These aren't theoretical risks—these are things that have actually hurt big companies and cost serious money and reputation.

So, isolation and safety become absolutely critical. We don't want a bug in one extension to crash our entire system. We don't want a poorly-written plugin causing our service to slow down. We definitely don't want external code exposing our internal data to attackers.

But achieving isolation isn't easy. There's always a tradeoff. If you isolate too strictly, you lose expressiveness—extensions become so limited they're no longer useful. If you're too loose, bugs or malicious code can harm your system.

### Slide 6

So, how have engineers traditionally tried to deal with this balance?

We've tried many things. Early on, we used dynamically loadable modules—shared libraries, DLLs, LD_PRELOAD hacks. They're great for speed and flexibility. But they have virtually no isolation. A bug in a dynamically-loaded module is a bug in your entire application. There's no safety boundary.

Later, people adopted scripting languages like Lua and Python for extensions. Languages like Lua are still hugely popular because they're easy to integrate and relatively safe. Redis and nginx are good examples here. But Lua scripts have their own challenges. They rely heavily on the host application to provide security boundaries, which means if the host makes a mistake—maybe not checking array bounds correctly or missing resource limits—the safety promise falls apart. And as we've seen from real incidents, host applications frequently get this wrong.

A lot of extensions, especially in the cloud-native space, are integrated through process level isolation, such as subprocesses or API calls. They uses OS context switches, adding microseconds of overhead—which is OK for distributed systems, but not for performance-critical applications.

Wasm and eBPF are two popular extension frameworks that have been around for a while. Let's take a look at how they handle the balance between interconnectedness and safety.

### Slide 7 wasm

Then WebAssembly (or Wasm) came along, promising much better isolation and performance. Wasm uses software fault isolation techniques, which means it's safer because it doesn't blindly trust the extension code. That's why many modern browsers and even server-side applications are moving towards Wasm. But Wasm introduces another issue—the interfaces between extensions and hosts. Wasm needs explicit import/export mechanisms to talk to the host application, and managing this communication can be tricky. It's powerful, but it's still heavyweight, especially if you're running thousands of little extensions or making frequent calls back and forth.

### Slide 8 ebpf

At the kernel level, eBPF has become the star of the show. Originally designed for network packet filtering, eBPF is now widely used for security monitoring, observability, and even performance optimization at kernel level. eBPF programs run isolated from the kernel itself, thanks to a verifier and JIT compiler that ensures safe execution. But eBPF is not limited to kernel, for userspace tracing, eBPF leverages mechanisms like uprobes and USDT probes. These probes eliminate the need for manual instrumentation - developers don't need to modify their application code to enable tracing. Uprobes can dynamically instrument any function in userspace programs, while USDT provides static tracepoints that are more efficient but require compile-time integration. This automatic instrumentation capability makes eBPF powerful for observability without requiring changes to the target application.

Compare with Wasm, eBPF has a history of focusing on performance first, which lead to the design of using verifier to check the safety of the extension at load time, instead of checking runtime checking or SFI (software fault isolation) like Wasm.

### Slide 9 Three requirements for extension frameworks

Alright, now let me dive deeper into the core issue we're really focusing on today—this fundamental tension between **interconnectedness** and **safety**. this is what we think makes software extensions challenging, and also why getting the interface right is so hard, yet so important.

So, first off, what do I mean by interconnectedness? Simply put, interconnectedness is how much power we give an extension to interact with the host application. Think of it like this: extensions usually need to do something meaningful—they're not just isolated pieces of code floating around. They actually need to communicate with the main application to read data, modify state, or even call existing functions inside the application. For example, let's say you have an nginx web server. An extension that monitors web requests for security needs to read the request details from the host application. It might also need to call a built-in nginx function to quickly respond or block a suspicious request. This is interconnectedness—extensions working directly with the application's resources or calling its internal functions.

On the flip side, safety is how much we limit an extension's ability to interact with or alter the main application. Safety means that if there's a bug in your extension—let's face it, we all write buggy code sometimes—this bug won't crash your whole web server or compromise your entire application. If we didn't care about safety, extensions could freely do whatever they wanted—read or write any memory, call any function, open files they shouldn't, or even alter sensitive configuration data. Obviously, this would be a recipe for disaster. A single small mistake in an extension could take down a whole production system or open it up to security vulnerabilities. And trust me, that has happened many times before. Like that time Bilibili's nginx extensions got into a loop and brought their production servers offline, causing a major outage.

So here's the key issue: interconnectedness and safety are fundamentally at odds. The more interconnectedness you allow—the more you let extensions interact with the host—the less inherently safe it becomes. And vice versa: to keep things perfectly safe, you have to restrict interconnectedness. And that means extensions become severely limited in their usefulness. You can't have an effective firewall extension if you don't allow it to inspect web requests. You can't meaningfully monitor performance if your monitoring extension can't read internal state from the host application. So, balancing this tension is a core challenge.

Now, historically, software frameworks have not handled this tradeoff very well. Usually, they fall into one of two extremes. Either they allow too much interconnectedness, like dynamically loaded modules—these run fast, sure, but they provide almost no safety at all. One bug and your entire application crashes. Or, on the other extreme, they provide strong safety through heavy isolation, like sandboxed scripting environments or subprocess isolation methods. But these can cripple interconnectedness and performance—extensions often become slow and limited in what they can do.

So, what we've found is that the key to managing this tension—this interconnectedness versus safety tradeoff—is the interface we choose for extensions. If your extension framework's interface can carefully define and verify exactly which resources and functions an extension can use, you can precisely manage this tension. Ideally, you give the extension just enough interconnectedness to do its job—but absolutely no more. This sounds simple, but current systems struggle to achieve this.

### Slide 10 Wasm interface: from module to components

For instance, consider WebAssembly (Wasm). While Wasm provides strong memory isolation through its sandboxed environment, it faces challenges with interface design. The host application needs to explicitly define which functions and memory regions are accessible to Wasm modules through imports and exports. This creates a rigid boundary - either you expose too much functionality making the interface permissive and potentially unsafe, or you restrict access too much making extensions less useful. Additionally, the overhead of marshaling data between Wasm and host environments can impact performance, especially for extensions that need frequent host interactions. This "all-or-nothing" approach to function exposure, combined with the serialization costs, makes it difficult to achieve the fine-grained control needed for many real-world extension scenarios.

(Add component model related context)

### Slide 11 eBPF interface: from verifier to components

And then we have eBPF, which many of you might know and even use. Kernel-level eBPF uses a verifier—a special component—to check every extension program before running it. This verifier ensures that extensions never do unsafe things, like accessing random memory or running infinite loops. The verifier approach is powerful because it shifts safety checks from runtime (when things are happening quickly and mistakes are costly) to load-time (when you first add the extension). This is exactly why kernel-level eBPF has become popular—it guarantees safety upfront and doesn't impose runtime overhead for safety checks once the program is loaded.

But there's an important catch: user-space extensions using current eBPF features (like uprobes) don't get the full benefit of this verifier model. They're isolated in terms of memory safety, but they're expensive. Every time a user-space extension runs, it triggers a costly kernel trap—this means your application needs to pause, switch context into the kernel, run the extension, and then switch back again. That's way too slow for many modern, performance-critical scenarios.

### Slide 12: Introducing EIM - Our Verifier-Based Approach

Our approach, the Extension Interface Model or EIM, takes inspiration from eBPF's verifier-based model and Wasm's component model, but applies it at the user-space level for general extension frameworks.

Why did we choose a verifier-based approach? There are three key reasons:

1. **Load-time safety guarantees** - We verify everything before execution, eliminating runtime overhead
2. **Fine-grained control** - We can precisely tailor what each extension can access
3. **Elimination of manual safety checks** - We remove the burden from developers, preventing the bugs and security flaws that have plagued extension systems

The core insight of EIM is treating every interaction between an extension and its host as an explicit capability that can be controlled. Think of it as a permission system designed specifically for extensions.

### Slide 13: How EIM Works - The Principle of Least Privilege

Here's how EIM works in practice:

First, application developers identify and define all possible capabilities their application can safely expose - whether that's reading variables, calling functions, or accessing resources.

Then, system administrators specify exactly which of those capabilities each extension needs - and nothing more. It's like issuing carefully tailored access cards that grant precise permissions.

This solves the fundamental tension I described earlier. Instead of choosing between "full power but unsafe" or "safe but useless," EIM lets you find the perfect balance for each extension.

Let me give you a concrete example: In an nginx web server, you might have:
- An observability extension that only needs read access to request variables
- A firewall extension that needs both read access and the ability to call request-modification functions

With EIM, each gets exactly what it needs - no more, no less.

### Slide 14: The Benefits of EIM with bpftime

By combining EIM's precise interface model with our bpftime runtime, we achieve three things that were previously impossible to get simultaneously:

1. **Near-native performance** - No runtime checks means extensions run at close to native speed
2. **Guaranteed safety** - The verifier ensures extensions can't exceed their defined boundaries
3. **Precise control** - Fine-grained capabilities enable the principle of least privilege

This approach gives developers and administrators exactly what they need - a way to define extension boundaries with surgical precision. No more "all-or-nothing" access control that either compromises security or cripples functionality.

With this foundation established, let me now explain how bpftime implements and enforces these capabilities through its runtime design...

### Slide 13 bpftime: a runtime system for EIM

And bpftime complements this by efficiently enforcing these permissions. It builds on top of eBPF concepts but brings them fully into user-space. bpftime uses the same verification and safety guarantees as kernel-level eBPF, ensuring that an extension can't step outside its defined boundaries. Plus, it leverages modern hardware isolation features (like Intel's Memory Protection Keys—MPK) to give near-native performance without the heavyweight overhead of context switches or traps that traditional solutions like uprobes suffer from.

But beyond just safety and speed, bpftime also provides a dynamic and flexible loading mechanism. Extensions can be loaded at runtime, allowing administrators to quickly adapt their systems without restarts. And because it's compatible with eBPF, it also integrates seamlessly with existing kernel-level eBPF monitoring tools, bridging user-space and kernel-space observability.

This new approach isn't just theoretical. Right now, there are already thousands of users running bpftime extensions for various real-world applications. They're using it for performance profiling in microservices, fault injection testing, caching optimizations in file systems, hot-patching running applications, and even security enhancements in web servers.

For example, with bpftime, one user implemented an SSL-aware distributed tracing tool. Compared to traditional eBPF-based uprobes, their performance overhead was nearly four times lower. Another case involved improving Redis durability without significantly reducing throughput—previous approaches would have made such fine-grained tuning too costly or too difficult to safely implement.

So, as we wrap up this motivation today, what I want you to take away is simple:

Software extensions aren't going away—in fact, they're becoming more and more essential. But traditional approaches aren't cutting it anymore. They're either too risky, too limited, or too expensive. We need new, flexible interfaces and runtime systems that let us clearly and safely define exactly how extensions can interact with applications.

That's why we built EIM and bpftime. They're about giving you, the system administrators and engineers, a better way to safely and efficiently extend your software systems, without the painful compromises of the past.

This brings us to the core of what we've developed: first, the Extension Interface Model, or EIM, and second, bpftime—an efficient, practical runtime system designed specifically to enforce EIM.

Let's dive deeper into EIM first. We knew from the start that simply isolating extensions wasn't enough; we needed fine-grained control. The question we asked ourselves was simple yet powerful: "What if we treated every action an extension can perform—like reading a variable, calling a function, or writing to a state—as a resource?" With that mindset, an application's developers can enumerate clearly which resources are potentially available to extensions. Later, at deployment time, a system administrator or DevOps engineer selects exactly which of these resources are allowed per specific extension. This means, if you're deploying an observability extension, you might only allow read access to performance counters. For a security extension, you might allow inspecting incoming requests but restrict all write operations. By modeling each capability explicitly, EIM provides both developers and administrators with precision and clarity in managing what extensions can—and crucially, cannot—do.

But of course, defining precise boundaries is just half the solution. We needed a runtime environment capable of enforcing these rules efficiently. That's why we developed bpftime—an innovative, lightweight extension runtime inspired by the success of eBPF, which many of you probably already use for kernel-level extensions and observability tasks.

bpftime takes this idea to the user-space: it checks extensions at load time using advanced eBPF-style verification, meaning that all safety constraints defined by EIM are enforced before any extension ever runs in production. No runtime overhead, no expensive context switches—just fast, secure execution. To further secure the system, bpftime also integrates hardware isolation technologies, like Intel Memory Protection Keys (MPK), to ensure even accidental or rogue extensions can't tamper with critical application memory.

One of our most exciting innovations with bpftime is what we call "concealed extension entries." This idea is about efficiency. Normally, extensions introduce overhead at their insertion points—each time your application reaches an extension hook, there's a cost. But why pay that cost when no extension is actively being used at a given point? Concealed extension entries dynamically inject hooks into applications at runtime only when necessary, using safe binary rewriting techniques. This means zero overhead when extensions aren't in use, dramatically improving performance compared to traditional approaches.

Now, as industry engineers, you might ask, "How practical is all this? How easy is it to adopt?" Well, bpftime is fully compatible with the existing eBPF ecosystem. If your team already uses eBPF—for monitoring, security, or debugging—you can seamlessly integrate bpftime. You don't need to rebuild your tooling from scratch; you just extend your current setup to benefit from these powerful new capabilities.

Let me share a bit about our experimental validation. We've extensively tested bpftime across diverse practical scenarios. In a microservices observability case, bpftime improved monitoring throughput by 50% compared to traditional eBPF. For Redis, we implemented extensions that significantly enhanced durability with just a minimal performance tradeoff—something previously impractical. We even accelerated filesystem metadata operations in FUSE and dramatically reduced overhead in SSL traffic monitoring scenarios—proving that our theoretical benefits translate directly into real-world performance improvements.

Our experiments showed not only that bpftime could enforce security constraints rigorously but that it did so at speeds close to native execution. This was crucial for adoption because no matter how safe extensions might be, if they slow down your service significantly, they're not practical in a high-performance environment.

Finally, our approach isn't just theoretical—it's available today, experimentally maintained and tested by thousands of active users on GitHub. Engineers across the industry are already leveraging bpftime for tasks ranging from deep performance diagnostics to rapid security patching, all without compromising their application's integrity or responsiveness.

To conclude, our experimental work with EIM and bpftime opens new possibilities for safely extending software at runtime. It's a path forward for developers and administrators alike, enabling highly customized, performant, and secure applications.



