# Speech Script for OSDI 2025 bpftime talk "Extending Applications Safely and Efficiently"

The paper is https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng

> Abstract: This paper presents the Extension Interface Model (EIM) and bpftime, which together enable safer and more efficient extension of userspace applications than the current state-of-the-art. EIM is a new model that treats each required feature of an extension as a resource, including concrete hardware resources (e.g., memory) and abstract ones (e.g., the ability to invoke a function from the extended application). An extension manager, i.e., the person who manages a deployment, uses EIM to specify only the resources an extension needs to perform its task. bpftime is a new extension framework that enforces an EIM specification. Compared to prior systems, bpftime is efficient because it uses extended Berkeley Packet Filter (eBPF)-style verification, hardware-supported isolation features (e.g., Intel MPK), and dynamic binary rewriting. Moreover, bpftime is easy to adopt into existing workflows since it is compatible with the current eBPF ecosystem. We demonstrate the usefulness of EIM and bpftime across 6 use cases that improve security, monitor and enhance performance, and explore configuration trade-offs.

≈ 2,250 words, ~15 minutes

## [Slide 0]

Good morning! I'm **Yusheng Zheng** from UC Santa Cruz, presenting our work on a safe and efficient userspace extension.

## [Slide 1] Extensions Everywhere

Extensions are everywhere in modern software.

Extensions allow us to customize software to a specific deployment's needs without modifying the original application source code, for adding new functionality, modifying behavior, or integrating with external systems. 

We use extensions instead of modifying the source code because it is easier to maintain and update our systems.

## [Slide 2] Nginx Extension Example

Take Nginx as an example. 

Different Nginx deployments need different extensions: some need firewalls to block malicious requests, others need load balancers to distribute traffic, and many need monitoring for observability.

Consider a user wants to deploy a firewall extension to block malicious requests.

Here's how the firewall extension execution model works in Nginx. 

Before deployment, the user writes firewall logic using Nginx APIs and associates the firewall with request processing extension entries. 

During runtime, when Nginx reaches a request processing entry, it jumps to the firewall extension. 

The firewall executes in the extension runtime execution context, analyzing the request and deciding whether to block or allow it. 

Once the firewall completes its security checks, execution returns to Nginx's core processing.

## [Slide 3] Extension Problems & Requirements

Extension frameworks need three key features, as shown in the figure.

First, fine-grained safety/ interconnectedness trade-offs. 

Extensions need to interact with applications by reading data and calling functions. 

For example, a firewall extension needs to read request headers and decide whether to block or allow the request, while a monitoring extension need read-only access. 

However, allowing too much interconnecteness comes with risks: real world incidents describe production outages from safety violations in nginx, apache, and redis extensions.  

So, system managers want to specify safety/interconnectedness tradeoffs that allow only the minimum permissions each extension needs. 

Second, extension frameworks need isolation to protect extensions from application bugs.  We use extensions to provide security guarantees, so isolation is paramount for correctness.

Third, efficiency.  Extensions often run on critical paths where every millisecond matters for user experience, so we need them to be quick.

## [Slide 5] State-of-the-Art Falls Short

Unfortunately, existing approaches cannot satisfy all requirements simultaneously. Dynamic loading achieves speed but provides no isolation or  fine-
grained safety-interconnectedness policies. Software Fault Isolation systems like WebAssembly deliver safety but carry 10–15 percent performance penalties. Subprocess isolation ensures separation but has higher IPC overhead. eBPF in kernel can also be used for userspace extensions, but Kernel eBPF uprobes offer isolation but trap into the kernel on every invocation, make it less efficiency.

## [Slide 5.5] Contribution

We present a two-part solution that addresses all three requirements. First, to help navigate fine-grained safety/ interconnectedness trade-offs, we create the Extension Interface Model, or EIM. EIM treats every extension capability as a named resource and uses a two-stage specification approach.

Second, We created bpftime, a new userspace eBPF runtime that provides efficient support for EIM and isolation. It uses three key techniques: eBPF verification for zero runtime safety checks, MPK for isolation, and concealed extension entries that eliminate performance overhead. Together, they provide safety with efficiency while maintaining eBPF compatibility with kernel and existing eBPF tools.

Our evaluation on six real-world applications shows bpftime achieves all three requirements: fine-grained safety controls, strong isolation, and efficiency with up to 6 times better performance compared to solutions like WebAssembly.

## [Slide 6] Outline and transition to EIM

Now that we've discussed the motivation of the problems and challenges, let me explain the Extension Interface Model to enable fine-grained safety/interconnectedness trade-offs.

## [Slide 7] EIM: Extension Interface Model

The goal of EIM is to enable fine-grained trade-offs between safety and interconnectedness.

This is challenging because safety/interconnectedness trade-offs are a per-deployment decision that depend upon what the person who deploys the system wants to enable. But, the person deploying the system is not a developer of the system, so they lack application expertise to know what extension features should be allowed by the application. 

EIM's solution to this challenge is a two-phase specification. We model all resources as capabilities and split the process into a development phase and a deployment phase. 

It works like this.

During Development, the application developer, who understands the host application's internals, defines the possible interaction points for extensions. They create a Development-Time EIM Spec, which lists all the functions an extension could call or data it could access, like a menu of capabilities.

Before Deployment, the extension manager, who understands the specific needs of a deployment, is responsible for security and configuration. They review the capabilities offered and create a Deployment-Time EIM Spec. This spec grants the minimal set of privileges an extension actually needs to do its job, following the principle of least privilege. They are choosing from the menu created by the developer.

Finally, when the extension is deployed, the Extension Runtime uses the deployment spec to verify that the extension only performs allowed operations. This ensures that policies are enforced at runtime.

## [Slide 8] EIM Development-Time Specification

Now let me show you how EIM works in practice, using Nginx as an example. 

During development time, Nginx developers annotate their code to declare what extensions could possibly do. They can add a state capability called `readPid` for accessing the process ID, a function capability `nginxTime()` for getting timestamps, complete with pre- and post-conditions, and extension entries like `processBegin` when request processing starts.

These annotations are automatically extracted and compiled into the binary. This happens once during development and creates a complete map of what extensions could ever access. The key insight is that developers only declare possibilities—they don't decide what actually gets used.


## [Slide 9] EIM Deployment-Time Specification

At deployment time, the extension manager writes simple YAML policies that grant minimal privileges to each extension, enabling fine-grained safety-interconnectedness trade-offs.

For example, an observability extension might only read request data and call logging functions, while a firewall extension needs both read and write access to modify responses.

These policies live completely outside the application code. You can adjust the safety-interconnectedness balance per extension without recompiling. The policy is typically compact, like in 30 lines of YAML.

## [Slide 9.5] Transition to bpftime

In the next part, I will introduce our new userspace eBPF extension framework called bpftime.

## [Slide 10] bpftime: userspace eBPF extension framework

Our goal is to efficiently support EIM and isolation for userspace extensions. 

Now you might ask, "Can't we just use existing frameworks to enforce EIM policies?" Unfortunately, as we discussed in the previous work, current frameworks  use heavyweight techniques for safety and isolation, which introduces significant performance overhead.

So, our solution is a new design that exploits verification, binary rewriting, and hardware features to enable efficient intra-process extensions. We are using eBPF here because it provides verification based safety and a rich ecosystem we can reuse.

Let me walk you through how we achieve this with the bpftime framework.

An eBPF application same as kernel eBPF Application, which acts as our extension, is first passed to the bpftime Loader. This loader includes a verifier that performs (1) verification for efficient EIM support. By checking the eBPF code against the specifications before execution, we can enforce our EIM policies with zero runtime overhead.

To maintain isolation between the host and the extension, we use (2) hardware features like Intel Memory Protection Keys. This prevents extensions from accessing unauthorized memory and protects the host application from buggy or malicious extensions, and vice-versa.

For invoking extensions, the (3) conceal extension entries using binary rewriting for efficiency. Instead of heavyweight hooks, we patch the host application with lightweight trampolines that redirect execution to the extension at the right moment. This is similar to how kernel eBPF uprobes work but entirely in userspace.

As you can see, the Host application and the extension run in the same process, which enables (4) intra-process extensions for efficiency, eliminating cross-process communication costs.

Finally, our entire framework is (5) compatible with eBPF. Existing eBPF applications and toolchains work with bpftime out of the box. This compatibility also allows user-space extensions to communicate with kernel-space eBPF programs, enabling powerful, full-system observability and control.

## [Slide 10.5] transition to real-world use cases

Now, let me show you how bpftime works in practice, and our evaluation results.

## [Slide 11] Real-World Use Cases

bpftime is open source on GitHub with an active community, with thousands of stars and many users.

We built six real-world applications to demonstrate how people are actually using it. 

First, we use extensions to customize applications, including an Nginx firewall, a Redis optimization, and a FUSE cache. Second, we use extensions for observability, by adapting existing tools like DeepFlow, syscount, and sslsniff.

You can find more details in our paper, but due to time limits, I will focus on just two examples.

## [Slide 12] Customization: Nginx Firewall

We implement an Nginx firewall with EIM and bpftime and compare it with different extension approaches. In this diagram, higher is better. Lua and WebAssembly extensions impose 11–12 percent throughput loss—that's significant overhead that many operators can't accept in production. Plus, these older methods don't let you control safety and interconnectedness trade-offs like EIM does. bpftime achieves only 2 percent overhead. That's a 5× to 6× improvement over existing approaches.

## [Slide 13] Performance Results: SSL Monitoring

Then we use bpftime for sslsniff, which monitors encrypted TLS traffic, as part of the bcc eBPF tools and compare to kernel eBPF.

We run sslsniff on an nginx deployment to test different data sizes from 1K to 256K bytes. The y-axis shows throughput, where higher is better. You can see the performance difference is most clear with 1K data size.

With kernel eBPF, this monitoring costs 28 percent throughput loss. That's too much overhead for production use. With bpftime, the same monitoring functionality costs only 7 percent overhead.

## [Slide 14] Take-Aways

In summary, our contributions are EIM and bpftime. EIM enables fine-grained safety/interconnectedness trade-offs that allow you to specify least-privilege policies without touching application source code. And bpftime supports efficient safety/interconnectedness tradeoffs and isolation with near-native performance using offline verification, Intel Memory Protection Keys, and concealed trampolines. 

## Thank You & Questions (On Outline, not separate slide)

**bpftime** is open-source under the MIT license at GitHub. You can get started today by running it as a drop-in replacement for eBPF applications. We welcome your issues, pull requests, and collaboration.  

I'm happy to take your questions.

## Deprecation notes

---




<!-- 
Looking ahead, we're expanding bpftime to support GPU and ML workloads, broadening the scope of safe, efficient extension deployment beyond traditional systems programming.

However, current bpftime and EIM still have some limitations. First, EIM tools and policies are mainly for compiled applications, and we are working on supporting more languages. Also, you need to write the extension code in eBPF, which is not easy for some users. -->



<!-- 
Let me explain this using our Nginx example. In the extension ecosystem, we have 2 key roles. First, Nginx application developers write the core web server code. Second, extension developers create plugins like firewalls, load balancers, and monitoring tools. Third, the extension manager—typically a system administrator or DevOps engineer—decides which extensions to deploy and what privileges each should have. Finally, end users send HTTP requests that trigger both the host application and extensions. -->

<!-- 
## [Slide 10.5] bpftime: userspace eBPF extension framework

But ensuring eBPF compatibility and efficiency presented a major challenge. The Linux eBPF ecosystem consists of tightly coupled components, like compilers, runtime libraries, and the kernel, they are nearly impossible to disentangle. Prior user-level eBPF systems tried re-implementing the entire eBPF technology stack and ultimately failed to provide reasonable performance and compatibility.

Instead, bpftime takes a different approach. We identify a narrow waist in the current eBPF ecosystem and interpose at that point. Specifically, we intercept eBPF-related system call, and use the shared map mechanism for data sharing between extensions, applications, and kernel eBPF. This lets us reuse the proven eBPF ecosystem while adding just the minimal new components needed for userspace deployment.

So, Here's how bpftime works at a high level. We intercept eBPF application system calls before they reach the kernel. Our loader converts EIM policies into bytecode assertions and feeds everything through the kernel's proven eBPF verifier for safety guarantees. After JIT compilation to native code, we use dynamic binary rewriting to patch trampolines into the target application only when extensions are actually loaded. At runtime, we use memory protection keys to protect and execute the extension.
 -->
