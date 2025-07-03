# Speech Script for OSDI 2025 bpftime talk

≈ 2,250 words, ~15 minutes

## [Slide 0]

Good morning! I'm **Yusheng Zheng** from UC Santa Cruz, presenting our OSDI '25 paper, which is working on extending systems and extension frameworks.

## [Slide 1] Extensions Everywhere

Extensions are everywhere in modern software. PostgreSQL uses over 100 extensions like PostGIS and TimescaleDB, VS Code transforms from a text editor into a full IDE through language servers and debuggers, and Nginx gains features through modules—all without changing core application code.

What are extensions? Extensions are a way to customize software without modifying the original application source code. They allow developers to add new functionality, modify behavior, or integrate with external systems while keeping the core application unchanged.

Why do we need them? Different deployments have different needs. Some applications need security features like firewalls, others need performance optimizations like caching, and many need observability tools for monitoring. So, extensions enable each deployment to add only the functionality it needs while preserving maintainability and allowing different teams to develop features independently.

## [Slide 2] Nginx Extension Example

Take Nginx as an example. Different Nginx deployments need different extensions: some need firewalls to block malicious requests, others need load balancers to distribute traffic, and many need monitoring for observability.

Consider a user wants to deploy a firewall extension to block malicious requests.

Here's how the firewall extension execution model works in Nginx. Before deployment, the user writes firewall logic using Nginx APIs and associates the firewall with request processing extension entries. During runtime, when Nginx reaches a request processing entry, it jumps to the firewall extension. The firewall executes in the extension runtime execution context, analyzing the request and deciding whether to block or allow it. Once the firewall completes its security checks, execution returns to Nginx's core processing.

## [Slide 3] Extension Problems & Requirements

However, extension systems face serious safety and performance challenges. Real-world incidents show the risks: In 2023, a broken Lua plugin created an endless loop inside a web server, bringing down an entire content delivery network for hours. Similar issues happen with other software—database scripts can allow remote attacks, and web server modules crash from memory bugs. These examples show that even well-tested extension systems can break production services.

Meanwhile, many people use sandbox technology like WebAssembly for safer extensions, but these approaches slow down request processing by 10–15 percent—that's too much overhead for production use. People often turn off these safer extension systems because they're too slow.

Extension frameworks also need three key features, as shown in the figure, the pink box is the challenge, the green box is the requirement.

First, fine-grained safety and interconnectedness trade-offs. Extensions need to interact with applications by reading data and calling functions, but system managers want to give only the minimum permissions each extension needs. For example, a firewall extension needs to read request headers and decide whether to block or allow the request, while a monitoring extension may need reading only access to the entire request.
Second, isolation to protect extensions from application bugs and protect applications from extension bugs. Third, efficiency with fast execution, since extensions often run on critical paths where every millisecond matters for user experience.

## [Slide 5] State-of-the-Art Falls Short

Unfortunately, existing approaches cannot satisfy all requirements simultaneously. Dynamic loading achieves speed but provides no isolation or  fine-
grained safety-interconnectedness policies. Software Fault Isolation systems like WebAssembly deliver safety but carry 10–15 percent performance penalties. Subprocess isolation ensures separation but has higher IPC overhead. eBPF in kernel can also be used for userspace extensions, but Kernel eBPF uprobes offer isolation but trap into the kernel on every invocation, make it less efficiency.

> include citations and papers in the slide 

## [Slide 5.5] Contribution

We present a two-part solution that addresses all three requirements. First, to help navigate fine-grained safety and interconnectedness trade-offs, we create the Extension Interface Model, or EIM. EIM treats every extension capability as a named resource and uses a two-stage specification approach.

Second, We created bpftime, a new userspace eBPF runtime that provides efficient support for EIM and isolation. It uses three key techniques: offline eBPF verification for zero runtime safety checks, Memory Protection Keys for isolation, and concealed extension entries that eliminate performance overhead. Together, they provide safety with efficiency while maintaining eBPF compatibility with kernel and existing eBPF tools.

Our evaluation on six real-world applications shows bpftime achieves all three requirements: fine-grained safety controls, strong isolation, and efficiency with up to 6 times better performance compared to solutions like WebAssembly.

## [Slide 6] Outline and transition to EIM

Now that we've discussed the motivation of the problems and challenges, let me outline our two-part solution and walk you through how we'll address these problems. I'll first explain our Extension Interface Model for fine-grained policy specification, then demonstrate how bpftime efficiently enforces these policies while maintaining eBPF compatibility.

## [Slide 7] EIM: Extension Interface Model

So, how do we solve these problems? Our solution is the Extension Interface Model, or EIM.

The goal of EIM is to enable fine-grained trade-offs between safety and interconnectedness. But here's the key challenge: supporting these trade-offs on a per-deployment basis. 

Let me explain why this is challenging. Safety and interconnectedness trade-offs are a per-deployment decision. It depends on what the person who deploys the system wants to enable. For example, one deployment might prioritize maximum security and limit extension capabilities, while another deployment might need extensions to have broader access to perform their tasks effectively.

However, the person deploying the system lacks application expertise to know what extension features could be allowed by the application. They understand their security requirements and deployment needs, but they don't know the internal workings of the host application well enough to make informed decisions about what capabilities should be available.

EIM's solution to this challenge is a two-phase specification. We model all resources as capabilities and split the process into a development phase and a deployment phase.

Here is a simplified version of our model in the paper. As you can see in the diagram, there are two main roles: the Application Developer and the Extension Manager.

During Development, the application developer, who understands the host application's internals, defines the possible interaction points for extensions. They create a Development-Time EIM Spec for "What's possible". It lists all the functions an extension could call or data it could access, like a menu of capabilities.

Before Deployment, the extension manager, who understands the specific needs of a deployment, is responsible for security and configuration. They review the capabilities offered and create a Deployment-Time EIM Spec. This spec grants the minimal set of privileges an extension actually needs to do its job, following the principle of least privilege. They are choosing from the menu created by the developer.

Finally, when the extension is deployed, the Extension Runtime uses the deployment spec to verify that the extension only performs allowed operations. This ensures that policies are enforced at runtime.

This two-phase approach separates the declaration of capabilities from the granting of privileges. This allows for flexible, per-deployment policies without modifying the original application code.

## [Slide 8] EIM Development-Time Specification

> Let us use nginx as an example....

Now let me show you how EIM works in practice. During development time, Nginx developers annotate their code to declare what extensions could possibly do. They might add a state capability called `readPid` for accessing the process ID, a function capability `nginxTime()` for getting timestamps, complete with pre- and post-conditions, and extension entries like `processBegin` when request processing starts.

These annotations are automatically extracted and compiled into the binary. This happens once during development and creates a complete map of what extensions could ever access. The key insight is that developers only declare possibilities—they don't decide what actually gets used.

> white background for the code

> keep only the developer, and make the figure around host application.

## [Slide 9] EIM Deployment-Time Specification

> more around fine-grained safety-interconnectedness trade-offs

At deployment time, the system administrator writes simple policies that grant minimal privileges to each extension. 

An observability extension might only read request data and call logging functions. A firewall extension gets both read and write access to modify responses. A load balancer needs network capabilities to contact upstream servers.

These policies live completely outside the application code. You can refine security settings in production without recompiling anything. This separation enables true least-privilege deployment while keeping the original application unchanged.

> use the same figure 

## [Slide 9.5] Transition to bpftime

Now you might ask, "Can't we just use existing frameworks to enforce EIM policies?" Unfortunately, as we discussed in the previous work, current frameworks make painful trade-offs that prevent efficient EIM enforcement. So, we built a new userspace eBPF extension framework called bpftime.

## [Slide 10] bpftime: userspace eBPF extension framework

We built bpftime specifically to enforce EIM efficiently while maintaining complete eBPF compatibility. Why we are using eBPF? It provides proven safety through verification and a rich ecosystem we can reuse. Our efficiency comes from binary rewriting with concealed extension entries, which is similar to eBPF Uprobes, and we achieve isolation using Intel Memory Protection Keys. This compatibility is crucial—existing eBPF tools work immediately with bpftime, and extensions can share data with kernel eBPF programs for full system customization, from user-level to kernel-level.

> add popup or diagram near the verifier, to  

## [Slide 10.75] transition to real-world use cases

Now, let me show you how bpftime works in practice, and our evaluation results.

## [Slide 11] Real-World Use Cases

bpftime is open source on GitHub with an active community, and these applications demonstrate both the versatility of our approach and real-world user adoption.
... we design the 6 useca
we built six real-world applications. For security, we created an Nginx firewall that blocks malicious URLs in real time. For reliability, we built a Redis extension that bridges the durability gap between losing thousands of writes versus taking a 6× performance hit. For performance, we accelerated FUSE file operations with in-process caching. For observability, we ported existing tools like DeepFlow, syscount, and sslsniff to demonstrate seamless eBPF compatibility.

For more detail, you can check the paper.... because of time limit, I will not go into the details...

we will only look at ....

## [Slide 12] Customization: Nginx Firewall

Let me show you the performance impact for customization purposes plugins. For our Nginx firewall, we compared different extension approaches under a realistic workload. In this diagram, the more to the top, the higher throughput, the better. Lua and WebAssembly extensions impose 11–12 percent throughput loss—that's significant overhead that many operators can't accept in production. Our bpftime implementation achieves the same security functionality with only 2 percent overhead. That's a 5× to 6× improvement over existing approaches.

> add lua, wasm not proviing safety/interconnectedness trade-offs...

> add a axis for better direction.
> y axis request per second 
> put bpftime to the right

> label and group them no provideing xxx, 

## [Slide 13] Performance Results: SSL Monitoring

Beyond web servers, let's look at observability. Consider sslsniff, which monitors encrypted TLS traffic—crucial for debugging production microservices. 
> we run sslsniff on nginx deployment...
> have more reference for what the things are
> label the y axis...
 The figure shows a clear performance comparison across different data sizes from 1K to 256K bytes. Also, the more to the top, the better. With kernel eBPF, this monitoring costs 28 percent throughput loss. That's prohibitive for production use. With bpftime, the same monitoring functionality costs only 7 percent overhead. 

> add a axis for better direction.
> only have largest and smallest.

## [Slide 14] Take-Aways

Let me close with three key takeaways that address our original three requirements. First, EIM enables fine-grained safety and interconnectedness trade-offs—you can now specify precise least-privilege policies per extension entry without touching application source code. Second, bpftime provides both isolation and efficiency—we achieve hardware-level isolation with near-native performance using offline verification, Intel Memory Protection Keys, and concealed trampolines. Third, maintaining eBPF compatibility means you can adopt our approach immediately without changing your existing workflows, getting all three requirements satisfied together.

> 

<!-- 
Looking ahead, we're expanding bpftime to support GPU and ML workloads, broadening the scope of safe, efficient extension deployment beyond traditional systems programming.

However, current bpftime and EIM still have some limitations. First, EIM tools and policies are mainly for compiled applications, and we are working on supporting more languages. Also, you need to write the extension code in eBPF, which is not easy for some users. -->


## Thank You & Questions (On Outline, not separate slide)

Thank you for your attention. **bpftime** is open-source under the MIT license at GitHub. You can get started today by running it as a drop-in replacement for eBPF applications. We welcome your issues, pull requests, and collaboration. I'm happy to take your questions.

## Deprecation notes

---


<!-- 
Let me explain this using our Nginx example. In the extension ecosystem, we have 2 key roles. First, Nginx application developers write the core web server code. Second, extension developers create plugins like firewalls, load balancers, and monitoring tools. Third, the extension manager—typically a system administrator or DevOps engineer—decides which extensions to deploy and what privileges each should have. Finally, end users send HTTP requests that trigger both the host application and extensions. -->

<!-- 
## [Slide 10.5] bpftime: userspace eBPF extension framework

But ensuring eBPF compatibility and efficiency presented a major challenge. The Linux eBPF ecosystem consists of tightly coupled components, like compilers, runtime libraries, and the kernel, they are nearly impossible to disentangle. Prior user-level eBPF systems tried re-implementing the entire eBPF technology stack and ultimately failed to provide reasonable performance and compatibility.

Instead, bpftime takes a different approach. We identify a narrow waist in the current eBPF ecosystem and interpose at that point. Specifically, we intercept eBPF-related system call, and use the shared map mechanism for data sharing between extensions, applications, and kernel eBPF. This lets us reuse the proven eBPF ecosystem while adding just the minimal new components needed for userspace deployment.

So, Here's how bpftime works at a high level. We intercept eBPF application system calls before they reach the kernel. Our loader converts EIM policies into bytecode assertions and feeds everything through the kernel's proven eBPF verifier for safety guarantees. After JIT compilation to native code, we use dynamic binary rewriting to patch trampolines into the target application only when extensions are actually loaded. At runtime, we use memory protection keys to protect and execute the extension.
 -->
