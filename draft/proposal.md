Of course. I have reorganized and refined the provided information into two distinct parts as you requested.

* **Part 1 is a clear "Requirements Specification"** that summarizes the rules, deadlines, and evaluation criteria for the NLnet NGI0 Commons Fund. This section serves as a practical checklist.
* **Part 2 is the polished "Grant Proposal Draft"** itself. It has been rewritten into a cohesive narrative using paragraphs, as requested. The proposal frames a larger story around "user-space OS extensions for the AI era," with `bpftime` and its AI applications as the central focus.

---

### **Part 1: NLnet/NGI0 Commons Fund Requirements Specification**

This section outlines the key requirements, constraints, and evaluation criteria for a successful application, based on the fund's guidelines.

#### **A. Application Logistics & Deadlines**
* **Submission Portal:** All applications must be submitted via the official NLnet "Apply for funding" form.
* **Current Deadline:** The current funding round closes on **October 1, 2025, at 12:00 Central European Summer Time (CEST)**.

#### **B. Funding & Budget Rules**
* **First-Time Applicant Limit:** The maximum request for a first-time project is **€50,000**. Larger proposals are only considered after successful completion of an initial project.
* **Per-Proposal Cap:** The absolute maximum funding for any single proposal is €150,000.
* **Lifetime Cap:** The total funding an individual or organization can receive from the NGI0 Commons Fund is €500,000.
* **Payment Model:** Funding is disbursed as donations upon the verified completion of pre-agreed milestones outlined in a Memorandum of Understanding (MoU).

#### **C. Mandatory Project Requirements**
* **Open Licensing:** All resulting software and hardware must be released under recognized FOSS/Open Hardware licenses. All scientific research and documentation must be Open Access.
* **European Dimension:** The project must demonstrate a clear benefit to the European internet ecosystem. For non-EU applicants, this can be shown through targeting EU-based users, packaging for common European Linux distributions (e.g., Debian, Fedora), or collaborating with EU partners.
* **Accessibility:** All user-facing outputs are expected to meet a high standard of accessibility, generally benchmarked against the Web Content Accessibility Guidelines (WCAG).

#### **D. Evaluation Criteria & Process**
* **Initial Scoring:** Proposals are scored based on:
    * **Relevance & Impact (40%)**
    * **Technical Feasibility (30%)**
    * **Value for Money (30%)**
* **Review Process:** After an initial screening, reviewers may ask clarifying questions to refine the proposal's scope, budget, and technical plan before a final decision is made.
* **AI-Generated Content:** The use of generative AI to write the proposal must be disclosed. Failure to do so may result in rejection.

#### **E. Post-Selection Obligations**
* **Public Status Page:** Grantees are required to maintain a public project status page with updates at least every two months.
* **Independent Audits:** Projects with a value over €50,000 may require an independent security audit before final payment.
* **Utilization of NGI0 Services:** Grantees are strongly encouraged to use the provided support services (security audits, accessibility reviews, packaging support, etc.) to improve project quality.

---

### **Part 2: Refined Grant Proposal Draft**
This is a polished version of your proposal, ready to be adapted for the NLnet application form.

**Project Title:**
User-space eBPF Extensions for the AI Era: `bpftime` for AI-Aware Observability and Secure OS-Level Instrumentation

**Abstract**
Modern AI applications, from LLM agents to complex inference pipelines, operate across a fragmented landscape of processes, runtimes, GPUs, and network boundaries. Traditional application-level instrumentation is often incomplete and brittle, while kernel-mode eBPF, though powerful, introduces significant friction related to permissions, verifier constraints, and system stability. Our proposal re-imagines OS-level extensibility as a first-class primitive for AI systems, delivered safely and efficiently in user-space. We present `bpftime`, a high-performance user-space eBPF runtime that provides a universal extension framework. It offers familiar hooks (Uprobe, USDT, syscalls), maps, and helpers with an LLVM JIT/AOT compiler, all without requiring kernel privileges. This enables critical AI-aware observability—such as tracing agent tool-use and model I/O at the system boundary—and secure sandboxing. We will harden the `bpftime` verifier, add GPU/XPU event sources essential for AI pipelines, deliver signed Debian/Fedora packages and OCI images, update documentation to WCAG standards, and pilot the framework on AI workloads with EU partners. All outcomes will be released under OSI-approved licenses and validated through NGI0's expert review services.

**Track Record**
The eunomia-bpf project is the founder and primary maintainer of `bpftime`, a user-space eBPF runtime designed as a universal extension framework compatible with the existing eBPF toolchain. Our public documentation and performance studies already demonstrate a user-space Uprobe path that is significantly faster than the kernel equivalent in certain scenarios, alongside unique features like programmatic syscall hooking and shared maps, all powered by our LLVM-based VM (`llvmbpf`). We have a history of creating detailed developer tutorials on topics ranging from user-space eBPF to CO-RE workflows. This "boundary observation" approach has been successfully implemented in our `AgentSight` project, which achieves zero-instrumentation observability for LLM agents by monitoring SSL/TLS and process signals. We also maintain experimental projects like `GPTtrace` for AI-assisted eBPF generation, further cementing our expertise at the intersection of OS-level tracing and AI systems.

**Comparison with Existing or Historical Efforts**
While kernel-mode eBPF remains essential for privileged, low-latency data paths, its adoption carries real costs in privilege, verifier conservatism, and operational friction. `bpftime` complements kernel eBPF by shifting many tracing and policy enforcement tasks to user-space, preserving the familiar eBPF programming model while dramatically lowering deployment risk and iteration time. Unlike ad-hoc `BCC` or `libbpf` examples, our focus is on delivering extensibility as a stable product: reliable user-space hooks, broad API compatibility, and robust packages that work across common EU distributions, lowering the barrier for SMEs and research labs. Furthermore, where proprietary APM/AI SDKs require code modification and create vendor lock-in, the `AgentSight`-style boundary tracing enabled by `bpftime` provides framework-agnostic, zero-change observability that remains fully open-source.

**Significant Technical Challenges and Our Approach to Them**
Our primary challenge is ensuring security and semantic alignment. A user-space verifier must rigorously reject unsafe programs and clearly document any intentional deviations from kernel semantics. We will mitigate this by building an extensive test corpus, adding differential testing against the kernel where feasible, publishing a formal threat model, and providing secure-by-default configurations for namespaced and seccomp-sandboxed deployments. A second challenge is the AI-centric event correlation; signals like GPU queue submissions and DMA copies must be intelligently linked with process and network context. We will implement dedicated GPU/XPU event sources and develop examples that fuse these signals with the boundary tracing techniques already proven in `AgentSight`, validating the overhead on representative AI workloads. Finally, to ensure broad adoption, we will tackle the challenge of packaging and reproducibility by providing signed Debian/Fedora packages, OCI images with SBOMs, and integrating feedback from NGI0 Review for security, licensing, and WCAG compliance.

**Ecosystem, European Dimension, and Adoption**
The primary beneficiaries of this work are the EU's small-to-medium enterprises (SMEs) and research groups who are building AI systems that require trustworthy, low-overhead observability and secure extensibility without kernel-level privileges. By officially targeting Debian/Ubuntu and Fedora for packaging and providing OCI images, we directly reduce the friction of adoption within common European development and CI/CD environments. We will actively recruit EU-based users as pilot partners, submit talks and demonstrations to European developer forums like FOSDEM, and coordinate with distribution maintainers where appropriate. The combination of open licensing, reproducible packaging, and audit-driven hardening aligns perfectly with NGI0's goal of strengthening the internet commons with sustainable, vendor-neutral infrastructure.

**Requested Support and Budget**
We are requesting **€48,000** over a nine-month period. This budget is allocated primarily to engineering efforts focused on verifier hardening, GPU/XPU event source development, and robust packaging. It also includes dedicated resources for documentation overhaul to meet WCAG standards and for managing CI/build infrastructure. The budget adheres to the €50,000 guideline for first-time proposals and is structured for milestone-based donations as per the standard NLnet MoU, with a clear daily rate and task breakdown to be provided in the application form.

**Work Plan and Milestones (9 Months)**
We propose six sequential milestones, each yielding publicly verifiable deliverables. **Months 1-2:** We will harden the verifier/loader, expand the test suite, establish a CI matrix across common distributions, and publish a v0.1 threat model. **Months 2-4:** We will implement the initial GPU/XPU event sources, providing an end-to-end example that correlates model I/O with process context. **Months 3-5:** We will produce signed Debian/Fedora packages and OCI images, complete with SBOMs and reproducible build instructions. **Months 4-6:** We will formally request an NGI0 security review, remediate all findings, and publish hardening guides. **Months 5-7:** We will execute the WCAG-focused documentation update, adding two AI-centric quick-starts for tracing agent tool-use and GPU pipelines. **Months 7-9:** We will conduct a pilot with an EU partner, measuring overhead and latency on a representative AI workload, and publish a report with slides and a recorded demonstration. A public status page will be maintained with bi-monthly updates throughout this period.