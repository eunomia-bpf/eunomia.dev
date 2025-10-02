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

### **Part 2: Grant Proposal Draft - NGI0 Application Form Responses**

**Project Title:**
eBPF Offload for GPU-Native Observability and Acceleration

---

#### **Question 1: Abstract - Can you explain the whole project and its expected outcome(s). (1200 characters max)**

GPUs, as complex SIMT accelerators, pose significant observability challenges. Current host-side tracing and extension tools (like eBPF on CUDA libraries) treat the GPU as a black box, lacking insight into running kernels. Conversely, device-specific profilers (e.g., CUPTI, NVBit) are siloed from CPU-side eBPF pipelines, preventing holistic system analysis. We bridge this gap by offloading eBPF programs for native execution directly on the GPU. Leveraging our bpftime framework and an LLVM compiler, we compile eBPF to device bytecode (PTX/SPIR-V), define GPU-specific attach points (kernel launch, memory ops), and implement a GPU-aware verifier with on-device maps and helpers. This unlocks fine-grained, vendor-neutral profiling and optimization at the warp/instruction level and enables adaptive, in-kernel optimizations impossible with current tools.

Outcomes: We already have POC for it. We will deliver an open-source eBPF-on-GPU framework achieving 3-10x better performance than instrumentation tools like NVBit. The project includes signed Debian/Fedora packages, WCAG-compliant documentation, and demonstrations of advanced profiling, strengthening the EU's open toolkit for AI infrastructure.

---

#### **Question 2: Have you been involved with projects or organisations relevant to this project before? And if so, can you tell us a bit about your contributions? (2500 characters max)**

Yusheng Zheng is the creator and primary maintainer of [`bpftime`](https://github.com/eunomia-bpf/bpftime), a user-space eBPF runtime that serves as the foundational technology for this GPU offload project. The project has been published at [OSDI'25](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) as first author with the paper "Extending Applications Safely and Efficiently," and has been presented at [Linux Plumbers Conference 2023](https://lpc.events/event/17/contributions/1639/), [KubeCon Europe 2025](https://eunomia.dev/others/miscellaneous/bpftime-kubecon-draft/), and [eBPF Summit 2024](https://ebpf.io/summit-2024/). The existing `bpftime` framework provides the ideal interception layer at the host boundary (CUDA/ROCm user-space libraries) to inject our eBPF runtime. Critically, our LLVM-based eBPF compiler [`llvmbpf`](https://github.com/eunomia-bpf/llvmbpf) will be extended to target GPU bytecode like PTX and SPIR-V. We have already prototyped key components including an NVIDIA attach mechanism and initial benchmarks ([`benchmark/cuda`](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/cuda)).


Yusheng has been recognized with the "Open Source Innovation Pioneer Award" and serves as a mentor for [Google Summer of Code 2024](https://summerofcode.withgoogle.com/programs/2024/organizations/eunomia-bpf) and Open Source Promotion Plan (OSPP) 2023-2024. She has presented at [OSS Summit Europe 24](https://eunomia.dev/en/) (Panel Discussion on eBPF), [KubeCon North America 2023](https://kccncna2023.sched.com/event/1R2uf) ("eBPF + Wasm: Lightweight Observability on Steroids"), and Apsara Conference 2022, ensuring we can effectively disseminate this project's outcomes to the broader technical community.

---

#### **Question 3: Compare your own project with existing or historical efforts. E.g. what is new, more thorough or otherwise different. (4000 characters max)**

Current GPU observability and extension approaches fall into two categories with significant limitations. Host-side tools like standard eBPF, BCC, or libbpf can place uprobes/kprobe on GPU driver APIs (e.g., `libcuda.so`), but this treats the GPU device as a black box and cannot inspect behavior inside running GPU kernels, measure on-device resource contention, or trace memory access patterns between streaming multiprocessors and device memory. Vendor-specific device profilers like NVIDIA's CUPTI/NSight, AMD's ROCprof, and Intel's oneAPI tools provide deep on-device visibility but are proprietary and vendor-locked, making it difficult to correlate on-device events with broader system events captured by eBPF on the CPU. They are primarily for post-mortem analysis and lack programmability for real-time, adaptive control. Our project unifies these worlds by bringing the universal, programmable, and open eBPF programming model directly onto the GPU. For the first time, developers can use a single framework to write logic that correlates events seamlessly across the CPU, kernel, and deep inside GPU kernels. Instead of passive profiling, our solution enables programmable actions where an eBPF program running on a streaming multiprocessor could dynamically adjust memory access patterns or signal the host scheduler based on observed latency, enabling adaptive optimizations impossible today. Our approach is vendor-neutral by design, targeting common intermediate representations like PTX and SPIR-V, providing a sustainable, open-source alternative to fragmented proprietary toolchains. Early benchmarks show 3-10x better performance than instrumentation-based tools like NVBit.

---

#### **Question 4: What are significant technical challenges you expect to solve during the project, if any? (5000 characters max, optional but recommended)**

The primary technical challenges stem from adapting the CPU-centric eBPF model to GPU architecture and execution context. First, we must design a GPU-aware verifier that ensures safety by analyzing program logic against the GPU's memory and execution model. Unlike CPUs, GPUs have a complex memory hierarchy (registers, shared/LDS, L2, global memory) and a SIMT execution model based on warps/wavefronts. The verifier must prevent unsafe memory accesses within the GPU memory space and handle concurrent thread block execution to guarantee that eBPF programs cannot crash the GPU or compromise data integrity. Second, robust compilation and toolchain integration requires extending our LLVM backend to reliably compile eBPF bytecode into optimized PTX and SPIR-V. This involves mapping eBPF concepts to GPU equivalents, managing register pressure (a critical GPU performance factor), and handling different GPU architecture generations (e.g., NVIDIA's Ampere vs. Hopper). Ensuring stability across driver versions will require comprehensive continuous integration and testing matrices. Third, efficient host-device data communication is critical since eBPF relies heavily on maps for state-keeping. We must implement highly efficient eBPF maps within the GPU's memory hierarchy and create low-overhead channels (e.g., ring buffers over shared memory) to exfiltrate data from device to host for correlation with system-wide eBPF events while minimizing the observer effect. Finally, defining semantic-rich attach points requires identifying the most useful and stable hook points within GPU execution flow, beyond simple function entry/exit, to implement attach points for events like SM scheduling decisions, memory copy operations (DMA), and synchronization primitives, requiring deep reverse-engineering of GPU driver behavior and hardware capabilities.

---

#### **Question 5: Describe the ecosystem of the project, and how you will engage with relevant actors and promote the outcomes? E.g. which actors will you involve? Who should run or deploy your solution to make it a success? (2500 characters max)**

We maintain the [`eunomia-bpf`](https://github.com/eunomia-bpf) organization with several innovative projects and have created detailed [developer tutorials](https://github.com/eunomia-bpf/bpf-developer-tutorial) on eBPF programming. The entire github organization has over 7k stars.

Our ecosystem spans AI/ML infrastructure engineers, HPC researchers, and cloud-native observability tool developers across the EU. Primary actors and beneficiaries include EU-based AI startups and SMEs who need cost-effective, deep insights into GPU workloads without vendor lock-in, enabling them to optimize performance and reduce cloud computing costs. Academic and research institutions in HPC and systems can use our framework to prototype novel GPU scheduling algorithms, memory management techniques, and security models. Observability and APM vendors building next-generation monitoring tools can integrate our solution to offer unified, cross-component tracing from network sockets down to GPU warps. We will create and maintain official, signed packages for Debian and Fedora to lower barriers to entry, present findings and demonstrations at key European conferences such as FOSDEM and KubeCon EU, actively seek pilot partners from EU-based companies and research labs to validate our solution on real-world AI workloads, and engage with upstream LLVM and Mesa communities to contribute relevant parts of our toolchain. Success means European cloud providers or AI platform companies deploying our solution to provide customers with enhanced, open-source GPU observability aligned with NGI0's goal of strengthening the internet commons.

---

#### **Question 6: Explain what the requested budget will be used for? Does the project have other funding sources, both past and present? (2500 characters max)**

We are requesting €7,500. The budget breakdown is as follows: **Hardware (€2,500):** One NVIDIA RTX 5090 GPU platform for development, testing, and validation of eBPF programs running on latest-generation GPU architecture with LLM workloads, ensuring compatibility with cutting-edge AI infrastructure. **Travel & Dissemination (€2,000):** Two conference trips (€1,000 each) to present project results at major European conferences (e.g., FOSDEM, KubeCon EU) including travel, accommodation, and registration fees, ensuring effective knowledge transfer to the EU technical community. **Maintainer Support (€3,000):** Additional compensation at €500/month for 6 months to support one core maintainer dedicated to GPU offload development, covering tasks including: extending the LLVM backend to compile eBPF to PTX/SPIR-V with architecture-specific optimizations; implementing GPU-aware verifier logic for GPU memory spaces and SIMT execution constraints; developing on-device eBPF primitives (helpers, maps) for GPU memory hierarchy; creating end-to-end demonstrations (e.g., `kernelretsnoop`, `threadhist` for GPU); producing signed Debian/Fedora packages; and writing WCAG-compliant documentation with technical reports.

**Existing Funding Sources:** PLCT Lab (Institute of Software, Chinese Academy of Sciences) currently sponsors two team members at approximately €500/month each (~€12,000/year total) for general bpftime maintenance focused on RISC-V architecture support and compiler toolchain integration, start from 2022. This existing funding covers baseline runtime maintenance but does not extend to GPU-specific R&D, hardware procurement, or European community engagement activities. The NGI0 grant is crucial as it funds the dedicated hardware, travel for EU dissemination, and focused development effort required to extend eBPF to GPUs—a high-risk, innovative scope not covered by existing PLCT Lab resources.

---

#### **Work Plan and Milestones (9 Months)**

The project will deliver five sequential milestones over nine months, each with publicly verifiable deliverables on our project's public status page: Milestone 1 (Months 1-2) delivers a working eBPF-to-PTX compiler capable of handling basic profiling programs and publishes initial compiler test suite (Foundational Toolchain); Milestone 2 (Months 3-4) releases a v0.1 verifier that can safely reject programs with out-of-bounds memory accesses in GPU context (GPU Verifier Prototype); Milestone 3 (Months 5-6) implements on-device eBPF maps and helpers and demonstrates successful attachment to CUDA kernel entry/exit points (On-Device Primitives & Core Hooks); Milestone 4 (Months 7-8) releases a functional GPU thread profiler (`threadhist` example) and delivers signed v0.1 packages for Debian and Fedora (End-to-End Profiling & Packaging); Milestone 5 (Month 9) publishes WCAG-compliant documentation with tutorials for new GPU capabilities and final technical report with performance benchmarks and lessons learned (Documentation & Final Report). A public status page will be maintained with bi-monthly updates throughout this period.