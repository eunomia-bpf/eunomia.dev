Below is a ready-to-paste set of answers for each field on the eBPF Foundation Fellows application. I tailored it to emphasize community education and hands-on materials (the core of the program), and mapped concrete deliverables to a 6-month timeline. The plan explicitly aligns with the Foundation’s stated focus on “growing and educating the eBPF community.” ([eBPF][1])

---

# Form answers

**Email**
`yunwei356@gmail.com`

---

## Please provide a brief summary of your proposed project or contribution (2–3 sentences).

I will ship a polished reboot of the open-source bpf-developer-tutorial with a focus on using libbpf/libbpf-rs/ebpf-go for all the tutorials and replacing the old eunomia-bpf framework. I will add new labs covering eBPF features such as bpf_arena and use cases like ML observability. I will also add compatibility checks for the tutorials, and provide bilingual (EN/中文) guides to help bring more first-time contributors into the ecosystem.

---

## Describe your proposed project in detail, outlining its objectives, methodology, and expected outcomes. How will this project benefit the eBPF ecosystem and community?

bpf-developer-tutorial is a CO-RE based guide for eBPF application development, offering practical examples from beginner to advanced levels using frameworks in C, Go, and Rust. Its major goal is to help developers quickly learn eBPF techniques through short, standalone tool examples rather than long theoretical explanations, and try to cover different and updated features and use cases of eBPF ecosystem. The tutorial has been maintained by me for 3 years and has received 3.6k stars on Github.

In the next six months, I plan to modernize bpf-developer-tutorial by replacing the old eunomia-bpf framework with libbpf in C, libbpf-rs in Rust, and ebpf-go in Go across the lessons. Currently the first 10 lessons are using eunomia-bpf framework, and the major of them are already using libbpf/libbpf-rs/ebpf-go. Each example will demonstrate idiomatic usage of these ecosystems, ensuring that the tutorials remain relevant and aligned with the current practices of the eBPF community. Alongside this migration, I will introduce new labs that cover recently added capabilities such as bpf\_arena, BPF token, exceptions, iterators, new kfunc and struct\_ops usage patterns, and the use of bpf\_rbtree and bpf\_list. I will also add practical machine learning observability use cases and other use cases like HID-BPF. These updates will expand the tutorial's scope into areas where many developers are experimenting but where reliable, well-structured learning material is still scarce.

The project already has basic CI checks, but to make the learning path reproducible and accessible, I will provide compatibility checks that record minimum kernel versions, required capabilities, sysctls, and make it as a preflight doctor script. Golden-output tests will be included so that each lesson can be validated with a short run, confirming that the environment is working as intended. To further lower friction, I will publish a ready-to-use devcontainer (and optionally a nix flake) so that learners have a consistent toolchain. Each lesson will include standardized documentation with objectives, requirements, quickstart instructions, expected output, and troubleshooting guidance. All content will be kept bilingual in English and Chinese, supported by lightweight linting and checks to keep translations in sync.

The work will be delivered incrementally: in the first two months I will land the doctor script, add compatibility badges, convert eight core lessons, and ship the devcontainer with golden-output tests. In the following two months I will add six to ten new labs covering new eBPF features and use cases. In the final two months I will complete the framework migration for the remaining priority lessons, refine the testing harness, finalize the bilingual documentation, and write a focused “verifier troubleshooting” guide that can be linked from all labs.

This project will make bpf-developer-tutorial a more robust and modern foundation for both newcomers and long-time practitioners. By moving away from eunomia-bpf and aligning with libbpf, libbpf-rs, and ebpf-go, the tutorials will be more compatible with mainstream usage and easier to maintain as upstream evolves. The addition of labs for features can help fill major documentation gaps and provide practical examples of features that are already shaping real-world eBPF applications. The compatibility checks and preflight tools will significantly reduce the frustration that beginners often face with environment setup, making it easier to reach a “first success” quickly. Finally, keeping the materials bilingual will extend the reach of the tutorial to a much larger community, especially first-time contributors in non-English-speaking regions. Taken together, these improvements will strengthen the educational ecosystem around eBPF, reduce barriers to entry, and accelerate adoption and innovation across the community.

---

## Please describe your past contributions to the eBPF ecosystem. This could include code contributions, documentation, community organizing, educational content, etc.

I have been maintaining the https://github.com/eunomia-bpf organization for more than 3 years, and creating and maintaining some projects including the userspace eBPF runtime bpftime, some experimental projects like combining eBPF with Wasm and AI, and the bpf-developer-tutorial. I have also given talks and workshops about eBPF and related technologies in KubeCon, eBPF Summit, Linux Plumbers Conference, and other conferences. I have also mentored students on GSOC and OSPP.


*(If helpful, I can provide a short appendix linking representative repos, slides, and recordings as supporting material.)*

---

## How do you plan to engage with the eBPF community during and after your fellowship?

During the fellowship I will engage with the community by maintaining active discussions on GitHub issues and PRs for bpf-developer-tutorial, publishing new tutorials and blog post updates, and sharing progress reports and migration notes to make adoption easier for other projects. After the fellowship, I will continue maintaining the tutorials, expanding coverage for new kernel features, and keeping the bilingual content updated so it remains accessible to both English and Chinese speakers. I also plan to keep exploring new eBPF features and use cases, sharing them through blog posts, open-source projects, and research papers to further benefit the community.


---

## Are you currently paid to work on an eBPF-related project?

**No.**
*(If this should be “Yes” due to any current RA/industry engagement touching eBPF, it can be toggled, but my fellowship work is community-oriented and not funded by a vendor.)*

---

## Please upload any supporting documents, such as a resume/CV, portfolio, or examples of your work. (Upload one file, max 10 MB.)

Suggested single PDF (merge items):

* **CV (1–2 pages)** with links to eBPF work.
* **One-page project brief** (timeline, deliverables, success metrics).
* **Appendix (links only):** tutorials repo, representative labs, and any talk slides/recordings.

Filename suggestion: `Zheng_Yusheng_eBPF_Fellowship_Packet.pdf`

---

## How did you hear about the eBPF Foundation Fellows Program?

* **eBPF Foundation website** (announcement blog post). ([eBPF][1])

---

## Please provide your full name.

**Yusheng Zheng**

---

## Please provide your email address.

**[yunwei356@gmail.com](mailto:yunwei356@gmail.com)**


Here are the **recent eBPF capabilities** from the 6.8–6.11 time frame that are important and, if you haven’t already added them, worth covering next. I’m giving you the “why it matters” plus a concrete lab you can drop into *bpf-developer-tutorial* for each.

**BPF token (delegated privileges, v6.9).** This lets a privileged manager hand out narrowly-scoped BPF capabilities to an otherwise unprivileged process—perfect for containerized setups and CI where you don’t want to run everything as root. A focused lab can show a runtime mounting a userns-bound BPFFS, minting a token, and letting a child process load a simple tracing/XDP program only via that token; include a feature probe and failure modes when the BPFFS mount isn’t correct. ([kernelnewbies.org][1])

**BPF arena (pointer-friendly shared memory, v6.9+).** Arenas relax verifier rules for arena-owned pointers so you can build real pointer-based data structures (lists, hash tables, queues) without the old “index into a giant array” dance. A compelling lab is “ring-buffered work queue in an arena” with producers/consumers across CPUs, plus a comparison to classic map-backed structures. ([LWN.net][2])

**BPF exceptions / `bpf_throw` (v6.8).** Exception support gives you a fast, explicit bail-out path and enables assert-style checks in BPF. It’s good pedagogy for verifier-hard paths where proofs are messy but runtime checks are fine. Add a small tracing lab that validates a precondition, calls `bpf_throw()` on violation, and demonstrates the optional exception callback; document what happens across tail calls and program nesting. ([LWN.net][3])

**TCX (link-based TC with multi-program, v6.6+).** TCX is the modern, link-based attach API for traffic control with first-class multi-program support and better ownership semantics than the legacy `tc` qdisc path. Add a networking lab that contrasts classic `tc` vs TCX attach and shows parallel programs attached at the same hook; make it kernel-gated at 6.6+ and show how to detect and fall back. ([Red Hat Docs][4])

**Open-coded iterators & new iterator targets (v6.8).** Recent kernels added iterator kfuncs for `task`, `task_vma`, `css`, and `css_task`. Iterators are a clean way to walk kernel objects without writing your own traversal. A short lab can print a filtered process table via open-coded `task` iteration, then repeat with the `SEC("iter/task")` style and compare pros/cons. ([Linux Kernel Documentation][5])

**HID-BPF expansions (v6.11).** HID-BPF grew new hooks and helpers so you can implement small input-device fixes and filters in BPF instead of carrying out-of-tree drivers. A fun lab: intercept a HID device and tweak a report (e.g., dead-zone or key remap), with a guardrail about when HID-BPF is appropriate. ([Linux Kernel Documentation][6])

**Multi-attach and multi-link polish (kprobe/fentry multi, uprobe link\_info, 6.2–6.8+).** While not brand new, multi-attach is under-taught and very practical for tracing broad function sets. Add a “multi-attach 101” lab using `kprobe.multi`/`fentry.multi`, and another showing uprobe multi-link with `link_info` inspection so learners understand link ownership. ([LWN.net][7])

**Per-CPU kptrs and timer improvements (v6.8).** Newer kernels improved working with per-CPU objects, including support for local per-CPU `kptr` and pinning `bpf_timer` to the current CPU. A tiny lab that keeps per-CPU state via kptr and emits periodic stats with a CPU-pinned timer will teach both patterns cleanly. ([Red Hat Docs][8])

**Compatibility & troubleshooting to fold into the docs.** Two things are biting people now: the 6.9+ **FRED** stack layout change, which can break raw `pt_regs` scraping in custom code, and the general kernel-feature drift that your doctor script should cover. Add a “Why your `pt_regs` looks wrong on 6.9+” callout and extend `doctor.sh` to detect TCX, exceptions, iterators, arenas, and token support. ([tanelpoder.com][9])

If you want, I can turn each of these into a one-file lab skeleton (C/libbpf + Rust/libbpf-rs + Go/ebpf-go variants) with a golden-line test and version gate so they drop straight into *bpf-developer-tutorial*.

[1]: https://kernelnewbies.org/Linux_6.9?utm_source=chatgpt.com "Linux_6.9"
[2]: https://lwn.net/Articles/961594/?utm_source=chatgpt.com "bpf: Introduce BPF arena."
[3]: https://lwn.net/Articles/938435/?utm_source=chatgpt.com "Exceptions in BPF"
[4]: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/configuring_and_managing_networking/assembly_understanding-the-ebpf-features-in-rhel-9_configuring-and-managing-networking?utm_source=chatgpt.com "Chapter 43. Understanding the eBPF networking features ..."
[5]: https://docs.kernel.org/bpf/bpf_iterators.html?utm_source=chatgpt.com "BPF Iterators"
[6]: https://docs.kernel.org/hid/hid-bpf.html?utm_source=chatgpt.com "HID-BPF"
[7]: https://lwn.net/Articles/885811/?utm_source=chatgpt.com "bpf: Add kprobe multi link"
[8]: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/9.5_release_notes/new-features?utm_source=chatgpt.com "Chapter 4. New features | 9.5 Release Notes"
[9]: https://tanelpoder.com/posts/ebpf-pt-regs-error-on-linux-blame-fred/?utm_source=chatgpt.com "When eBPF task->stack->pt_regs reads return garbage on ..."
