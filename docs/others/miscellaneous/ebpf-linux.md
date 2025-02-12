# Project Proposal for bpftime - eBPF Foundation  

## General Information

1.1. Name of Project

bpftime

1.2. Project Description (what it does, why it is valuable, origin and history)

bpftime is an userspace eBPF runtime that allows existing eBPF applications to operate in unprivileged userspace using the same libraries and toolchains. It offers Uprobe and Syscall tracepoints for eBPF, with significant performance improvements over kernel uprobe and without requiring manual code instrumentation or process restarts. The runtime facilitates interprocess eBPF maps in userspace shared memory, and is also compatible with kernel eBPF maps, allowing for seamless operation with the kernel's eBPF infrastructure. It includes a high-performance LLVM JIT for various architectures, alongside a lightweight JIT for x86 and an interpreter.

The project was started in 2023 by Yusheng Zheng and Tong Yu, as interns sponsored by [PLCT lab, ISCAS](https://plctlab.github.io/). It's hosted and maintained by the eunomia-bpf community.

1.3. How does this project align with the Foundation's Mission Statement to encourage/foster the deployment and use of eBPF in the industry (e.g., not merely allow using eBPF as an option)

1. Enable eBPF tools like bcc and bpftrace, or more complex commercial eBPF observability agents, to run in userspace on more platforms, on lower version kernels, in non-privileged container environments, or potentially in other platforms like MacOS, Windows, FreeBSD, etc, without any modification to the eBPF applications.
2. Have better performance in some cases, for example, the uprobes and LLVM JIT/AOT.
3. Enable more usecases and innovations, for example, use `bpf_override_return` for userspace functions, use XDP in DPDK applications, etc.

For more details, please refer to:

1. Our blog: <https://eunomia.dev/blogs/bpftime/>
2. Our talk at Linux Plumbers 2023: <https://lpc.events/event/17/contributions/1639/>
3. arxiv: <https://arxiv.org/abs/2311.07923>

1.4. Project website URL

<https://eunomia.dev/bpftime/>

1.5. Social media accounts, if any

A Discord server: <https://discord.gg/jvM73AFdB8>

## Legal Information

2.1. Project Logo URL or attachment (Vector Graphic: SVG, EPS), if any.

No logo for bpftime yet.

2.2. Project license.  We recommend an [OSI-approved license](https://opensource.org/licenses), so if the license is not one on the list, explain why.

MIT License

2.3. Existing financial sponsorship, if any.

The PLCT lab, ISCAS, has some open Internships positions for the project. Some professors from universities also provide some one time sponsorship for the project.

See <https://github.com/eunomia-bpf/bpftime#acknowledgement> for more details.

2.4. Was the project previously accepted by any other consortium or foundation?
     If so, does the project now accept the eBPF Foundation as its primary support
     body?

This project has not been accepted by any other consortium or foundation. We are open to accepting the eBPF Foundation as our primary support body.

2.5. Trademark status, if any.

2.6. Proposed Technical Charter, based on the [template].
Include doc as attachment or give URL of doc.  It is ok to change the
text (e.g., "Technical Steering Committee") to match the actual structure of
the project; projects are free to use whatever governance structure they want.

### Technical Information

3.1. High level assessment of project synergy with any existing projects under the eBPF Foundation, including how the project compliments/overlaps with existing projects, and potential ways to harmonize over time. Responses may be included both inline and/or in accompanying documentation.

3.2. Project Code of Conduct URL.  We recommend one based on the [Contributor Covenant v2.0](https://www.contributor-covenant.org/version/2/0/code_of_conduct/) or the [LF Projects Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

3.3. Source control URL

<https://github.com/eunomia-bpf/bpftime>

3.4. Issue tracker URL

<https://github.com/eunomia-bpf/bpftime/issues>

3.5. External dependencies (including licenses, and indicate whether each is a build time or runtime dependency)

- boost (Boost Software License) - runtime time
- <https://github.com/gabime/spdlog> (MIT) - runtime time
- <https://github.com/nlohmann/json> (MIT) - runtime time
- <https://github.com/p-ranav/argparse> (MIT) - runtime time
- bpftool (GPL-2.0-only OR BSD-2-Clauss) - optional, build time
- libbpf (LGPL-2.1 OR BSD-2-Clause) - optional, runtime time
- <https://github.com/vbpf/ebpf-verifier> (MIT) - optional, runtime time
- <https://github.com/iovisor/ubpf> (Apache 2.0) - optional, runtime time
- LLVM (Apache 2.0) - optional, runtime time

3.6. Standards implemented by the project, if any. Include links to any such standards.

<https://www.kernel.org/doc/html/next/bpf/instruction-set.html> - eBPF instruction set

Which is tested by the [bpf_conformance](https://github.com/Alan-Jowett/bpf_conformance)

3.7. Release methodology and mechanics

3.8. List of project's official communication channels (slack, irc, mailing lists)

- Discord: <https://discord.gg/jvM73AFdB8>
- Mailing list: <mailto:team@eunomia.dev>

3.9. Project Security Response Policy for handling any vulnerabilities reported

3.10. Expected budget request.  See [Project Benefits] for items that may be requested for potential support.

3.11. Any additional information the BSC and Board should take into consideration when reviewing your proposal.
