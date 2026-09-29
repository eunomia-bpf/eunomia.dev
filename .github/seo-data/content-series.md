# Daily Report content series

## Rolling topic policy

Use the newest ten actually published Daily Reports, excluding open, draft, closed-unmerged, or branch-only work.

Target mix:
- **5-7 eBPF-centered**
- **1-2 pure Agent maximum**
- the remainder adjacent systems topics

Publish exactly one new bilingual report per scheduled run. Prefer an active series when the next question is technically distinct and supported by primary evidence.

## Current published mix

Before the September 29 publication, the newest-ten window is **7 eBPF-centered / 1 pure Agent / 2 adjacent systems**. Today's selected eBPF report rotates out an older eBPF report, so successful publication keeps the window at **7 / 1 / 2**.

## Active series: eBPF Deployment Compatibility and Lifecycle

Working question: how should an eBPF application remain loadable, semantically correct, state-compatible, and operationally explainable as kernels, distributions, BPF interfaces, and application generations change?

Published boundaries:

1. **Capability evidence**  
   `/research/ebpf-kernel-capability-evidence/`  
   Admit an artifact from direct target evidence instead of kernel-version inference.

2. **Cross-kernel semantic compatibility**  
   `/research/ebpf-kernel-upgrade-semantic-compatibility/`  
   Re-prove application behavior when an artifact moves across kernel versions.

3. **Kernel-interface negotiation**  
   `/research/ebpf-kernel-interface-negotiation/`  
   Represent typed and scoped kfunc, iterator, `struct_ops`, and provider requirements before choosing an artifact variant.

Selected next boundary:

4. **Persistent map-state semantic compatibility**  
   `/research/ebpf-map-reuse-semantic-compatibility/`  
   Separate libbpf's kernel-visible map reuse checks from structural BTF compatibility and application-level semantic compatibility. Develop stable structural fingerprints, semantic revisions with migration contracts, and shadow validation before granting a new generation write authority.

## Novelty guards

Do not repeat the selected map-state boundary as generic "stateful upgrade." The August transactional-upgrade report already covers application-wide prepare, migrate, commit, retire, and rollback.

Do not repeat:
- kernel-version capability admission;
- generic cross-kernel semantic compatibility;
- typed interface negotiation;
- architecture specialization;
- userspace-runtime portability.

Future candidate boundaries include reboot-safe reconstruction after the old kernel object graph disappears and persistent BPF-link ownership after controller restart. Each candidate must be re-researched before publication and must remain distinct from map reuse within one live kernel.

## Topic selection rule

For each run:
1. calculate the newest-ten published mix;
2. inspect the active series first;
3. compare at least one credible alternative when novelty is uncertain;
4. reject topics that only summarize sources or rename an existing thesis;
5. select one question with a concrete gap, mechanism, artifact, discriminating evaluation, and failure condition;
6. publish exactly one English/Chinese pair and update this file only after the topic is selected.
