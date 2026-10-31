# Why does a generated BPF loader object get rejected when only subprograms or unloaded programs push the count past the cap, and how does the check move to the real loaded program count?

Because the cap is enforced against the object's total program count, which counts subprograms and autoload-disabled programs that the loader never loads, so the check rejects objects that should have loaded. The fix clamps the code-generation bookkeeping to the cap and defers the real rejection to the point where a program file descriptor is actually stored on the loader stack, which is reached only by programs that survive loading.

## The mechanism

1. **The fixed stack frame.** A generated loader is a BPF program that loads the rest of the object, and it keeps its working state in a fixed-size stack structure. That structure holds the BTF file descriptor, the inner map file descriptor, and an array of `MAX_USED_PROGS` program file descriptors. Every stack address in the emitted program is an offset from that fixed size, so the last slot sits at the bottom of the frame and the next slot after the array is the stack pointer itself, outside the reserved area.

2. **Why the count is wrong.** The object exposes its program count as a single number, and that number includes static and global functions that live in the text section as subprograms plus every program with autoload turned off. Those programs are skipped by the load pass, so the number can sit above the cap even when the programs that actually load are well under it. The loader's generated cleanup loop, however, was sized from that raw count. It emits one close of the file descriptor per slot, and when the count is one above the cap the last close addresses the slot just past the end of the frame, which the verifier rejects.

3. **The three states.** In mainline today there is no program count check at all in the loader generator. The cap exists only for maps, which are final at the prepare step, while programs are filtered later. The first attempt added an up-front rejection at generation time against the raw count, which is where the false rejection comes from: an object with one real program and thirty-two subprograms counted thirty-three and failed generation even though it should load. The follow-up, which landed in the next-release tree this week, removes that early check, clamps the cleanup loop to the cap, and puts the real check just before the file descriptor is stored on the stack, so it runs only for programs that survived the load pass. An object that truly loads more programs than the cap still fails with the size error, now at the right point.

## How to verify and debug

1. **Count the programs that actually load.** The subprograms in the text section and the autoload-disabled programs are skipped by the load pass and never occupy a slot. What the cap bounds is the number of entry programs that keep autoload on. If that number is at or under the cap, the object should load regardless of how many subprograms are in the text section.

2. **Read the failure stage.** On a mainline tree the generation succeeds silently and the object fails at load time when the verifier rejects the stack frame, with an out-of-bounds access reported at the bottom of the frame. On the next-release tree the rejection happens at generation time, with the loader warning about exceeding the program limit, and it is the real count that decides, not the inflated one. The same warning text with a different meaning across trees is the tell that the object is near the boundary.

3. **Check the stack frame directly.** In the generated program every store of a program file descriptor targets an offset inside the frame, and a store or close that lands at the stack pointer means the count exceeded the frame. That address is the cleanest local proof that the rejection is about the number of loaded programs, not about the source files.

## The limitation

- The follow-up is in the next-release tree, not mainline. Mainline still has no program count check in the generator, so an over-cap object there fails at load time instead of generation time, with the verifier rather than the loader reporting it.
- The cap on maps is final at prepare and cannot be deferred the way the program cap can, because the map count determines the size of fixed byte offsets emitted at generation time.
- The count that matters is the loaded program count, not the object's total, so the check belongs at the file descriptor store, where the filtered count is the only one in scope.

## References

- [Kernel `tools/lib/bpf/gen_loader.c`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/gen_loader.c) — the fixed loader stack frame, the map cap, and the unclamped cleanup loop in mainline.
- [Kernel `tools/lib/bpf/libbpf.c`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c) — the load pass that skips subprograms and autoload-disabled programs, and the call that hands the raw program count to the loader generator.
- [Public bpf-list patch: enforce program limit only for loaded programs](https://lore.kernel.org/bpf/20261030092248.7388-1-mahe.tardy@gmail.com/) — the rejection table, the out-of-bounds stack access, and the two-hunk fix that clamps the cleanup loop and defers the check.
- [Public bpf-list maintainer reply applying the fix](https://lore.kernel.org/bpf/CAEf4BzaUg_KPB%2Bk14ntDTaqU-3hEUQ8o5E2hSeUkaw4xwFwi9A@mail.gmail.com/) — the rebase that changed the commit the follow-up fixes, and its landing in the next-release tree.
- [Public bpf-list author follow-up](https://lore.kernel.org/bpf/auTq4Fhs8T-lO36-@gmail.com/) — the same-day confirmation of the merge.
- [Automated review of the patch](https://lore.kernel.org/bpf/c3d38e0d458ec830e8f94f5eb68bf2402a609b3c0b07fc9ba87115712b80b28e%40mail.kernel.org/) — the machine-generated question about the fixes tag, which does not change the merged result.

## Community discussion today

The watchlist for this run covers six communities: two chat workspaces with ten channels, two archive-backed chat workspaces with three channels, one mailing list, and one practitioner forum. The three archive-backed channels were read through their approved read-only archive for the 24-hour and 7-day windows and returned no messages at all, so they were quiet rather than skipped. The ten channels in the two chat workspaces were not covered this run because reading them needs an authenticated browser session that was not available, and they are recorded as not covered rather than quiet.

The mailing list carried the loader work this answer addresses. The program cap series spans three eras: the 2021 relaxation that stopped treating a large program count as a hard error, the reject commit that moved the rejection to generation time against the raw count, and today's follow-up that corrects the count and defers the check. The maintainer applied the follow-up the same afternoon after a rebase changed the commit it fixes, and an automated reviewer raised a question about the fixes tag that the author did not answer, which is the one open thread on the series.

A second, larger mailing-list thread in the window proposes a query interface over the kernel symbol table plus a loader path that resolves untyped kernel symbols through it. The review on it concentrated on the interface shape rather than the idea, so it is an open design question rather than a settled boundary.

On the practitioner forum the only substantive thread inside the window was a production-experience post about operating eBPF at scale, with the newest technical discussion a few days old and still running on a packet-crafting scanner built on the zero-copy packet socket API. Two questions there stay open and are the ones an engineer hits the same way: how much a kernel-wakeup flag buys on a single transmit ring, and whether the busy-poll loop is ever worth the pinned core it costs on a loaded virtual machine.

This run's archive-backed sources produced no messages in either window, so the question above was sourced from the permitted public bpf mailing list and verified against its patch thread and the released kernel sources, with the practitioner forum supplying the operational themes in this section.
