# Why does the iocost cgroup cost model file only switch the built-in model between auto and user coefficient sets, and how does a BPF struct ops cost model take over a device while that file stops selecting a model?

Because the cgroup file is the interface of the built-in model, and a BPF program is attached out of band. The blk-iocost cgroup cost model file's write path parses the `ctrl=` token (`auto` or `user`), the `model=` token (always `linear`), and the seven linear coefficient tokens, and flips the `user_cost_model` bit; it has no program to run. The public patch series binds a BPF struct ops implementation to one device's iocost instance through a struct ops link; while the link is attached, the device prices its IO with that program, and the cgroup file stops selecting a model. It keeps accepting `ctrl=auto` and `ctrl=user`, but the built-in coefficients take effect only after the link is detached.

## The mechanism

1. The built-in side (mainline). The blk-iocost cgroup file is `cost.model`, registered `CFTYPE_ONLY_ON_ROOT`. Its write handler, `ioc_cost_model_write()`, parses only the `ctrl=` token (`auto` or `user`), the `model=` token (accepted only as `linear`), and the seven linear coefficient tokens (`rbps`, `rseqiops`, `rrandiops`, `wbps`, `wseqiops`, `wrandiops`, `flushiops`). Writing any coefficient or `ctrl=user` sets the `user_cost_model` bit and `ioc_refresh_params` applies the user set; `ctrl=auto` clears it and the automatic profile stepped from the device class applies again. The show path prints `ctrl=user` or `ctrl=auto` together with `model=linear` and the current coefficients. There is no value that names a program.
2. The takeover (public patch). The series adds a struct ops cost model interface: a BPF program implements the model ops, and the attach path, gated on a BPF cost model config option, binds it to one device's iocost instance. The attach and detach paths match the attachment under the queue lock by comparing the program ops and the link against the ioc attached model (`ioc_attached_model_locked()`). The model is delivered `iocg_init()` and `iocg_free()` for the cgroups already online; both callbacks run with the ioc lock held and must not sleep. Device removal ejects the attached model, while the struct ops link remains until userspace destroys it. The same series adds an `iocost_ioc_tick` tracepoint that reports the per-period device summary.
3. The cgroup file while attached. The cgroup-v2 wording the series adds says that while a BPF model is attached the file's model value is `bpf`; the built-in coefficient tokens remain readable and writable, but they take effect only after switching back to `linear`, and the automatic profile stepping does not switch profiles while the BPF model is in use — as with user-set parameters, the profile selected from the device class still applies. Only detaching the link (or removing the device) restores built-in pricing.

## How to verify and debug

1. **Read the file.** Read the iocost cost model file for the cgroup of the device. With a BPF model attached the model value shows `bpf` while the built-in coefficient line still lists the linear coefficients; those values are inert until the link is detached.
2. **Detach out of band.** Destroy the struct ops link from userspace. The device prices with the built-in model again and the cgroup file's `ctrl` and coefficient writes take effect again. Device removal instead ejects the attached model while the link stays alive until userspace destroys it.
3. **Watch the per-period pricing.** Enable the `iocost_ioc_tick` tracepoint added by the same series. It reports the final per-period summary — busy level, cur period, nr active, and usage time — so the BPF model's effect is visible per period. Note that a mid-period write to `io.cost.qos` changes the period time and vrate only after the period has already run, so the summary reflects the values the period ran with.

## The limitation

- Mainline has no attach path: master `block/blk-iocost.c` knows only the built-in linear model and the `user_cost_model` bit; the struct ops binding is the public patch series, gated on a BPF cost model config option.
- The binding is per device: one model per device iocost instance, not a global override.
- Coefficient writes while a BPF model is attached are accepted but inert; they take effect only after switching back to the built-in model.
- Detach is asymmetric: device removal ejects the attached model, but the struct ops link remains until userspace destroys it.

## References

- [kernel `block/blk-iocost.c`](https://github.com/torvalds/linux/blob/master/block/blk-iocost.c) — the `cost.model` cgroup file registered `CFTYPE_ONLY_ON_ROOT`, its write handler that accepts only `ctrl` auto/user and `model=linear`, and the `user_cost_model` bit.
- [kernel Documentation/block/cgroup-v2.rst](https://github.com/torvalds/linux/blob/master/Documentation/block/cgroup-v2.rst) — the iocost block and the automatic profile stepping the doc patch extends.
- [Public bpf mailing-list patch series: blk-iocost BPF struct ops cost model support](https://lore.kernel.org/bpf/895cb52c-1542-4ff7-abaf-844a73126fa8@linux.dev/) — the attach/detach paths, the `iocg_init()` / `iocg_free()` callbacks, the `iocost_ioc_tick` tracepoint, and the cgroup-v2 doc wording.

## Community discussion today

A public bpf-list patch series adds BPF struct ops cost model support to blk-iocost: a BPF program bound to one device's iocost instance takes over its IO pricing while the cgroup cost model file stops selecting a model. The series is a five-part post: the shared-definition move (later folded back into the main file), the struct ops attach/detach core, a selftest, the per-period `iocost_ioc_tick` tracepoint, and the cgroup-v2 doc wording. The review thread concentrates on the lifetime edges: the attachment match must compare both the program ops and the link, the iocg init/free callbacks run with the ioc lock held and must not sleep, device removal ejects the model while the link stays alive, and the doc must say the built-in coefficients remain writable while a BPF model is attached but take effect only after switching back.

This run's archive-backed sources produced 0 messages in the 7-day window (the two opt-in archive sources were empty at snapshot time), so the question above was sourced from the permitted public bpf mailing list, reviewed through its public atom feed, and the patch thread was used as the primary reference. The Discord channels for eunomia-bpf and sched-ext and r/eBPF were not covered this run (no visible-browser session was available), so they are marked not-covered rather than quiet.
