# Why do eBPF kprobe string reads truncate long open and stat paths, and how do you read the full path?

Because the string read's destination buffer has to live inside the BPF program's 512-byte stack, a single `bpf_probe_read_user_str` call can capture at most 511 characters of a user string, while the kernel allows a full path up to `PATH_MAX`, 4096. Open and stat hand the program a pointer to a user-space path that can legally be thousands of characters long, so any read into a stack buffer of `sizeof(buf)` bytes drops everything past 511 characters and NUL-pads the tail. Read the full path by moving the string out of the stack: on recent kernels the dynamic-pointer variants of these read helpers, and on sleepable programs the sleepable copy-from-user string helper, both of which write past the 512-byte cap. Or, when the path is short, size the stack buffer to 511 and detect the truncation from the return value instead.

## The mechanism

Three limits line up:

1. **The syscall side.** `open`, `openat`, `stat` and the like take a user pointer to a NUL-terminated path. `openat`'s filename argument lands in `ctx->di` at `kprobe/sys_openat`, and the UAPI helper doc shows exactly this shape: the `bpf_probe_read_user_str` helper reading the filename pointer from `ctx->di` into a stack `char buf[PATHLEN]`, with `PATHLEN` defined as 256. The string it points to can be a full absolute path, up to `PATH_MAX`, 4096 characters.
2. **The helper contract.** `bpf_probe_read_user_str` copies a NUL-terminated string from the unsafe user address into `dst`, with `size` including the terminating NUL. If the string is longer than `size`, just `size - 1` bytes are copied and the last byte is set to NUL; on success it returns the number of bytes written, including the terminal NUL. `bpf_probe_read_kernel_str` has the same semantics for kernel addresses.
3. **The BPF stack cap.** The destination `buf` is a local of the BPF program, so it lives on the BPF stack, which the verifier bounds at `MAX_BPF_STACK`, 512 bytes. You cannot declare a 4096-byte stack buffer. The largest `size` you can pass to a stack-buffer string read is therefore 512, and a single call captures at most 511 characters plus the NUL.

So the boundary is: a stack-buffer string read is hard-capped at 511 usable characters, while `PATH_MAX` is 4096. Any real path longer than 511 characters (deeply nested mount paths, a `/proc/self/cwd`-resolved symlink, a long CWD-relative path) is truncated to its first 511 characters on every stack-buffer read.

## Verification and debugging path

1. **Reproduce at the helper level.** Attach a kprobe on `sys_openat`, read `ctx->di` into `char buf[512]` with `bpf_probe_read_user_str`, and compare the returned length `res` against `sizeof(buf)`.
2. **Detect truncation from the return value.** On success the helper returns the bytes written, including the terminal NUL. If the string fit, `res` is `len + 1`, strictly below `sizeof(buf)`. If it did not fit, `res` equals `sizeof(buf)` (all 512 bytes: 511 characters plus NUL). So `res == sizeof(buf)` is your truncation flag.
3. **Know the ambiguity at the cap.** The flag fires for `len >= 511`. A path of exactly 511 characters fills the buffer completely and also returns `sizeof(buf)`, so the return value alone cannot tell "fully captured at 511" apart from "truncated above 511". If you need the exact length at the boundary, use a non-stack sink instead of a stack buffer.
4. **Cross-check against an external tracer.** Run `strace -e trace=openat` or a parallel `perf trace` of the openat sysenter; both print the full path the kernel recorded through `getname`. When your eBPF path is shorter and ends mid-token, that is the 512-byte cap, not a probe bug.

## The limitation

- **511 is a hard wall for stack-buffer reads.** `MAX_BPF_STACK` is 512 bytes and the verifier rejects programs that exceed it, so you cannot simply make the buffer bigger past the cap. `char buf[512]` is the maximum.
- **`res == sizeof(buf)` conflates two cases.** It is true for a string of exactly 511 characters (fully captured) and for any longer string (truncated). The helper does not distinguish them.
- **The escapes are kernel-version-dependent.** The dynamic-pointer str variants and the sleepable `bpf_copy_from_user_str` helper are newer kfuncs; they need a kernel with BTF, and the dynamic-pointer str variants additionally sit behind the events feature gate. They are absent on older kernels that expose only the classic stack-buffer helpers.
- **`bpf_probe_read_user_str` is not for other tasks' paths.** It reads the current task's user memory. For a different task's user path you need the cross-task str variant, `bpf_copy_from_user_task_str`, and its dynamic-pointer form.

## References

- [kernel include/uapi/linux/bpf.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) - `bpf_probe_read_user_str` / `bpf_probe_read_kernel_str` contract: the `size - 1` copy cap, NUL set, bytes-written return, and the `kprobe/sys_open` example.
- [kernel include/linux/filter.h](https://github.com/torvalds/linux/blob/master/include/linux/filter.h) - the `MAX_BPF_STACK` 512-byte cap the verifier imposes on BPF program stacks.
- [kernel kernel/bpf/verifier.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) - where the verifier enforces the 512-byte BPF stack depth limit.
- [kernel include/uapi/linux/limits.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/limits.h) - `PATH_MAX` 4096 and `NAME_MAX` 255, the full path length the kernel allows.
- [kernel kernel/bpf/helpers.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c) - the `bpf_copy_from_user_str` / `bpf_copy_from_user_task_str` sleepable kfuncs and the dynamic-pointer str helper declarations.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced eleven messages this run. All three threads in the window were already answered or were not questions: the Cilium eBPF discussion cross-post is the 10-13 published entry, the FUSE and OBI trace-propagation thread is the 10-14 published entry, and the GenAI semantic-conventions PR note is a review-status update rather than a question. This page is therefore grounded in public primary documentation instead of a window thread: it answers a recurring, source-checkable eBPF mechanics question about stack-capped string reads at open and stat probes, which none of the published entries covers.

Channel coverage for this run: the two opt-in archives supplied eleven messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
