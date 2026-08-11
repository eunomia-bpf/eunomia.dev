# Why can `scxctl` accept a scheduler switch while the service still does not start as intended?

**Short answer:** bind scheduler arguments explicitly to `--args`, then verify the kernel state separately from the CLI exit status. If the scheduler argument starts with `-`, use `=`:

```console
$ scxctl switch --sched lavd --args="--autopower"
```

Without `=`, an option parser can interpret `--autopower` as another `scxctl` option rather than the value of `--args`. Quotes control shell word splitting; the equals sign identifies which option owns a dash-prefixed value. For multiple arguments, follow the installed version's `scxctl switch --help`; released versions have used a single comma-separated string such as `-a="-v,--performance"`.

A successful request is not a health check. Confirm the loader, ask `scxctl` what is active, and inspect the kernel's `sched_ext` state. Version skew matters because `scxctl` moved into the `sched-ext/scx` project and its interface has evolved.

## Three parsers are involved

The command crosses three boundaries: the shell converts text into `argv`; `scxctl` parses its options and packages scheduler arguments; the loader starts the scheduler, whose parser interprets those arguments. Shell quotes are normally removed before `scxctl` runs. These forms can therefore differ:

```console
# Unambiguous option/value binding
$ scxctl switch --sched lavd --args="--autopower"

# The dash-prefixed token may be parsed as another option
$ scxctl switch --sched lavd --args "--autopower"
```

Do not copy shell quoting mechanically into systemd `ExecStart=`. systemd has its own splitting rules and does not invoke a shell unless one is explicitly requested. First prove an interactive command, then preserve the same argument boundaries in the unit and inspect the effective definition with `systemctl cat`.

## Switch once, then verify every layer

Start from the local interface:

```console
$ scxctl --version
$ scxctl switch --help
$ scxctl list
$ sudo scxctl switch --sched lavd --args="--autopower"
```

Then verify the control plane and the kernel:

```console
$ systemctl status scx_loader.service --no-pager
$ scxctl get
$ cat /sys/kernel/sched_ext/state
$ cat /sys/kernel/sched_ext/root/ops
$ journalctl -u scx_loader.service -b --no-pager
```

The kernel documentation defines `state` as `disabled`, `enabled`, or `disabling`; `root/ops` reports the registered scheduler name. A successful CLI message with a failed loader, a non-`enabled` state, or the wrong `root/ops` value is not a successful deployment. Distribution unit names may differ, so discover the installed unit rather than guessing. Also look for a scheduler launched outside the loader, because two controllers can replace each other.

Classify the earliest failure:

- unknown option or missing value: command syntax;
- accepted request but failed loader: loader journal or unsupported scheduler argument;
- scheduler starts and exits: kernel support, BTF, verifier, or scheduler runtime failure;
- kernel is enabled with another `root/ops`: a competing controller switched it.

Record exact CLI, scheduler-package, and kernel versions. The deprecated standalone repository now points to the official project, so its examples explain argument binding but cannot replace local `--help`. During diagnosis, stop automatic restart loops, perform one controlled switch, and preserve the first error before restoring restart policy.

## References

- [Linux kernel documentation: `sched_ext` state and scheduler name](https://docs.kernel.org/scheduler/sched-ext.html)
- [Official `sched-ext/scx` repository](https://github.com/sched-ext/scx)
- [Standalone `scxctl` README and argument examples (deprecated)](https://github.com/frap129/scxctl)
- [systemd manual: command lines in `ExecStart=`](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#Command%20lines)
- [systemd manual: service status](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#status%20PATTERN%E2%80%A6)
- [BPF mailing list: reducing hash-map element memory use](https://lore.kernel.org/bpf/20260805223516.1495988-1-tjmercier@google.com/T/#t)
- [BPF mailing list: ARM64 arena kfunc and `struct_ops` arguments](https://lore.kernel.org/bpf/20260810190922.3408757-1-puranjay@kernel.org/T/#t)

## Community discussion today

Today's visible review covered all 6 approved communities and all 15 allowlisted channels or public pages; every target was accessible. The 24-hour window contained a real scheduler-control troubleshooting exchange, so no seven-day fallback was needed. Names, accounts, employers, channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained.

### Scheduler control needs an end-to-end check

The selected discussion involved a scheduler switch that did not initially produce the intended running service. A dash-prefixed scheduler argument was not bound unambiguously to the CLI option. Correcting that boundary allowed the scheduler to run. The broader lesson is that a confirmation line is only one checkpoint: verify the loader and the kernel's active scheduler.

An older unanswered forum question about minimizing kernel configuration fell outside both the daily and fallback windows, so it was not used. General project areas mostly contained newcomer activity, automated build notifications, or quiet specialist channels.

### Instrumentation discussions focused on ownership

The GenAI observability group discussed whether SDK vendors or the OpenTelemetry community should own particular agent integrations. Contributors coordinated planned TypeScript instrumentation while waiting for another SDK's native semantic-convention support, aiming to close coverage gaps without duplicate spans. The eBPF instrumentation group had no new substantive daily activity; its recent discussion continued to distinguish runtime eBPF instrumentation from compile-time language instrumentation.

### Upstream work emphasized interface correctness

The public BPF archive covered hash-map layout reductions, `BTF.ext` bounds hardening, AF_XDP metadata ABI alignment, overlapping RCU protection, ARM64 arena arguments, and aggregate returns. The public forum offered a verifier explainer and an older profiling article rather than a new troubleshooting report. General eBPF chat had a project announcement and a late reply on the previous socket-map topic, but no stronger unresolved question. Across the day, the common theme was interface precision: every layer must agree on who owns a value and how successful state is verified.
