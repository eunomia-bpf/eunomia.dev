# When should I collect eBPF samples in a BPF ring buffer instead of a perf event array?

**Short answer:** a perf event array and a BPF ring buffer move eBPF samples from
kernel to user space with different ownership models. A perf event
(`BPF_MAP_TYPE_PERF_EVENT_ARRAY`) is per-CPU: one perf fd and a fixed-size data
region per CPU, consumed per-CPU, and each CPU drops samples independently when
its own region fills. A BPF ring buffer (`BPF_MAP_TYPE_RINGBUF`) is a shared
multiproducer/singleconsumer (MPSC) ring with no perf fd, consumed once, where
`bpf_ringbuf_reserve()` writes zero-copy into one ring shared across CPUs. The
decisive boundary is who the producers are. If producers and the consumer live
on the same CPU, or you want per-CPU isolation with a fixed-size data region, a
perf event array is the right tool. If producers span many CPUs or threads and
you want one shared consumption stream without a perf fd per CPU, use the ring
buffer. A ring buffer can drop a record a perf array keeps: a ring buffer
reservation fails atomically for the shared ring when it is full, and even
while there is free space it can fail in NMI context while contending for the
ring lock; a perf event array overflow is bounded per-CPU, so each CPU's region
simply stops accepting new samples on its own.

## The two ownership models

A perf event array has one slot per CPU. Each slot is a perf buffer with a
fixed-size data region and one perf fd. The consumer reads each CPU's buffer
independently. When a CPU's region overflows, only that CPU stops accepting new
samples; every other CPU keeps going. Nothing is shared between CPUs except the
map, so a burst on one CPU cannot starve another CPU's samples.

A ring buffer is the opposite. A single `BPF_MAP_TYPE_RINGBUF` map presents one
power-of-2 sized ring to all CPUs (a pool of rings is available through
`BPF_MAP_TYPE_HASH_OF_MAPS` for sharded designs). Producers on any CPU reserve
space in the same ring; one consumer drains it. There is no perf fd and no
per-CPU region. The kernel documents the motivation directly: a shared ring
across CPUs gives better memory utilization and removes the per-CPU perf
buffers that perf event arrays require.

The helper API is where the difference shows up. `bpf_perf_event_output()`
copies a record into the per-CPU perf buffer. `bpf_ringbuf_output()` does the
same copy into the ring buffer. `bpf_ringbuf_reserve()` /
`bpf_ringbuf_commit()` / `bpf_ringbuf_discard()` instead hand the program a
pointer directly into the ring, so the program writes the sample in place with
no copy. `bpf_ringbuf_query()` reports `BPF_RB_AVAIL_DATA` (unconsumed bytes)
and `BPF_RB_RING_SIZE`, which is what a consumer uses to size and pace itself.

## Failure modes are not the same

The drop behavior differs in shape, which is the point most people miss:

- **Perf event array overflow** is per-CPU and independent. Each CPU's region
  fills on its own; that CPU's further samples are dropped while the others
  continue. There is a stable per-CPU backpressure model.
- **Ring buffer overflow** is global across the shared ring. When the shared
  ring is full, `bpf_ringbuf_reserve()` fails atomically and the sample is
  dropped for every producer at once.
- **Ring buffer reservation can fail even when not full.** In NMI context,
  `bpf_ringbuf_reserve()` may fail to acquire the ring lock, so the
  reservation fails despite free space. A program that samples in NMI or a
  tight atomic path must treat "not full" as "not guaranteed to succeed."

## How to verify which one you need

1. **Locate the producers.** If sampling runs on the same CPU that consumes,
   or you need per-CPU isolation, a per-CPU perf event array avoids the shared
   ring's global contention and its cross-CPU starvation.
2. **Watch the drop source, not just the count.** For a ring buffer, poll
   `bpf_ringbuf_query(BPF_RB_AVAIL_DATA)` before and after a load and confirm
   the consumer drains toward zero. For a perf event array, check each CPU's
   overflow independently. A ring buffer that holds high available data after
   a burst is dropping globally; a perf array will show the overflow only on the
   CPUs that were hot.
3. **Check the producer context.** If the program can run in NMI or a tight
   atomic path, log `bpf_ringbuf_reserve()` failures specifically. Free space
   does not guarantee a successful reservation there, so the copy-based
   `bpf_ringbuf_output()` or reduced lock contention are the mitigations, not a
   larger ring.
4. **Match the topology.** If producers span many cores, the shared ring
   removes the per-CPU perf-fd overhead of a perf event array. If you already
   pin consumer threads to CPUs, a per-CPU perf event array lines up cleanly.

## The limitation that decides it

A shared ring buffer trades per-CPU isolation for a single consumption stream.
A burst on one CPU can fill the shared ring and drop records for every producer
at once; there is no per-CPU backpressure in the shared design. A perf event
array keeps that isolation: a burst on one CPU only drops that CPU's samples.
Use the ring buffer when a shared stream and zero-copy reserve are worth
accepting global overflow; stay with a perf event array when per-CPU isolation
and a fixed per-CPU data region are the requirement.

## References

- [The Linux kernel documentation: BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html)
- [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [The Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/)

## Community discussion today

The monitored window was not technical. Over the previous 24 hours the two
opt-in Slack archives returned no messages; the 7-day fallback held only meeting
logistics — scheduling posts, a co-presenting thank-you, and a link to a
community session — with no eBPF question, symptom, or design dispute. The two
allowlisted chat workspaces had no visible browser session this run, and the
public mailing-list and forum archives were not reviewed. Those sources are
recorded as unavailable coverage, not quiet. Because the readable archive
yielded no question, this Q&A is grounded in public primary documentation
rather than a community message: the question is a recurring practitioner
decision — choosing between the BPF ring buffer and the perf event array for
moving samples to user space — verified against the kernel's ring buffer design
document and the `perf_event_open(2)` interface. No private text, identity,
channel, or link is reproduced here.
