---
date: 2026-09-25
slug: qwen3-ebpf-in-kernel-inference
description: Can Linux eBPF execute a real language model? A Qwen3-0.6B experiment shows exactly which work runs in the kernel, how fast it is, and why INT4 and arena memory remain open tradeoffs.
---

# Can a language model run in Linux eBPF?

Linux eBPF programs usually observe or influence work done elsewhere: trace a syscall, filter a packet, or choose a scheduling policy. [qwen3-ebpf](https://github.com/eunomia-bpf/qwen3-ebpf) asks a different question. Could a verified BPF program do the numerical work of a real language model, rather than merely watch a model server?

The experiment runs the 28 decoder layers of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) in Linux eBPF, scores its 151,936-token vocabulary, and chooses the next token inside the kernel. It also keeps a KV cache for subsequent tokens. This is a working forward pass, not a useful replacement for a CPU or GPU inference engine: one measured token still takes around 1.2 seconds on the test host, and C remains responsible for loading the model, text handling, and dispatching BPF operators.

<!-- more -->

## Start with one token

A language model does not receive text directly. A tokenizer first turns text into integer IDs. The model looks up an embedding for each ID, transforms that vector through a stack of decoder layers, then assigns a score, or *logit*, to every possible next ID. Taking the highest score gives a next token; decoding its ID produces text.

Here is the division of work in this prototype:

```text
text -> C tokenizer -> token ID and BF16 model weights
                         |
                         v
                 eBPF: 28 decoder layers
                       matrix math, normalization, RoPE, attention
                       KV-cache writes and reads
                       vocabulary projection and argmax
                         |
                         v
                    next token ID -> C text decoder
```

The C driver runs BPF socket-filter programs through `bpf_prog_test_run_opts`. They are test-run entry points, not filters attached to an interface, and the project does not install a persistent model service. BPF performs matrix-vector products, RMSNorm, SiLU, vector operations, RoPE rotation, causal attention, and the final argmax. C supplies an embedding, the model weights and RoPE trigonometric values, then invokes the programs in the order required by the model. For a longer prompt, every input position traverses the 28 layers. When generating another token, BPF-written K/V vectors let attention reuse previous positions instead of recomputing them.

That is what “in-kernel inference” means here: the forward arithmetic and token choice run in verified BPF programs. It does not mean that a text prompt enters the kernel and a finished sentence emerges from one call.

## Why the computation is split into pieces

An eBPF program must pass the Linux [verifier](https://docs.kernel.org/bpf/verifier.html), which reasons about execution paths and memory accesses before the program can run. The implementation makes its work bounded and moves intermediate state through BPF maps. It uses fixed-point integers rather than the original BF16 floating-point arithmetic: Q16 activations, Q24 matrix weights, and Q20 normalization scales. SiLU and attention softmax use integer approximations; C computes the trigonometric inputs while BPF rotates the RoPE vectors.

The important loop is `bpf_loop`. One matrix invocation computes up to 128 rows. Other invocations cover all 24 query/key normalization heads, all 16 query heads for a bounded attention-history chunk, or 256 past positions. New K/V pairs are written by the attention BPF program into its KV map. These changes cut repeated user/kernel transitions without pretending that all 28 layers have become one enormous verified program. A one-token run now makes about 4,169 `bpf` calls, versus tens of thousands in earlier, smaller-batch versions; that count is an implementation measurement, not a throughput claim.

Two failed attempts explain the boundary. An early monolithic RMSNorm exceeded the verifier's instruction-processing budget on the test kernel; restructuring it around bounded callbacks made it load. Putting BF16-to-Q24 conversion directly in a 3,072-column BPF loop hit a verifier complexity error. Those are outcomes of particular programs on Linux 6.17 arm64, not proof that every larger BPF program is impossible.

## Where the weights live

The official model is a roughly 1.5 GB BF16 Safetensors file mapped read-only by C. The default inference path copies the active matrix rows into a memory-mapped BPF work map after converting them to Q24. Preconverting every matrix to four-byte Q24 created a roughly 3.0 GB file and ran slower in exploratory trials on a host with about 4.6 GiB available memory. Doubling the weight footprint did not remove the bottleneck.

We also built a full-model *optional* arena path. It copies only the current 128-row BF16 batch into a roughly 2 MiB BPF arena. A 65,536-entry lookup table in that arena maps each BF16 bit pattern to the same Q24 integer used by the default driver; BPF then reads those weights and computes the dot product. This moves weight conversion for the active matrix batch out of C without trying to keep the whole model in kernel-accessible memory. On token ID `0` and a two-token continuation of `Hello, world!`, it returned the same token IDs and byte-identical full-vocabulary logits as the default path. The arena smoke test also matched 128 real Q-projection rows exactly.

The change did not establish a speedup. Six interleaved one-token measurements were 1.227/1.190/1.251 seconds for the default path and 1.220/1.228/1.234 seconds for the arena path. The reported timer starts after model and BPF setup, and this small same-host sample cannot establish throughput or tail latency. Arena addresses a data-access boundary; it neither compresses weights nor eliminates the remaining operator calls.

## Would four-bit weights solve it?

INT4 is attractive because two weights fit in a byte rather than one BF16 weight taking two bytes. A tested group-scaled INT4 representation packed the matrices used by this model into 316,616,704 bytes, including scales, and its BPF operator passed verifier and row tests. In one single-token comparison its forward pass took 0.956 seconds versus 1.243 seconds for Q24, excluding 1.928 seconds spent prepacking. That is a useful performance signal, but accuracy failed the more important test: the default path's next token for input ID `0` was `9`, while this INT4 version chose `284`; the full-vocabulary logit mean absolute error was 1.837. Other tested prompts also changed their next IDs. Even smaller quantization groups or leaving the vocabulary projection at higher precision did not restore that example.

So INT4 is a research path, not the default model. A calibrated or mixed-precision scheme might work, but it needs whole-model token and logit comparisons across varied prompts, memory and preprocessing measurements, and a fair latency distribution. A smaller weight file by itself does not show that the output is still the intended model.

## What is still outside the kernel?

The exact tokenizer reads a large `tokenizer.json`, splits text with a regular expression, applies byte-level BPE, and reconstructs bytes for output. C handles these text and I/O tasks; moving them into BPF would require a separate bounded implementation and would leave the expensive matrix work unchanged. C also decides which operator runs next and supplies the next matrix batch. Because the entire model is not resident in BPF-accessible memory, a single long-running kernel invocation could not simply advance through all of its layers using the present design.

The current KV map is another limit. Its layout reserves about 224 KiB per position; allocating all 40,960 configured positions would require roughly 8.75 GiB for KV alone. Attention can scan history in bounded chunks, but neither long-context memory use nor generation quality across arbitrary contexts has been established.

The evidence supports a narrow but concrete conclusion. On the test kernel, verifier-accepted BPF programs execute all 28 layers and choose real-model tokens; for `Hello, world!`, the first generated ID is `1096` (` This`), and the top five logits match the official BF16 reference. The reference comparison covers a small set of inputs, and the fixed-point full-vocabulary error is nonzero. The open engineering question is not whether a transformer can be expressed in eBPF. It is whether better weight formats, fewer boundaries, and bounded memory can make that expression useful for a workload where running in the kernel actually matters. The [source, commands, and measurements](https://github.com/eunomia-bpf/qwen3-ebpf) are available for that next test.
