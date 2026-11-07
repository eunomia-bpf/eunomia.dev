---
date: 2026-11-06
slug: ai-lean-proof-checking-cost
description: Re-checking a stored Lean proof normally skips the search that found it, but a proof by computation makes the kernel run the computation again. Measurements on OpenAI's 163,096-module AI-generated Lean library show both cases, how heavily AI-written proofs rely on computation, what compiling the library costs, how much of it a default build leaves out, and what Lean's own checker actually re-checks.
---

# Why re-checking some AI-written Lean proofs costs almost as much as compiling them

OpenAI has published a [repository of AI-generated mathematics](https://github.com/openai/math) whose results come with Lean proofs. At the revision we studied it contains 163,096 Lean modules, about nineteen times as many files as the Mathlib snapshot it builds on. Before anyone reuses a library like this, someone has to check it again, outside the environment that produced it. The common expectation is that this is cheap. A Lean tactic may search for minutes before it finds a proof, but Lean stores only the final proof term, and an independent checker validates that term without repeating the search.

That expectation holds for ordinary mathematical proofs. It breaks for a style of proof that AI-generated Lean uses heavily: proof by computation. When a proof says, in effect, "evaluate this finite check and confirm that the answer is `true`", the stored term is tiny, and the Lean kernel has to run the whole evaluation again every time the proof is checked. For those proofs, checking means recomputing.

We measured both cases on this library. We compiled 162,170 of its modules one at a time and recorded the cost of each, timed the kernel check of selected modules against their compilation, counted how often computational proofs appear, and compared the files a default build compiles with the files the repository contains. This post covers what we learned about Lean and about AI-generated proofs. The six-page technical report is available as a [PDF](../../papers/proofserve.pdf).

<!-- more -->

## What re-checking a Lean proof does

Lean compiles a source file in two stages. The elaborator runs tactics such as `simp`, `omega` or `linarith`, which may search, backtrack and call decision procedures, and produces a proof term: a fully explicit object that records only the steps that worked. The kernel then type-checks that term, and Lean writes the result into an `.olean` file. An independent check, such as Lean's bundled `leanchecker` or the `lake check` command added in [Lean 4.35](https://lean-lang.org/doc/reference/latest/releases/v4.35.0/), loads the stored terms and asks the kernel to check them again. Tactics never run during that second check, so whatever the search cost stays behind.

Proof by computation stores something different. For a decidable proposition `p`, the `decide` tactic produces a proof that says "`decide p` evaluates to `true`". The term is short, but the kernel can accept it only by reducing `decide p` until it reaches `true`, which means running the decision procedure on the actual finite object: every case of a combinatorial statement, every entry of a table, every coloring of a small graph. Plain `decide` evaluates the instance twice during compilation, once in the elaborator and once in the kernel; `decide +kernel` skips the elaborator's pass and leaves a single evaluation to the kernel. In both forms the kernel's evaluation is part of the proof check, so every re-check runs it again.

## Two kinds of proofs, two very different checking costs

To see the difference, we compared four modules under the same measurement boundary. For each module we timed reading and kernel-checking that one module after its dependencies were already loaded, and divided by the wall time of compiling the module from source. All runs used Lean 4.34.1 on the same eight performance cores of an Intel Core i9-14900K, one task at a time.

| Module | Kind of proof | Re-check time as a share of compile time |
|---|---|---:|
| `Grid` (AI library) | computational certificate | 80.28% |
| `Amplitudes` (AI library) | computational certificate | 82.35% |
| Mathlib `Nat.Basic` | ordinary proofs | 6.54% |
| `NearPin` (AI library) | ordinary proofs | 0.46% |

`Grid` and `Amplitudes` spend 255.5 and 110.0 seconds in their re-check, almost all of it inside the kernel's replay call. The two ordinary modules re-check in a small fraction of their compile time, as expected. For the two certificates, a re-check redoes about four fifths of the compilation, because most of the compilation was the computation.

Across more modules, the line between the two cases blurs. Over 31 modules where we measured complete compiler and checker processes, checking used a median of 45.1% of the compiler's CPU time. Modules with an explicit `decide +kernel` had a median of 49.8%, modules without one 39.1%, and the two groups overlap. Part of the reason is that plain `decide` and other tactics can also leave large evaluations in a proof term. A Ramsey K10 certificate with no `decide +kernel` in its source took 356 CPU-seconds to check against 1,725 to compile (20.7%). A marker in the source is a useful hint about checking cost, and only a measurement settles it.

Averages over a whole project hide these modules. Checking the full dependency closure of one development, Falconer, cost 6.31% of its compilation CPU, because many cheap dependencies dilute a few expensive certificates. When you plan a re-check, budget for individual certificates: a project-wide ratio says little about how long one of them takes, and a short source file can be among the most expensive to check.

## AI-written Lean relies on computation far more than Mathlib does

To see how common these proofs are, we counted `decide +kernel` calls after masking comments and strings. The AI library contains 1,354,379 of them in 5,848 files. The Mathlib snapshot it builds on contains seven, in four of its 8,529 files. Normalized by source size, the AI library uses this construct 10,668 times more densely.

We did not test why. One likely factor is that many statements in the AI library are about specific finite objects, where exhaustive evaluation is a proof that search can produce reliably, while Mathlib is organized around general theorems. The count is lexical: it shows where computation is likely, and the CPU cost of any particular call still has to be measured.

## What it costs to compile the whole library

We compiled the library module by module on that i9-14900K, using its 8 performance cores and 14 of its 16 efficiency cores, and recorded each module's CPU time on its own. Of the 163,096 modules, 162,170 (99.43%) compiled. No module ended in a final failure. We set aside the remaining 926: a few families, including `PadicHodge`, `Ramsey.CycleClique.Construction` and `BigCrystalline`, kept exhausting the memory available to their build jobs on a 64 GB machine, so we deferred those families and every module that depends on them.

The completed part took 31.5 hours of wall time, 169.0 CPU-hours on performance cores and 302.4 CPU-hours on efficiency cores. We report the two core types separately because an hour on each is a different amount of work, and we did not convert one into the other. Both numbers are lower bounds for the whole library. The deferred modules are the memory-hungry ones, which are likely to be among the most expensive, and the build followed dependency order rather than a random sample, so the completed part cannot be used to estimate the rest.

The cost is concentrated in a small number of modules, and file size is a poor guide to which ones. The median module took 4.0 CPU-seconds on a performance core, and the most expensive 1% of modules used about a quarter of all CPU time. The 5,848 files that contain `decide +kernel` are 3.6% of the compiled modules and used about 18% of the CPU on each core type. The single most expensive module, `OAI.Analysis.FreeWords.UltraExpectationFull`, is a 15.9 KB file with no `decide +kernel`. It took 9,314 CPU-seconds on an efficiency core. A 5.3 MB Ramsey certificate file with 11,114 `decide +kernel` calls took 2,241 CPU-seconds on the same core type. We did not measure how much of that 15.9 KB file's cost a re-check would repeat.

An earlier independent [report on the same library](https://github.com/nealmcb/openai-math-lean-tooling/blob/c9570a82f79b78a1a969a6316ac0fdd69aabef55/REPORT.md) built 122,458 modules of an earlier revision in 13.8 hours on 32 cloud vCPUs, and estimated about 410 CPU-hours in total, including about 330 hours of compilation extrapolated from 26 timed modules. Different revisions, hardware and scope rule out a direct comparison. What a per-module census adds is a measured cost for each of the 162,170 modules in place of an extrapolation. The per-module records are in [the repository](https://github.com/yunwei37/proofserve/tree/main/data/openai-math-4eaee875/census).

## A successful build covers only part of the library

A successful `lake build` is the usual evidence that a Lean project checks. A default build compiles the modules reachable from the project's declared root targets. In this library, those roots reach 130,834 of the 163,097 source units (counting the build configuration). The other 32,263 files, 19.78% of the total, are never compiled by a default build. They contain 451,008 of the 1,354,379 `decide +kernel` calls, or 33.30%. A third of the library's computational proofs sit in files that a successful default build skips.

The earlier report found the same pattern at its revision: 30,697 files outside the default graph, about a quarter of its source inventory. Anyone who reports that a library like this "checks" should say which roots they built and how many files those roots reach.

## Lean behavior worth knowing before checking a large library

**Lean's `-M` memory limit is expensive on computational proofs.** A build or checking service that must protect itself from runaway memory use may reach for Lean's `-M` option, a memory limit in megabytes. With `-M` set, Lean's memory check calls `getrusage` once every 200 times it runs, and the kernel runs that check throughout reduction. `getrusage` sums resource use over every thread of the process, so each call does real work. Compiling `Grid` with eight threads issued about 4.0 million `getrusage` calls. Removing `-M` while keeping the same 15 GiB limit outside Lean, as a cgroup, cut `Grid`'s compile time from 622 to 306 seconds and its system time from 436 to 107 seconds, as medians of three runs. `UMoments9` went from 142 to 88 seconds, while `EnvelopePsiB`, an ordinary module, barely moved (64.4 to 63.5 seconds). Every output was byte-identical. For computation-heavy proofs, a memory limit enforced outside the Lean process gives the same protection without this accounting cost.

**`leanchecker` can spend almost all of its time finding files.** In its default mode, the bundled checker first enumerates every `.olean` file on the search path and resolves each file name to a module name, and only then picks out the requested root. In the AI library's build environment, whose search path has 90 entries and yields 203,273 `.olean` path entries (some roots repeat), checking Mathlib's `Mathlib.Data.Nat.Basic` took 322.9 seconds, 234.6 of them in system CPU. With a search path that holds only the 1,330 modules this root imports, the same check took 0.39 seconds. A probe that runs only the two discovery steps spent 2.6 seconds enumerating directories and 271.4 seconds resolving module names. These are single observations in an unusually large environment, and the remedy is cheap: give the checker a search path that contains only the closure you want to check.

**The default mode and `--fresh` answer different questions.** The default mode trusts the imports and re-checks only the declarations that the target module adds. `--fresh` takes the final merged environment, every constant the root can see, and replays all of it into an empty kernel environment. Their verdicts on the same artifact can differ. After we removed one module's `.olean.private` part, the default mode still accepted it, while `--fresh` failed because it could not load that part. A proof written with the deprecated native-evaluation bridge `Lean.ofReduceBool` passed the default mode and failed under `--fresh`, which does not carry over the compiled code the kernel needs to run that evaluation.

**Checking the merged environment differs from checking every module.** When two modules prove a theorem with the same name and statement, importing both keeps a single body, the one from the module imported last. We corrupted that body in one module by editing its serialized file, something normal compilation never produces. Checked as its own root, the corrupted module was rejected in both modes. A root that imports the good module first and the corrupted one second was rejected by `--fresh`. With the import order reversed, the good body replaced the corrupted one in the merged environment, and `--fresh` accepted the root. The default mode, which does not re-check imports, accepted both roots. To know that every module's proofs pass the kernel, check each module as its own root.

None of these cases involves normally compiled source proving a false statement; they show what each command actually re-checks. Minimal inputs and scripts for all three are in the [repository](https://github.com/yunwei37/proofserve/tree/main/data/lean-observations).

## What to take away

If you plan to re-check an AI-generated Lean library, expect two kinds of cost. Ordinary proofs re-check in a small fraction of their compile time; computational certificates re-run their computation, so budget for them one by one instead of trusting a project average. Count how much of the library relies on computation before you size the job, state which roots your build or check actually covered, and enforce memory limits outside the Lean process.

## Where these results stop applying

All measurements come from one machine, one revision of one library, and Lean 4.34.1. The cost comparison uses a handful of selected modules, and the 31-module panel shows a spectrum rather than two clean classes. Counts of `decide +kernel` are lexical proxies for computation, not runtime measurements. The full-build cost is a lower bound over a dependency-ordered subset that leaves out the most memory-hungry families, and nothing here extrapolates it to the missing modules.

## The tool

We packaged this workflow as [ProofServe](https://github.com/yunwei37/proofserve), an open-source command-line tool; [version 0.1.0](https://github.com/yunwei37/proofserve/releases/tag/v0.1.0) ships as a prebuilt Linux x86-64 bundle. `proofserve check` re-checks a compiled Lean project with Lean's own kernel, in parallel and resumably, and writes a receipt that lists what it accepted and which axioms the accepted modules declare; with eight workers it shortened wall time by 1.7 to 2.1 times on our test projects, at a higher total CPU cost. `proofserve audit` reports which source files and `decide +kernel` calls a default build leaves out. The README describes its trust boundary, including the default setting that favors speed over exact agreement with the official checker. The six-page report is [here](../../papers/proofserve.pdf).
