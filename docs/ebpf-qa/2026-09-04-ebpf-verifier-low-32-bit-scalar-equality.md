# Why can the eBPF verifier lose a relationship that is true for the low 32 bits?

**Short answer:** eBPF registers are 64 bits wide, while a 32-bit instruction defines only a particular relationship between the low halves. A 32-bit move zero-extends its result, so after `w7 = w6`, `r7` is `zero_extend(low32(r6))`; it is not necessarily equal to `r6` as a 64-bit value. A 32-to-64-bit signed move has a different relationship again: its upper half is the sign extension of bit 31. If the verifier can represent only full-width equality, retaining the old equality would be unsound, so it must discard the link. Later branch information about one register then cannot refine the other, even though their low 32 bits still match at runtime.

This is a precision limitation, not proof that the program is unsafe. The verifier must reject whenever its abstract state cannot prove the required bound or return-value constraint. A current RFC proposes distinct low-32 equality links for zero-extending and sign-extending copies, but that proposal is still under review. Until the running kernel contains equivalent support, write the program so the branch constrains the same converted value that is later consumed.

## The runtime fact is narrower than 64-bit equality

Every general eBPF register is 64 bits. The ISA nevertheless distinguishes `ALU64` operations from 32-bit `ALU` operations. The [eBPF instruction-set specification](https://docs.kernel.org/bpf/standardization/instruction-set.html) defines a 32-bit register move as:

```text
r7 = (u32)r6
```

The cast is important: the low 32 bits are copied and the upper 32 bits of `r7` become zero. If `r6` is `0xffff_ffff_0000_0001`, then `r7` becomes `0x0000_0000_0000_0001`. The values are unequal as 64-bit integers even though their low halves are equal.

`MOVSX` has another contract. A 32-to-64-bit signed move reconstructs the high half from bit 31 of the low half. If that bit is one, the destination's upper half becomes all ones; otherwise it becomes all zeros. Therefore these three claims are different:

- `rA == rB` over all 64 bits;
- `low32(rA) == low32(rB)` and `high32(rB) == 0`; and
- `low32(rA) == low32(rB)` and `rB == sign_extend(low32(rA))`.

A verifier that conflates them can accept an unsafe program. A verifier that keeps only the first kind of relation must conservatively forget the relationship after a subregister conversion.

## Why losing the link can cause a false rejection

Consider this simplified instruction sequence:

```text
r6 = unknown_64_bit_scalar
w7 = w6
if w6 != 0 goto reject
if w7 == 0 goto ok
```

On the fall-through path of the first branch, the low 32 bits of `r6` are zero. Because `w7 = w6` zero-extended those same bits, `r7` must be the 64-bit constant zero. The second branch is therefore predictable at runtime.

The verifier does not execute concrete values. It symbolically tracks each register's signed and unsigned bounds plus a `tnum` that records known and unknown bits. The [verifier documentation](https://docs.kernel.org/bpf/verifier.html#register-value-tracking) explains how branches narrow those states. It also tracks identities between related values so knowledge learned from one copy can sometimes propagate to another.

If that identity means only “equal as 64-bit scalars,” it cannot survive `w7 = w6`: the high halves deliberately differ. When the first branch narrows `w6`, the verifier has no remaining reason to update `r7`. The verifier may then report that a callback return value is outside its required range, widen an errno-or-zero result back to a broad signed range, or explore too many loop states. Those are downstream symptoms; the missing low-half relation is the mechanism.

Compiler output determines whether this appears. Source that looks equivalent can produce different mixes of `MOV32`, `MOVSX`, and 64-bit operations. The active kernel discussion reports that sign-extension-heavy output exposes the gap more often, while another compiler may optimize the conversion away. This is why the useful artifact is the actual BPF instruction stream, not only the C source.

## What a sound low-32 link must remember

The active [low-32 scalar-linking RFC](https://lore.kernel.org/bpf/20260814231945.3884596-1-vineet.gupta%40linux.dev/T/#t) proposes keeping a shared scalar identity together with an explicit relationship flavor. Conceptually, a zero-extension link states:

```text
dst.low32  = src.low32
dst.high32 = 0
```

A sign-extension link instead states:

```text
dst.low32  = src.low32
dst.high32 = repeat(src.bit31, 32)
```

When a branch narrows the shared low half, the verifier can rebuild the destination's 64-bit range and known-bit state using the appropriate rule. It must not copy the source's unknown high bits into a zero-extended destination, and it must not assume zero extension for a signed destination.

The link also has to participate in every operation that decides whether two verifier states are compatible. The verifier prunes exploration when a previously accepted state safely covers the current one. As the [kernel verifier documentation on pruning](https://docs.kernel.org/bpf/verifier.html#pruning) describes, this comparison includes registers and spilled stack state. If pruning ignores whether a link is full-width, zero-extending, or sign-extending, it may merge states that imply different values. That would be a soundness bug rather than a harmless loss of precision.

There are four review boundaries to check:

1. **Creation:** form the link only for an instruction whose ISA semantics guarantee the relationship.
2. **Propagation:** when one low half narrows, update only facts logically implied for the other register; do not invent information about the source's high half.
3. **Invalidation:** clear or replace the link after writes, spills, fills, arithmetic, or casts that no longer preserve it.
4. **State comparison:** include the link flavor and identity in exact-state and safe-subset checks before pruning a path.

The current kernel source keeps the abstract register state in [`struct bpf_reg_state`](https://github.com/torvalds/linux/blob/master/include/linux/bpf_verifier.h) and implements state comparison and pruning in [`kernel/bpf/states.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/states.c). Those sources, rather than a version-number guess, are the right way to determine what a particular kernel tree actually supports.

## How to diagnose this class of rejection

Start from the rejected object and kernel, because a source-level cast does not prove which instruction form survived compilation.

1. **Save the full verifier log.** Locate the first place where a range becomes unexpectedly broad or a related register stops narrowing. The final rejection often appears much later than the precision loss.
2. **Disassemble the BPF object.** Identify whether the relevant copy is `MOV32`, 32-bit `MOVSX`, 64-bit `MOV`, or another ALU operation. Also note whether the later comparison is a 32-bit or 64-bit jump.
3. **Track both halves separately.** For each step, write down `low32`, the destination's high-half rule, and which register the branch actually constrains. Do not summarize `w7 = w6` as `r7 = r6`.
4. **Compare compiler outputs only as evidence.** If one optimized build loads and another does not, diff their BPF instructions. That can identify the conversion pattern, but it does not establish that the rejected program is invalid.
5. **Check the running kernel implementation.** Search its verifier state and selftests for the proposed low-32 relationship support. An RFC timestamp or a newer compiler alone does not mean the feature is present.
6. **Reduce without erasing the conversion.** Preserve the move, the narrowing branch, and the failing use in a minimal test. A reduction that optimizes away the subregister operation no longer tests the same verifier limitation.

Useful evidence includes the exact instruction numbers where the copy and branch occur, the register state immediately before and after them, the compiler and optimization level, and the running kernel commit. Raw production data is not needed.

## Practical source-level workarounds

Until the target kernel can preserve the relationship, arrange the program so proof does not depend on propagation through a lossy subregister link.

- **Compare the converted value.** If later code consumes the zero-extended or sign-extended destination, apply the range check to that destination too.
- **Normalize once and keep one value.** Store the intended `u32` or `s32` result in a variable with the correct semantics, then use that same value for the branch and the return, index, or loop condition.
- **Avoid unnecessary signed round trips.** If the domain is genuinely unsigned, keep it unsigned. Do not change signedness merely to satisfy the verifier when negative values are meaningful.
- **Keep callback returns explicit.** Convert a broad intermediate into the callback's documented return domain at the final decision point, rather than expecting a constraint on an earlier alias to propagate.
- **Verify the generated instructions after each rewrite.** Optimizers may coalesce variables or reintroduce an extension, so a source edit is not evidence that the problematic pattern disappeared.

Do not silence the rejection by inserting arbitrary masks or truncations unless they preserve the program's intended numeric domain. A verifier workaround that changes negative error handling, wrap behavior, or loop termination is a functional bug.

## How to validate a kernel-side fix

A complete fix needs more than one accepting example. Selftests should cover:

- zero-extending and sign-extending copies;
- source-driven and destination-driven narrowing without inferring unrelated high bits;
- equality and inequality branches at both 32- and 64-bit widths;
- spill/fill and overwrite invalidation;
- links combined with constant deltas;
- loop convergence and state pruning;
- incompatible link flavors at the same checkpoint; and
- accepted and deliberately rejected programs under multiple compiler code-generation patterns.

Run the full BPF selftest suite, not only new positive tests. A precision improvement may make more programs load while also changing pruning, loop exploration, or stale-link behavior. Any new acceptance must be explained by the intended low-32 relation, and every negative test must remain rejected for the same safety reason.

## References

- [BPF mailing-list RFC: track scalar equality across the low 32 bits](https://lore.kernel.org/bpf/20260814231945.3884596-1-vineet.gupta%40linux.dev/T/#t)
- [Linux kernel documentation: eBPF instruction-set semantics](https://docs.kernel.org/bpf/standardization/instruction-set.html)
- [Linux kernel documentation: verifier register-value tracking](https://docs.kernel.org/bpf/verifier.html#register-value-tracking)
- [Linux kernel documentation: verifier state pruning](https://docs.kernel.org/bpf/verifier.html#pruning)
- [Linux kernel BPF design Q&A: 32-bit subregister requirements](https://docs.kernel.org/bpf/bpf_design_QA.html#q-bpf-32-bit-subregister-requirements)
- [Linux kernel source: `struct bpf_reg_state`](https://github.com/torvalds/linux/blob/master/include/linux/bpf_verifier.h)
- [Linux kernel source: verifier state comparison and pruning](https://github.com/torvalds/linux/blob/master/kernel/bpf/states.c)
- [GitHub documentation: repository custom instructions for Copilot](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions)

## Community discussion today

The visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected verifier discussion was active within the strict 24-hour window, so the seven-day fallback was not used. This synthesis removes participant, account, employer, project, and channel identities; message links; exact times; private topology; raw logs; and searchable original wording. No raw transcript was retained, and no reply, reaction, direct message, follow, invitation, or moderation action was performed.

### Subregister precision was the strongest technical question

The kernel discussion returned to a verifier gap where a 32-bit copy preserves the low half but changes how the high half is constructed. The practical symptom was a program that is safe for a narrow callback return or loop bound at runtime, yet loses the proof after sign or zero extension. Review attention was not limited to accepting more programs: it focused on whether link flags participate in state comparison, whether stale relationships are cleared, and whether verifier logs expose the new state. That emphasis is important because an incorrect pruning merge can become an acceptance bug, whereas dropping the relationship merely rejects a valid program.

Other active kernel topics included scalar-bound inference, integer overflow in map operations, borrowed-reference lifetime, JIT memory checking, socket-map accounting, and flaky randomized selftests. Their common theme was boundary evidence: arithmetic width, reference ownership, or nondeterministic inputs must remain explicit at every verifier or test transition.

### Maintainers discussed guardrails for AI-generated pull requests

An observability project discussed how to classify unusually large, machine-generated changes without blocking useful contributions. The reported failure mode was reviewer overload: a long description and large diff can create an appearance of completeness while making scope, reproduced behavior, and test evidence harder to identify. Suggestions included repository-specific review instructions and warning-only metadata checks based on change size or structure.

Those signals are useful for triage, but none proves that a patch is correct or AI-generated. A defensible gate should therefore ask for a bounded problem statement, reproduced behavior, reviewable commits, and tests tied to the claimed change. Size-based automation should classify or request review rather than automatically reject. Public repository instructions can make expectations consistent, but the final decision still needs human examination of code and evidence.

### The remaining targets were accessible but quiet

The approved project help and feature areas had no new technical question in the window. A scheduler support area contained an older installation workaround, but it was outside the strict window and was not needed as fallback. An instrumentation discussion had only coordination activity around a stabilization effort. The eBPF chat channel's newest technical question had already been answered in an earlier daily Q&A, and the newest forum post likewise repeated yesterday's networking topic. These are checked-but-quiet results, not coverage gaps reported as zero activity.
