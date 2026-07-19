# bpfix Blog Figure Inventory

Source: `docs/papers/bpfix.pdf` and `docs/papers/bpfix.txt`, arXiv 2607.02748.

| Paper item | Claim supported | Blog decision | Asset or handling |
|---|---|---|---|
| Figure 1 | A verifier terminal line stops at the symptom, while bpfix reconstructs the missing proof and maps it back to source. | Include after the opening packet example because it directly supports the article thesis. | `draft/blog/imgs/bpfix-figure-1-diagnostic-gap.png` |
| Figure 2 | Correct source can be rejected because compiler lowering hides verifier-visible pointer provenance. | Omit as a standalone figure because the compiler-lowering layer distinction is now explained in prose, while Figure 1 and the text carry the main running example. | Mentioned in text through the compiler-lowering discussion. |
| Figure 3 | bpfix turns verifier logs and optional source/object metadata into proof reconstruction and repair-oriented diagnostics. | Include in the mechanism section because it gives the reader a compact mental model of the workflow. | `draft/blog/imgs/bpfix-figure-3-workflow.png` |
| Figure 4 | A real aya rejection needs a map-value proof derived through a correct helper. | Omit because it is a narrower instance of the proof-family idea and would require extra setup for aya-specific context. | Covered by the map-value pointer examples in prose. |
| Figure 5 | The same terminal message can correspond to a source bug or a compiler-lowering artifact. | Omit for now to keep the article from becoming a paper-shaped gallery. Keep as a candidate if the post needs a second concrete ambiguity example. | The ambiguity result is represented in the debugging table and prose. |
| Figure 6 | Replacing raw verifier logs with bpfix diagnostics improves LLM repair success across three models. | Include in the LLM section because it supports the claim that proof localization helps automated repair. | `draft/blog/imgs/bpfix-figure-6-repair-success.png` |
| Table 1 | The 191 source bugs span 12 root-cause categories, mostly eBPF-specific. | Use in prose and a reader-facing debugging table instead of embedding the paper table image. | Numbers summarized in the empirical section. |
| Table 2 | Common terminal message templates span multiple root causes. | Use in prose and the debugging table because the article needs the ambiguity result, not the raw table layout. | `R# invalid mem access 'scalar'` result cited in text. |
| Table 3 | bpfix mainly reduces verifier-load and source-semantics failures in LLM repair. | Use in prose because the stage breakdown is easier to interpret in text than as a table image. | Stage-level interpretation in the LLM section. |
