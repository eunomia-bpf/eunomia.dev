# Not Published / Pending Media Ledger

Last checked: 2026-11-28

This file tracks source posts that still need platform work. The latest checked `.github/publisher/posts_queue.txt` is empty; historical rows below are retained as candidate-source tracking, not proof that Medium/Dev.to are still queued.

For full per-platform counts across all configured source files, run `python .github/publisher/media/check_media_ledger.py --show-missing`. The canonical machine-readable ledger uses one JSON file per platform under [`platforms/`](platforms/).

2026-11-28: the `publish-posts` workflow (run 46088331081) published `docs/tutorials/28-detach/README.md` to DEV
(<https://dev.to/yunwei37/running-ebpf-after-application-exits-the-lifecycle-of-ebpf-programs-fhk>, `2026-11-28T00:31:40Z`, article id 5135832)
from `draft/media/2026-11-28/28-detach/devto.md`, so it is no longer a pending DEV source; Medium still has no copy of this source, and
Medium stays unpublishable while `MEDIUM_API_KEY` is absent and `/me/stories` is signed out.
The artifact is a byte-identical copy of the English sync source (9,248 bytes, same sha256) and so is the upstream
`bpf-developer-tutorial` copy, so this source needed no repair: 1 H1, 4 body H2, 1 H3, 2 H4, 12 labelled fences (1 `c`, 5 `bash`),
0 images, 0 tables, 0 relative links. The published `body_markdown` matches the artifact after the H1 strip character for character
(9,175 chars), and the visible page renders the exact title, 4 body H2, 1 H3, 2 H4, 6 `pre` blocks with their language labels,
0 images, 0 tables, 0 leaked fences, 4 tags, 0 reactions, 0 comments and the complete ending. Canonical is still DEV self, the
twenty-sixth consecutive API syndication with that defect. It used the LA 2026-11-27 normal slot. `.github/publisher/posts_queue.txt`
is empty again because the workflow pops each entry after a successful publish; the author listing still returned 88 posts without
the new copy after publication, while its per-article endpoint and public page served it, so the listing lag reproduced again.
2026-11-27: the `publish-posts` workflow (run 45915263748) published `docs/tutorials/24-hide/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-practical-tutorial-hiding-process-or-file-information-2f6n>, `2026-11-27T00:14:08Z`, article id 5128109)
from `draft/media/2026-11-27/24-hide/devto.md`, so it is no longer a pending DEV source; Medium still has no copy of this
source (its feed of 11 stories does not list it), and Medium stays unpublishable while `MEDIUM_API_KEY` is absent and
`/me/stories` is signed out.
The artifact repairs two defects of the English sync source that the Chinese source does not have: `README.md:368` repeats
`and then print the corresponding message.` as its own paragraph after the paragraph already ending with it (`README.zh.md:364`
has it once), and `:384` writes the repository link as `https: [//github.com/...`, which the live site renders as
`Full source code: https: //github.com/...` (`README.zh.md:380` keeps the clean `<https://...>` form); the upstream copy of this
file matches the local English source byte for byte (21,738 bytes), so both defects belong upstream. The rest of the post is a
byte-identical copy: 14 labelled code blocks (9 `c`, 1 `bash`, 1 `sh`, 3 `console`), 0 images, 0 tables, 0 relative links. The
published `body_markdown` matches the artifact after the H1 strip character for character (21,551 chars), and the visible page
renders the exact title, 4 body H2, 1 H3, 14 `pre` blocks with their language labels, 0 images, 0 tables, 0 leaked fences,
4 tags, 0 reactions, 0 comments and the complete ending. Canonical is still DEV self, the twenty-fifth consecutive API
syndication with that defect. It used the LA 2026-11-26 normal slot. `.github/publisher/posts_queue.txt` is empty again
because the workflow pops each entry after a successful publish; the public author listing had already caught up by the check
(88 posts including the new copy and the previous day's), so no listing lag reproduced this time.
2026-11-26: the `publish-posts` workflow (run 45729677157) published `docs/tutorials/22-android/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-using-ebpf-programs-on-android-cj2>, `2026-11-26T00:34:36Z`, article id 5120170)
from `draft/media/2026-11-26/22-android/devto.md`, so it is no longer a pending DEV source; its Medium copy was confirmed earlier.
The artifact repairs one defect of the English sync source that the Chinese source does not have: `README.md:44` fuses two list
rows as `... /data/eadb/deb.tar.gz`+ `rm -r /data/local/tmp/assets`, so the `rm` row never became its own bullet
(`README.zh.md:42-43` keeps both separate); the upstream copy of this file also matches it byte for byte. The published
`body_markdown` matches the artifact after the H1 strip character for character (12,595 chars), and the visible page renders
the exact title, 7 body H2, 1 H3, 2 H4, 4 `console` code blocks with language labels kept, the repaired `rm -r` row as its own
list item, 0 leaked fences, 0 images, 0 tables, 4 tags, 0 reactions, 0 comments and the complete ending. The three `[^...]`
footnote markers stay literal exactly as the live site renders them. Canonical is still DEV self, the twenty-fourth consecutive
API syndication with that defect. It used the LA 2026-11-25 normal slot. `.github/publisher/posts_queue.txt` is empty again
because the workflow pops each entry after a successful publish; the public author listing returned 86 posts without the new
copy while its per-article endpoint and public page served it, and the signed-out profile page reported 87 posts.
2026-11-25: the `publish-posts` workflow (run 45516590397) published `docs/tutorials/20-tc/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-20-tc-traffic-control-224e>, `2026-11-25T00:24:40Z`, article id 5112060)
from `draft/media/2026-11-25/20-tc/devto.md`, so it is no longer a pending DEV source; its Medium copy was confirmed in 2023.
The artifact is a byte-identical copy of the source (6,238 bytes, same sha256): this English source has no fence or punctuation
defect, no image, no table and no relative link, and all six fences carry `c`/`console`/`shell` labels, so nothing needed
repair. The published `body_markdown` matches the artifact after the H1 strip character for character (6,185 chars), and the
visible page renders the exact title, 5 body H2, 0 H3, 6 code blocks (`c` x2, `console` x3, `shell` x1), 0 leaked fences,
0 images, 0 tables, 4 tags, 0 reactions, 0 comments and the complete ending. Canonical is still DEV self, the twenty-third
consecutive API syndication with that defect. It used the LA 2026-11-24 normal slot. `.github/publisher/posts_queue.txt` is
empty again because the workflow pops each entry after a successful publish; the public author listing returned 84 posts
without the new copy while its per-article endpoint and public page served it.
2026-11-24: the `publish-posts` workflow (run 45320596815) published `docs/tutorials/19-lsm-connect/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-19-security-detection-and-defense-using-lsm-321l>, `2026-11-24T03:36:01Z`,
article id 5104605) from `draft/media/2026-11-24/19-lsm-connect/devto.md`, so it is no longer a pending DEV source; its Medium
copy has been confirmed since 2023. The artifact repairs one defect of the English sync source that the Chinese source does not
have: `README.md:154` ends the trace log with `16843009".` plus an inline fence, which closed the `console` block early and pushed
both `wget-7061` rows into prose (`README.zh.md:153-156` is clean). The published `body_markdown` matches the artifact after the
H1 strip character for character (8,000 chars), and the visible page renders the exact title, 7 body H2, 0 H3, 9 code blocks
(`console` x6, `conf` x1, `c` x1, `shell` x1), 0 leaked fences, the repaired trace block with both `wget` rows inside it, 0 images,
0 tables, 4 tags, 0 reactions, 0 comments and the complete ending. Canonical is still DEV self, the twenty-second consecutive API
syndication with that defect. It used the LA 2026-11-23 normal slot. `.github/publisher/posts_queue.txt` is empty again because
the workflow pops each entry after a successful publish.
2026-11-23: the `publish-posts` workflow (run 45087813444) published `docs/tutorials/17-biopattern/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-17-count-randomsequential-disk-io-11e7>, `2026-11-23T00:17:04Z`,
article id 5096382) from `draft/media/2026-11-23/17-biopattern/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact repairs four defects of the English sync source that the Chinese
source does not have: `README.md:145` terminates the `counters` map snippet with `.` instead of `;`, `:206` repeats `This
change may cause compatibility issues with eBPF programs on different versions of the kernel.` twice in one paragraph,
`:202-204` drop three of the four `Statistics update` sub-bullets back to the parent list level, and seven lines
(`:195`, `:197`, `:203`, `:226`, `:263`, `:265`, `:322`) close their inline code spans after the parenthesis instead of
before it, so `(filter_dev` and `true )` render as literal backticks. The published `body_markdown` matches the artifact
after the H1 strip apart from DEV's trailing-newline trim (16,768 vs 16,769 chars), and the visible page renders the exact
title, 4 body H2, 1 H3, 11 code blocks (`shell` x2, `c` x8, `console` x1), 0 images, 0 tables, 4 tags, 0 reactions,
0 comments and the complete ending. Canonical is still DEV self, the twenty-first consecutive API syndication with that
defect. It used the LA 2026-11-22 normal slot. `.github/publisher/posts_queue.txt` is empty again because the workflow
pops each entry after a successful publish.
2026-11-22: the `publish-posts` workflow (run 44965605572) published `docs/tutorials/15-javagc/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-15-capturing-user-space-java-gc-duration-using-usdt-4fd4>, `2026-11-22T00:14:19Z`,
article id 5090916) from `draft/media/2026-11-22/15-javagc/devto.md`, so it is no longer a pending DEV source; its Medium
copy has been confirmed since 2023. The artifact repairs four defects of the English sync source that the Chinese source does
not have: `README.md:47` fuses `solve the problem.` with `In Linux, USDT ...`, `:105` drops the `;` on the `perf_map`
`BPF_MAP_TYPE_PERF_EVENT_ARRAY` definition, `:185` fuses `respectively.` with `Finally, we have a LICENSE array ...`, and
`:317` loses the `## Summary` heading and renders `Summary.Through this introductory eBPF tutorial ...`. The published
`body_markdown` matches the artifact after the H1 strip character for character (16,143 chars), and the visible page renders
the exact title, 6 body H2, 3 H3, 10 code blocks (`shell` x5, `c` x4, `console` x1), 0 images, 0 tables, 4 tags, 0 reactions,
0 comments and the complete ending; all six outbound targets answer 200. Canonical is still DEV self, the twentieth
consecutive API syndication with that defect.
2026-11-21: the `publish-posts` workflow (run 44843640872) published `docs/tutorials/14-tcpstates/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-14-recording-tcp-connection-status-and-tcp-rtt-4ggl>, `2026-11-21T00:25:03Z`,
article id 5084990) from `draft/media/2026-11-21/14-tcpstates/devto.md`, so it is no longer a pending DEV source; its Medium
copy has been confirmed since 2023. The artifact repairs six defects of the English sync source that the Chinese source does not
have: `README.md:49-50` fuses the `dports` map behind an inline fence (the `c` block closes early and `dports` renders as
prose), `:176` opens a paragraph with a stray closing backtick, `:252` fuses the standalone `Instructions:` heading onto the
previous sentence, `:303-304` wrap two `tcpstates` output rows in stray quotes, `:351-352` end the `ecli` help block with
`framework".` and fuse `See https://github.com/eunomia-bpf/eunomia-bpf …` onto the closing fence, and `:387` fuses the
`256 -> 511` and `512 -> 1023` histogram rows. The published `body_markdown` matches the artifact after the H1 strip
character for character (21,856 chars), and the visible page renders the exact title, 5 body H2, 4 H3, 9 code blocks
(`shell` x2, `c` x4, `console` x3), 0 leaked fences, 0 images, 0 tables and the complete ending, with 0 reactions and
0 comments; all five outbound targets and the tail link answer 200. Canonical is still DEV self, the nineteenth consecutive
API syndication with that defect. It used the LA 2026-11-20 normal slot. `.github/publisher/posts_queue.txt` is empty
again because the workflow pops each entry after a successful publish.
2026-11-20: the `publish-posts` workflow (run 44644797186) published `docs/tutorials/13-tcpconnlat/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-13-statistics-of-tcp-connection-delay-with-libbpf-45kc>, `2026-11-20T00:18:40Z`,
article id 5077727) from `draft/media/2026-11-20/13-tcpconnlat/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact repairs four things the Chinese source does not need: the relative
`tcpconnlat1.png` becomes the 200-answering `_content-assets` URL, `README.md:209` opens the kernel comment as `/*".`
(artifact `/**`), `:544` fuses the build output into `.bpf.o".GEN-SKEL` (artifact splits it), and `:573` links the stale
`/tutorials/13-tcpconnect` slug (artifact `/tutorials/13-tcpconnlat`). The published `body_markdown` matches the artifact after
the H1 strip character for character (24,645 chars), and the visible page renders the exact title, 5 body H2, 3 H3, 10 code
blocks with the `c`/`console` labels preserved, 0 leaked fences, 1 loaded image, 0 tables and the complete ending, with 0
reactions and 0 comments; all five external link targets answer 200. Canonical is still DEV self, the eighteenth consecutive
API syndication with that defect. It used the LA 2026-11-19 normal slot. `.github/publisher/posts_queue.txt` is empty again
because the workflow pops each entry after a successful publish, and this time the public author API listed 80 posts
including the new one, so the multi-minute listing lag of 2026-11-17..19 did not reproduce.
2026-11-19: the `publish-posts` workflow (run 44429067976) published `docs/tutorials/12-profile/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-12-using-ebpf-program-profile-for-performance-analysis-4im2>, `2026-11-19T00:27:07Z`,
article id 5069111) from `draft/media/2026-11-19/12-profile/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact is a byte-identical copy of the source (18,575 bytes, same
sha256): this English source has no fence defect, no relative link, no image and no table, so unlike tutorials 4, 8
and 11 nothing needed repair. The published `body_markdown` matches the artifact after the H1 strip character for
character (18,490 chars), and the visible page renders the exact title, 2 body H2, 3 H3, 15 code blocks with the
`c`/`console`/`rust` labels preserved, 0 leaked fences, 0 images, 0 tables and the complete ending, with 0 reactions
and 0 comments; all five external link targets answer 200. Canonical is still DEV self, the seventeenth consecutive
API syndication with that defect. It used the LA 2026-11-18 normal slot. `.github/publisher/posts_queue.txt` is
empty again because the workflow pops each entry after a successful publish, and the public author API still listed
79 posts minutes after publication without 5069111, so the listing remains unusable as publish evidence.
2026-11-18: the `publish-posts` workflow (run 44210208198) published `docs/tutorials/11-bootstrap/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-11-develop-user-space-programs-with-libbpf-and-trace-exec-and-exit-2mci>, `2026-11-18T00:44:18Z`,
article id 5060707) from `draft/media/2026-11-18/11-bootstrap/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact repairs three defects of the English sync source that the Chinese
source does not have: `README.md:125` fuses `if (start_ts)` with its body, `:269` closes the `c` fence inline on a code
line (which swallowed the next nine lines and the `bootstrap.h` block into prose), and `:617` ends the `Compile and Run`
output with `(0ms)".`; the artifact splits the `if`, deletes the inline fence so the block count returns to the source's
19, and drops the stray `".`. The published `body_markdown` matches the artifact after the H1 strip character for
character (22,532 chars), and the visible page renders the exact title, 7 body H2, 2 H3, 19 code blocks with every fence
language preserved (16 `c`, 2 `shell`, 1 `console`), 0 leaked fences and the complete ending, with 0 reactions and
0 comments. Canonical is still DEV self, the sixteenth consecutive API syndication with that defect. It used the LA
2026-11-17 normal slot. `.github/publisher/posts_queue.txt` is empty again because the workflow pops each entry after a
successful publish, and the public author API still listed 77 posts minutes after publication, so the listing remains
unusable as publish evidence.
2026-11-17: the `publish-posts` workflow (run 43986441112) published `docs/tutorials/8-exitsnoop/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-8-monitoring-process-exit-events-output-with-ring-buffer-5ea2>, `2026-11-17T00:38:36Z`,
article id 5052614) from `draft/media/2026-11-17/8-exitsnoop/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact deviates from the source in one line: the source's `Run:` console
header ends with `COMM".` and the artifact drops the stray `".` (the Chinese source has the clean header). The published
`body_markdown` matches the artifact after the H1 strip character for character (6,901 chars), and the visible page renders
the exact title, 4 body H2, 1 H3, 5 code blocks with every fence label preserved, the complete ending, 0 reactions and
0 comments. Canonical is still DEV self, the fifteenth consecutive API syndication with that defect. It used the LA
2026-11-16 normal slot. `.github/publisher/posts_queue.txt` is empty again because the workflow pops each entry after a
successful publish. The same run waited 21 minutes for a runner (bpftime held 39-40 queued runs), and the public author
API still listed 77 posts 20 minutes after publication, so the listing is again not usable as publish evidence.
2026-11-16: the `publish-posts` workflow (run 43781972664) published `docs/tutorials/7-execsnoop/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-7-capturing-process-execution-output-with-perf-event-array-5d6c>, `2026-11-16T00:43:45Z`,
article id 5044058) from `draft/media/2026-11-16/7-execsnoop/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact is a byte-identical copy of the source (same sha256). The published
`body_markdown` matches the artifact after the H1 strip character for character (5,013 chars), and the visible page renders
the exact title, 3 body H2, 6 code blocks with all three fence languages preserved, and the complete ending with 0 reactions
and 0 comments. Canonical is still DEV self, the fourteenth consecutive API syndication with that defect. It used the LA
2026-11-15 normal slot. `.github/publisher/posts_queue.txt` is empty again because the workflow pops each entry after a
successful publish. Same run's listing check also found the duplicate CUDA tutorial copy (`2523222`, 2025-05-25) that
predates the ledgered `3135152`; both are public and the newer one stays the recorded mapping.
2026-11-15: the `publish-posts` workflow (run 43657764558) published `docs/tutorials/6-sigsnoop/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-6-capturing-signal-sending-and-store-state-with-hash-maps-12h9>, `2026-11-15T00:21:37Z`,
article id 5038589) from `draft/media/2026-11-15/6-sigsnoop/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact is a byte-identical copy of the source. The published
`body_markdown` matches the artifact after the H1 strip character for character (4,807 chars), and the visible page
renders the exact title, 2 body H2, 5 code blocks and the complete ending with 0 reactions and 0 comments. Canonical
is still DEV self, the thirteenth consecutive API syndication with that defect. It used the LA 2026-11-14 normal slot.
`.github/publisher/posts_queue.txt` is empty again because the workflow pops each entry after a successful publish.

2026-11-14: the `publish-posts` workflow (run 43568653176) published `docs/tutorials/4-opensnoop/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-4-capturing-opening-files-and-filter-with-global-variables-39bh>, `2026-11-14T07:02:12Z`,
article id 5034069) from `draft/media/2026-11-14/4-opensnoop/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed since 2023. The artifact changes one hunk of the source: lines 107-109 fuse the `trace_pipe`
command with its first output line through a stray `".\`, so the artifact restores the three correct lines. The published
`body_markdown` matches the artifact after the H1 strip character for character (8,100 chars), and the visible page renders
the exact title, 3 body H2, 6 code blocks, and the complete ending with 0 reactions and 0 comments. Canonical is still DEV
self, the twelfth consecutive API syndication with that defect. It used the LA 2026-11-13 normal slot. The dispatch waited
~6 h in the org-wide `ubuntu-latest` runner backlog and started 7 minutes after `publish-posts.yml` was pinned to
`ubuntu-24.04`; `.github/publisher/posts_queue.txt` is empty again because the workflow pops each entry after a successful
publish.

2026-11-13: the `publish-posts` workflow (run 43335255543) published `docs/tutorials/3-fentry-unlink/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-3-monitoring-unlink-system-calls-with-fentry-k28>, `2026-11-13T00:13:10Z`,
article id 5026194) from `draft/media/2026-11-13/3-fentry-unlink/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed as `cb6a374da48b` since 2023. The artifact copies the source except for one body link: the
source's `../2-kprobe-unlink/README.md` becomes `https://eunomia.dev/tutorials/2-kprobe-unlink/` so DEV does not render a
dead relative path. The published `body_markdown` matches the artifact after the H1 strip character for character (6,678
chars), and the visible page renders the exact title, 3 body H2, 1 H3, 9 code blocks, and the complete ending with 0
reactions and 0 comments. It used the LA 2026-11-12 normal slot. A manual dispatch was needed because the scheduled run has
no queue entry and does not invent one. `.github/publisher/posts_queue.txt` is empty again because the workflow pops each
entry after a successful publish.

2026-11-12: the `publish-posts` workflow (run 43126676054) published `docs/tutorials/2-kprobe-unlink/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-2-monitoring-unlink-system-calls-with-kprobe-259l>, `2026-11-12T00:21:38Z`,
article id 5019807) from a byte-identical copy at `draft/media/2026-11-12/2-kprobe-unlink/devto.md`, so it is no longer a
pending DEV source; its Medium copy was already confirmed as `bbdd8c020dbb` in 2023. The published `body_markdown` matches the
artifact body after the H1 strip character for character, and the visible page renders the exact title, 3 body H2, 8 code
blocks, the 10-item list, and the complete ending with 0 reactions and 0 comments. It used the LA 2026-11-11 normal slot. A
manual dispatch was needed because the scheduled run has no queue entry and does not invent one. `.github/publisher/posts_queue.txt`
is empty again because the workflow pops each entry after a successful publish.

2026-11-11: the `publish-posts` workflow (run 42929758642) published `docs/tutorials/1-helloworld/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-1-hello-world-framework-and-development-2kmb>, `2026-11-11T00:17:57Z`,
article id 5013503) from `draft/media/2026-11-11/1-helloworld/devto.md`, so it is no longer a pending DEV source; its
Medium copy has been confirmed as `33d4220022a4` since 2026-08. The artifact differs from the source only where the source
(and upstream `bpf-developer-tutorial`) joins the "Github Packages." bullet and the following "GitHub Actions" bullet on
one line; the artifact splits them so DEV renders two items. It used the LA 2026-11-10 normal slot. A manual dispatch was
needed because the scheduled run has no queue entry and does not invent one. `.github/publisher/posts_queue.txt` is empty
again because the workflow pops each entry after a successful publish.

2026-11-10: the `publish-posts` workflow (run 42732728828) published `docs/tutorials/0-introduce/README.md` to DEV
(<https://dev.to/yunwei37/ebpf-tutorial-by-example-0-introduction-to-core-concepts-and-tools-1j0f>, `2026-11-10T01:01:58Z`)
from `draft/media/2026-11-10/0-introduce/devto.md`, the source byte-for-byte apart from the source's stray closing `".` at the
end of its last sentence, so it is no longer a pending DEV source. It used the LA 2026-11-09 normal slot. The first
workflow trigger read an empty queue because the queue entry had only been committed locally; the queue and artifact were
pushed, then the run published. `.github/publisher/posts_queue.txt` is empty again because the workflow pops each entry
after a successful publish. The same check also confirmed three older tutorial DEV copies (cgroup, BPF Token, BPF dynamic
pointers) that had been live since 2026-02/03 but were missing from every ledger.

2026-11-09: the `publish-posts` workflow published `docs/blog/posts/claude-code-analysis.md` to DEV
(<https://dev.to/yunwei37/reverse-engineering-claude-codes-ssl-traffic-with-ebpf-5bod>, `2026-11-09T01:04:06Z`)
from a byte-identical copy at `draft/media/2026-11-09/claude-code-analysis/devto.md`, so it is no longer a
pending DEV source; its Medium copy was already confirmed as `1dde03bcc7ef`. The run used the LA 2026-11-08
normal slot, entered in `draft/plan/publishing-queue.zh.md`. `.github/publisher/posts_queue.txt` is empty again
because the workflow pops each entry after a successful publish.

2026-11-08: the `publish-posts` workflow published `docs/blog/posts/ai-lean-proof-checking-cost.md`
to DEV (<https://dev.to/yunwei37/ai-now-writes-lean-proofs-faster-than-lean-can-check-them-51mh>,
`2026-11-08T01:00:47Z`) from `draft/media/2026-11-08/ai-lean-proof-checking-cost/devto.md`, which differs from
the source only in two absolute-URL rewrites, so it is no longer a pending DEV source; its Medium copy is
still unpublished. The run used the LA 2026-11-07 normal slot.

2026-11-07: the `publish-posts` workflow published `docs/blog/posts/actplane.md` to DEV
(<https://dev.to/yunwei37/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf-2doh>, `2026-11-07T01:31:46Z`)
from a byte-identical copy at `draft/media/2026-11-07/actplane/devto.md`, so it is no longer a pending DEV source;
its Medium copy was already confirmed as `8077f9164a34`. The run used the LA 2026-11-07 normal slot, entered in
`draft/plan/publishing-queue.zh.md`. `.github/publisher/posts_queue.txt` is empty again because the workflow pops
each entry after a successful publish.

2026-11-06: the `publish-posts` workflow published `docs/blog/posts/schedcp-agentic-os.md`
to DEV (<https://dev.to/yunwei37/can-an-ai-agent-tune-the-linux-scheduler-inside-schedcp-and-sched-agent-4cj0>,
`2026-11-06T10:44:00Z`) from `draft/media/2026-11-06/schedcp-agentic-os/devto.md`, so it is no longer a
pending DEV source; its Medium copy is still unpublished. The run used the LA 2026-11-06 normal slot,
entered in `draft/plan/publishing-queue.zh.md`.

2026-11-04: the scheduled `publish-posts` workflow published `docs/blog/posts/gthulhu-vllm-cpu-pressure.md`
to DEV (<https://dev.to/yunwei37/when-does-cpu-scheduling-help-ai-inference-5did>, `2026-11-05T01:39:39Z`),
so it is no longer a pending DEV source; its Medium copy was published 2026-10-31 and re-verified as the
9-H2 source body. No new queue task was created: the entry that authorized this run was inserted for
LA 2026-11-04 in `draft/plan/publishing-queue.zh.md` and is now checked off.

## Current Publisher Queue

The machine ledger currently reports 41 unmatched Chinese sources on Zhihu and
31 on Juejin (78/109 mapped). These raw counts include paused material,
non-standalone index or reference pages, and duplicate source representations.
The ordered platform actions that remain suitable candidates are maintained
one-by-one in `draft/plan/publishing-queue.zh.md`; 23 Zhihu and 0 Juejin tasks
remain. The latest checked `.github/publisher/posts_queue.txt` is empty; the DEV backfill
continues with one tutorial per day. Each checkbox is one platform task. The normal target is one
publication per day; documented catch-up slots may add another platform, never
a second post on the same platform that day unless it redeems a catch-up gap.

The LinkedIn link shares for tutorials 50 through 54 are complete. There are no
remaining LinkedIn queue tasks. SchedCP, AgentCgroup, CPU noise, Weekly Analysis
reports, and content with confirmed LinkedIn publication records are not
included in this LinkedIn batch.

Medium and DEV have completed the queued BPFix, tutorials 50 through 54,
AgentCgroup, CPU noise, Agent Sandbox, ACRFence, and Runtime Security articles,
the 2026-11-08 DEV copy of the Lean proof-checking post, and the 2026-11-10 DEV
copy of tutorial 0 (the first of the historical tutorial backfill on that platform). Neither platform
has an unfinished queue task. Raw unmatched-source counts do
not authorize additional publication. Medium tutorial 53's flattened
Requirements table was repaired in place on 2026-08-30; no duplicate was created.

## Additional Platform Status

| Platform | Confirmed published | Not published / unresolved | Next action |
| --- | --- | --- | --- |
| X / Twitter | Ten self-authored posts are confirmed on `@yunwei37`; see `published.md`. The newest is <https://x.com/yunwei37/status/2122011756633665955> (2026-11-15T17:00:02Z), sharing `docs/blog/posts/bpf-gpu-server.md` with an eunomia.dev card that resolves to the 2026-10-17 blog route | The configured/planned `@eaborai` account currently showed `此账号不存在`; no full X archive export was done; the 2026-11-16 pass read only the first profile screen plus one status page, so the 2026-08-02 through 2026-11-14 stretch remains unenumerated | Replace or verify `@eaborai` in planning/metadata; paginate/export `@yunwei37` before declaring full X history complete |
| LinkedIn | Normal visible browser checks confirmed the `Yusheng Zheng` / `yunwei37` profile and tutorial 50-54 shares; see `platforms/linkedin.json` | Machine ledger maps 11/124 English sources; no unfinished LinkedIn queue task | Monitor existing posts; unmatched sources are not automatic publication tasks |
| Zhihu | Normal visible browser checks confirmed the fsession tutorial publication on 2026-08-14; machine ledger maps 68/109 Chinese sources | 41 configured Chinese sources remain unmatched; the visible profile was rechecked on 2026-09-23 and `/creator` still redirects to `/signin`, with no `z_c0` auth cookie, so the session is not signed in | Resume the first queued AgentCgroup task after a visible Zhihu session is signed in; recheck the title before submission |
| Juejin | xpu/npu-kernel-driver at <https://juejin.cn/post/7698947426490318888> (2026-10-21) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-21 normal slot with no catch-up gap redeemed, alongside xpu/gpu-kernel-driver at <https://juejin.cn/post/7698552541928554511> (2026-10-20) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-20 normal slot with no catch-up gap redeemed, alongside xpu/flamegraph at <https://juejin.cn/post/7698287001586040872> (2026-10-19) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-19 normal slot with no catch-up gap redeemed, alongside struct_ops at <https://juejin.cn/post/7697622920013709347> (2026-10-18) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-18 normal slot with no catch-up gap redeemed, alongside dynptr at <https://juejin.cn/post/7697186290753159202> (2026-10-17) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-17 normal slot with no catch-up gap redeemed, alongside bpf_wq at <https://juejin.cn/post/7697094073967116323> (2026-10-16) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-16 normal slot with no catch-up gap redeemed, alongside bpf_token at <https://juejin.cn/post/7696759034846445583> (2026-10-15) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-15 normal slot with no catch-up gap redeemed, alongside bpf_iters at <https://juejin.cn/post/7696384745624141824> (2026-10-14) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-14 normal slot with no catch-up gap redeemed, alongside bpf_arena at <https://juejin.cn/post/7695977967902933026> (2026-10-13) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-13 normal slot with no catch-up gap redeemed, alongside cgroup-tutorial at <https://juejin.cn/post/7695667191396040738> (2026-10-12) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-12 normal slot with no catch-up gap redeemed, alongside bpftrace-tutorial at <https://juejin.cn/post/7695068144415244323> (2026-10-11) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-11 normal slot with no catch-up gap redeemed, alongside 1-helloworld at <https://juejin.cn/post/7694934986782064649> (2026-10-10) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-10 normal slot with no catch-up gap redeemed, alongside 17-biopattern at <https://juejin.cn/post/7694534084639195174> (2026-10-09) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-09 normal slot with no catch-up gap redeemed, alongside 19-lsm-connect at <https://juejin.cn/post/7694131662616428571> (2026-10-08) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-08 normal slot with no catch-up gap redeemed, alongside 20-tc at <https://juejin.cn/post/7693579496968831022> (2026-10-07) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-07 normal slot with no catch-up gap redeemed, alongside 21-xdp at <https://juejin.cn/post/7693018382872330278> (2026-10-06) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-06 normal slot with no catch-up gap redeemed, alongside 22-android at <https://juejin.cn/post/7692860578532933642> (2026-10-05) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-05 normal slot with no catch-up gap redeemed, alongside 29-sockops at <https://juejin.cn/post/7691585964054003758> (2026-10-02) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-02 normal slot with no catch-up gap redeemed, alongside 28-detach at <https://juejin.cn/post/7691345821565141019> (2026-10-02) is confirmed public with full public-page QA, published through the visible editor as the second publication of the LA 2026-10-02 slot, redeeming the 2026-09-03 catch-up gap, alongside 27-replace at <https://juejin.cn/post/7691510191482519603> (2026-10-02) is confirmed public with full public-page QA, published through the visible editor as the third publication of the LA 2026-10-02 slot, redeeming the 2026-09-16 catch-up gap and leaving no outstanding catch-up gaps, alongside 26-sudo at <https://juejin.cn/post/7692143474695012402> (2026-10-03) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-03 normal slot with no catch-up gap redeemed, alongside 24-hide at <https://juejin.cn/post/7692741078310912027> (2026-10-04) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-04 normal slot with no catch-up gap redeemed, alongside 31-goroutine at <https://juejin.cn/post/7691326130511167526> (2026-10-01) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-01 normal slot with no catch-up gap redeemed, alongside 32-wallclock-profiler at <https://juejin.cn/post/7691151105851066391> (2026-09-30) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-30 normal slot with no catch-up gap redeemed, alongside 33-funclatency at <https://juejin.cn/post/7691081437695482937> (2026-09-29) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-29 normal slot with no catch-up gap redeemed, alongside 34-syscall at <https://juejin.cn/post/7690979496298637338> (2026-09-28) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-28 normal slot with no catch-up gap redeemed, alongside 35-user-ringbuf at <https://juejin.cn/post/7690813678851970571> (2026-09-27) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-27 normal slot with no catch-up gap redeemed, alongside 37-uprobe-rust at <https://juejin.cn/post/7690645355813357078> (2026-09-26) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-26 normal slot with no catch-up gap redeemed, alongside 38-btf-uprobe at <https://juejin.cn/post/7689852222630122548> (2026-09-25) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-25 normal slot with no catch-up gap redeemed, alongside 40-mysql at <https://juejin.cn/post/7691185608911134758> (2026-09-24) is confirmed public with full public-page QA, published through the visible editor as a gap-funded extra task on the LA 2026-09-24 slot, alongside 41-xdp-tcpdump at <https://juejin.cn/post/7691124509454822408> (2026-09-24) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-24 normal slot, alongside 39-nginx at <https://juejin.cn/post/7691334193581959952> (2026-09-24) is confirmed public with full public-page QA, published through the visible editor on the LA 2026-09-24 slot ahead of its targeted LA 2026-09-25 normal slot, alongside 36-kfunc at <https://juejin.cn/post/7688489073967251494> (2026-09-22, review cleared 2026-09-23), and 42-xdp-loadbalancer at <https://juejin.cn/post/7688532185121947658> (2026-09-23), joining Runtime Security, 48-energy, 47-cuda-events, and 46-xdp-test | No known unresolved Juejin publication defect; the machine ledger maps 78/109 sources, with 31 unmatched and 0 open rolling-queue tasks | Monitor [author profile](https://juejin.cn/user/4288563097635144/posts); keep pre-staging the next queue artifact before its slot |
| Medium | Does moving a GPU server's request loop into the kernel save CPU? at <https://medium.com/@yunwei356/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d8c070b5bd42> (2026-11-03T14:45:15Z) is the newest confirmed Medium item, published by the scheduled `publish-posts` workflow through the documented Medium API with topics `ebpf`/`gpu`/`kernel` and no `canonical_url`. Its public page carries the same three API-import defects as `medium-5001bb9afab6`: no rendered story title (`h1: 0`, body opens at the Mermaid block), the Mermaid fence kept as raw flowchart code, and the results table flattened into `| ... | — -|` prose rows; both paper links also resolve to <https://medium.com/papers/bpf-gpu-server.pdf> (403) because the source's relative link was sent unchanged. The body itself is complete (~1,976 words, 8 body H3 from the 8 source `##`, full ending, GitHub links intact) | <https://medium.com/@yunwei356/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d8c070b5bd42>. `medium-59aff1cc69a5` still renders the pre-restructure 10-H2 Gthulhu body | Repair titles and tables for both API-published stories once a logged-in visible Medium editor session or `MEDIUM_API_KEY` is available; `/p/d8c070b5bd42/edit` and `/p/5001bb9afab6/edit` both still redirect to `/m/signin` |
| DEV Community | eBPF Tutorial by Example 20: tc Traffic Control at <https://dev.to/yunwei37/ebpf-tutorial-by-example-20-tc-traffic-control-224e> (2026-11-25T00:24:40Z) is the newest confirmed DEV item, published from `docs/tutorials/20-tc/README.md` through `publish-posts` run 45516590397 from `draft/media/2026-11-25/20-tc/devto.md`, which is a byte-identical copy of the source (6,238 bytes, same sha256); the public `body_markdown` matches the artifact after the H1 strip character for character (6,185 chars), the visible page renders 1 exact-title h1, 5 body H2, 0 H3, 6 `pre` (`c` x2, `console` x3, `shell` x1) with every fence label intact, 0 leaked fences, 0 images, 0 tables and the complete ending, and the copy carries 0 reactions and 0 comments. Canonical is DEV self rather than eunomia.dev, the twenty-third consecutive API syndication with that defect; the upstream `media-publisher` payload rebuild is still the root cause. The backfill series runs one tutorial per day (LA slots) from `docs/tutorials/`; 79/128 English sources are mapped. |
| Reddit | Historical `u/yunwei123` posts are confirmed for eBPF tutorial, GPTtrace, Code-Survey-like discussion, and Wasm-bpf; see `published.md` | No evidence yet that current 2026 flagship posts were submitted to r/eBPF, r/netsec, r/LocalLLaMA, HN, or lobste.rs | Use Reddit only manually for flagship posts; check subreddit fit before posting |
| Xiaohongshu / RedNote | None confirmed | No account URL or published note URL found; current browser search requires login; public search found no clear `eunomia.dev` / `eunomia-bpf` / `bpftime` / `AgentSight` result | Treat as not started; create account and visual-note workflow only after there is image-card/video capacity |

## Pending Verification

These items need follow-up before moving to `published.md` or clearing them:

| Item | Why it needs verification | Suggested check |
| --- | --- | --- |
| Full Zhihu article history | Profile reports 114 articles; normal visible scrolling collected 113 unique links and exact-title matching resolved two tutorial/blog duplicates | Recheck visible title/keyword matches immediately before publishing any remaining Zhihu-missing tutorial; do not use APIs or hidden endpoints |
| Full X history | `from:yunwei37` search recorded visible project hits, but not a complete account export | Use X advanced search or account data export before declaring the ledger complete |
| Full Reddit history | `author:yunwei123` search recorded visible project hits, but not all comments or deleted/crossposted content | Use Reddit user listing and subreddit search for `eunomia.dev`, `eunomia-bpf`, `bpftime`, `AgentSight` |
| Xiaohongshu login/account | Search results were hidden behind login and no account URL is known | Sign in or provide account URL before doing a definitive platform audit |
| Repo-referenced platform URLs | Some links appear in documentation as references; they may be our posts, partner posts, or citations | Open each URL and confirm author/account before marking as confirmed |

## Add New Pending Items

Use this format when adding a post:

```md
| `source/path.md` | Human title | Not queued / queued | Not published / draft / published URL | Not published / draft / published URL | Next concrete action |
```
