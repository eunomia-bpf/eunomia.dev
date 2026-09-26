## 2026-09-25 eBPF Q&A run-report (eunomia-community-radar)

- Question: "Why does a full BPF hash map reject new keys with E2BIG, while updating an existing key still succeed?" (EN + ZH).
- Slug: `2026-09-25-bpf-hash-map-full-update-vs-insert`.
- Selection: no retained candidate and no new community material this run — the 7-day rolling snapshot across both watchlist-opted Slack archives returned zero messages (`snapshot=ok bytes=0 messages=0`; the empty-archive coverage gap is disclosed on both pages), and Discord / bpf@vger / r/eBPF remain visible-browser-only with no browser session available. Fell back to a genuine recurring practitioner boundary of the monitored community ("my map rejects new keys at capacity, yet updating the same key works" / "a key disappeared from my LRU map without a delete"); topic verified absent from the de-dup list (09-23's zeroing-on-delete page covers unlink semantics, not the capacity boundary).
- Source basis: torvalds/linux master @ 2026-09-25 — `kernel/bpf/hashtab.c` (`alloc_htab_elem` E2BIG at freelist exhaustion / `is_map_full` for new keys, the "when map is full and update() is replacing old element" comment, the `extra_elems` swap, `BPF_F_LOCK` in-place `copy_map_value_locked` shortcut), `kernel/bpf/bpf_lru_list.c` (`bpf_common_lru_pop_free` local → global-fetch-with-shrink → steal → ENOMEM), `include/uapi/linux/bpf.h` (the `E2BIG` contract text, `struct bpf_map_info`), `Documentation/bpf/map_hash.rst`, `kernel/bpf/syscall.c`.
- Content gate: `npm --prefix app run test:content` 82/82 pass, 0 fail.
- Commit A: `286ae01beb74` `docs(ebpf-qa): bpf-hash-map-full-update-vs-insert (2026-09-25)` on `main` (4 paths: EN/ZH pair + both indexes), pushed to origin/main; Pages run 36202826984 (`deploy-static-app`) completed `success`.
- Validator re-verify: receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-09-25.json`, `status=published` (`branch=ok`, `candidate_paths=ok`, `index_links=ok`, `privacy=ok`, `remote_contains_commit=ok`, `public=ok`; content-test/build/render/commit-push checks `skipped_already_published`).
- Live QA: H1 verbatim + body anchors verified on `https://eunomia.dev/ebpf-qa/2026-09-25-bpf-hash-map-full-update-vs-insert/` and `/zh/…` (fresh `?cb=` bust, browser UA); both index hrefs live on `/ebpf-qa/` and `/zh/ebpf-qa/`; fixed spin-lock and LRU-step wording confirmed live, stale wording absent.
- Public URLs:
  - EN: https://eunomia.dev/ebpf-qa/2026-09-25-bpf-hash-map-full-update-vs-insert/
  - ZH: https://eunomia.dev/zh/ebpf-qa/2026-09-25-bpf-hash-map-full-update-vs-insert/
