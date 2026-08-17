# Media Publishing Notes

Last checked: 2026-08-17

This folder records cross-platform publishing state, public-page QA, and
community feedback for eunomia.dev content.

Canonical machine-readable ledgers use one JSON file per platform under [platforms/](platforms/), with shared source scanning rules in [sources.json](sources.json). To refresh the per-platform coverage count, run:

```bash
python .github/publisher/media/check_media_ledger.py --show-missing
```

## Current Setup

Medium and DEV.to are API-first. Use their documented endpoints with local
`MEDIUM_API_KEY` and `DEV_TO_API_KEY` credentials, or the approved secret-backed
publisher in `.github/publisher/`. Never print or commit credentials. Check for
an exact-title duplicate before creation, then inspect the complete public page
in a normal visible browser after publication.

The legacy shared `publish.py` path removes the source H1 before sending a
shared content body. Medium's API does not render its separate `title` field in
the article body, so any shared publisher must explicitly restore the visible
H1 for Medium. Medium may also reject Markdown with parser error `2012`; the
reliable fallback is semantic HTML generated from the prepared local artifact.

Zhihu and Juejin currently need browser-based manual review because their editors, account state, category/tag choices, and final publish dialogs are platform UI workflows rather than stable local APIs.

Other social/media platforms remain visible-browser-only. Never use hidden
endpoints or background interfaces. API response data establishes creation for
Medium and DEV.to, while normal browser interaction establishes final rendered
QA and ongoing community observations.

## Recommended Model

1. Keep the canonical draft in `docs/blog/posts/` or the relevant docs/tutorial source. Default coverage intentionally excludes legacy `docs/blogs/` pages.
2. Use `.github/publisher/posts_queue.txt` only as an optional Medium/DEV.to
   input queue; use `platforms/*.json` as the canonical cross-platform ledger,
   with `published.md` and `not-published.md` as readable snapshots.
3. Prepare a platform copy before calling the API or opening an editor:
   - Remove YAML front matter.
   - Keep one clear H1 title.
   - Convert relative image links to public `https://eunomia.dev/...` URLs or upload images through the platform editor.
   - Review code blocks, tables, Mermaid, math, footnotes, and HTML blocks after paste/import.
4. Stop before the API creation call or visible publish action unless the user
   has authorized publication or the run is executing a `排队` item through
   `eunomia-content-patrol`. Comments, likes, follows, reposts, and other social
   actions still require their own authorization.
5. After a real publish, add the platform URL to `published.md` and remove/update the item in `not-published.md`.

## What Others Do

The safest common pattern is "Markdown source, platform editor confirmation." OpenWrite describes a Chrome-extension workflow where Markdown is written once, distributed to multiple platforms, then confirmed in each platform editor before publishing: <https://openwrite.cn/>.

For Juejin, the old xitu/gold-miner guide still captures the core flow: enter the write page, fill title, paste Markdown, choose category and tags, optionally upload a cover, then publish. It also emphasizes selecting accurate categories and tags: <https://github.com/xitu/gold-miner/wiki/%E5%88%86%E4%BA%AB%E5%88%B0%E6%8E%98%E9%87%91%E6%8C%87%E5%8D%97>.

For Zhihu, Markdown import/paste needs extra QA. Community tooling such as `md2zhihu` exists because Zhihu formatting can need conversion for tables, formulas, and images: <https://blog.openacid.com/toolkit/md2zhihu/>.

## Local Skills And Ledgers

- Canonical Zhihu skill: [`../../../.agents/skills/zhihu-publisher/SKILL.md`](../../../.agents/skills/zhihu-publisher/SKILL.md)
- Canonical Juejin skill: [`../../../.agents/skills/juejin-publisher/SKILL.md`](../../../.agents/skills/juejin-publisher/SKILL.md)
- Canonical Xiaohongshu skill: [`../../../.agents/skills/xiaohongshu-publisher/SKILL.md`](../../../.agents/skills/xiaohongshu-publisher/SKILL.md)
- Daily content patrol skill: [`../../../.agents/skills/eunomia-content-patrol/SKILL.md`](../../../.agents/skills/eunomia-content-patrol/SKILL.md)
- Media Zhihu notes: [zhihu-skill.md](zhihu-skill.md)
- Media Juejin notes: [juejin-skill.md](juejin-skill.md)
- Source-set config: [sources.json](sources.json)
- Per-platform JSON ledgers: [platforms/](platforms/)
- [Community feedback ledger](community-feedback.md)
- Ledger checker: [check_media_ledger.py](check_media_ledger.py)
- [Confirmed published items](published.md)
- [Not published / pending items](not-published.md)

## Editor Screenshots

These screenshots were captured from the logged-in browser session without publishing anything. Future platform checks should default to the sidebar / in-app browser.

![Zhihu publish page](screenshots/zhihu-publish-page.png)

![Juejin publish page](screenshots/juejin-publish-page.png)
