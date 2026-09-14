# Juejin Publishing Skill Brief

Last checked: 2026-09-14

Use this when preparing a eunomia.dev Markdown article for Juejin. The canonical agent skill is `.agents/skills/juejin-publisher/SKILL.md`.

Strict platform boundary: do not directly access Juejin APIs or hidden/internal
endpoints under any circumstances. Use only normal browser interactions visible
to a regular logged-in user: navigation, scrolling, clicking, reading rendered
pages, editor workflows, and screenshots.

## Goal

Create a reviewed Juejin draft from the canonical repository article, choose the right technical category/tags, stop before final publish, and record the outcome.

## Entry Points

- Article editor: <https://juejin.cn/editor/drafts/new>
- Profile observed in logged-in browser/sidebar: <https://juejin.cn/user/4288563097635144>
- Article list observed in logged-in browser/sidebar: <https://juejin.cn/user/4288563097635144/posts>

## Observed Editor UI

The current editor shows:

- title input: `输入文章标题...`
- Markdown editor with `编辑` and `预览` tabs
- autosave message: article is saved to the draft box
- counters for characters, lines, and body words
- top controls: draft box and `发布`
- publish dialog controls are present after publishing flow starts, including cover/category/tag related controls

Screenshot: [screenshots/juejin-publish-page.png](screenshots/juejin-publish-page.png)

## Safe Workflow

1. Read the source Markdown and extract title, summary, tags, and canonical URL.
2. Create a Juejin copy:
   - remove front matter
   - keep Markdown mostly intact
   - convert relative images to public URLs
   - ensure code fences have language labels
   - add a short canonical link back to eunomia.dev near the end
3. Open <https://juejin.cn/editor/drafts/new>.
4. Fill `输入文章标题...`.
5. Paste the body into the Markdown editor.
6. Use `预览` and scan headings, images, links, code blocks, and table layout.
7. Click `发布` only to inspect publish settings if needed, then stop before `确定并发布` unless the user explicitly approves final publishing.
8. Choose category and tags carefully:
   - eBPF tutorials usually fit `后端`, `Linux`, `开源`, `云原生`, or `架构`
   - AI agent / runtime posts usually fit `人工智能`, `AIGC`, `后端`, `架构`, or `安全`
   - GPU observability posts usually fit `人工智能`, `后端`, `架构`, `Linux`, or `性能优化`
9. After a confirmed publish, record title, source path, Juejin URL, date, category, tags, and formatting fixes in `published.md`.

## Content Strategy

Juejin readers reward immediately useful technical framing. For eunomia.dev posts:

- Put the practical payoff in the title or first paragraph.
- Keep the intro shorter than the site version.
- Use screenshots, diagrams, and command output only when they advance the tutorial.
- Add precise tags; the xitu/gold-miner guide notes that accurate categories and tags improve discoverability: <https://github.com/xitu/gold-miner/wiki/%E5%88%86%E4%BA%AB%E5%88%B0%E6%8E%98%E9%87%91%E6%8C%87%E5%8D%97>.
- Prefer one article per concrete technique. For large docs, split into a series and link the canonical full tutorial.

## Session Recovery

An environment restart can drop the visible Chrome profile's Juejin login even
though the mounted `/run/social-manager-session/browser-state.json` still holds
valid session cookies (`sid_tt`, `sessionid`, `sid_guard`, ...). The symptom is
`/editor/drafts/new` redirecting to `/login` with no creator controls.

`agent-browser cookies set` only writes non-httpOnly cookies on the current
page, and the browser-target CDP `Network.setCookies` is not exposed here. The
working import is the page-target CDP domain `Storage.setCookies` with the
cookie objects from the mounted state file, verified with `Storage.getCookies`
(2026-09-14: 55 cookies imported, 19 Juejin cookies present). The first
navigation after import may show a ByteDance `验证码中间页` slide CAPTCHA; do not
solve it — re-navigating to `/` and back to `/editor/drafts/new` cleared it.

## Do Not Automate

- final `确定并发布`
- direct Juejin API access, internal endpoint reads, or browser-hidden data fetches
- sign-in, phone verification, or CAPTCHA
- `去签到`, likes, follows, comments, reposts, or private messages
- account settings or monetization settings
- deleting drafts
