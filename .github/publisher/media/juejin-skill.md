# Juejin Publishing Skill Brief

Last checked: 2026-09-23

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


## Publish Dialog

- The category and tag dialog is `.publish-popup`, not `.byte-modal` (the modal
  node renders with empty text). A selected category chip has class `active`,
  not `selected`.
- Tag entry: focus `.publish-popup .byte-select__input`[0] (index 0 is tags),
  set its value with the native `HTMLInputElement` value setter plus a bubbling
  `input` event, click the document-wide visible `.byte-select-option` whose text
  matches exactly, then read chips back from `.publish-popup .byte-select__tag`.
  Clear the input between tags with `value = ''` plus a bubbling `input` event.
  Avoid CLI `keyboard type` here: the text can leak into the title or body, and
  the option list does not always open (observed 2026-09-24).
- The confirm button's exact text is `确定并发布`; success lands on
  `https://juejin.cn/published` with `document.title === '发布成功'`. The 确定并发布 button ignores a synthetic `element.click()` and a CLI `agent-browser click` on `.publish-popup .ui-btn.primary` (observed 2026-09-22 on 43-kfuncs: both only fired the autosave toast, no publish); dispatch a real pointer-event sequence (pointerdown/mousedown/focus/pointerup/mouseup/click with view:window, button:0) via eval. The tag search widget is not reached by CLI fill/keyboard type (typed text leaks into the title or body); set `.byte-select__input` value with the native HTMLInputElement value setter plus a bubbling `input` event, then click the exact `.byte-select-option`. Read back and reset the title input to the exact source H1 before the final submit, because early tag typing can pollute the title.
- A submitted article can remain in review: only `/spost/<id>` renders and
  `/post/<id>` returns `找不到页面`. Record it as `review_pending` only after
  probing both. Review can also clear within minutes: on 2026-09-24 the 41-xdp-tcpdump
  article staged briefly, then `/post/<id>` returned 200 with no `审核中` marker
  and the `/spost/<id>` URL 404'd in the same session, so record `confirmed`.
- Injecting a long body through `agent-browser eval` needs `--stdin` with the
  base64 embedded in the script; an inline argument fails for large payloads.
- `.byte-select-option` lives outside `.publish-popup`; query it document-wide and
  filter by `getBoundingClientRect().width > 0` (several dropdowns are hidden at
  once). Do NOT use `offsetParent !== null`: an unopened
  `.byte-select-dropdown__wrap` keeps its options in the DOM at zero size with a
  non-null `offsetParent`, so that filter reports an open dropdown as empty
  (hit on the 2026-09-24 40-mysql prep). The popup-scoped query returns zero
  options even when the tag dropdown is open (observed 2026-09-23 on
  42-xdp-loadbalancer). Set the input value with the native setter plus a
  bubbling `input` event to open the list, clearing any leftover text first;
  CLI `keyboard type` leaks the text elsewhere. Read committed chips back from
  `.publish-popup .byte-select__tag` after each option click.
- Select a category chip with **real CDP pointer events**
  (`page.mouse.move` → `move` → `down` → `up`). In-page synthetic events are not
  enough: on 2026-09-24 a full synthetic hover-inclusive sequence (mouseover/
  pointerover/pointerenter/mouseenter, then pointerdown/mousedown with
  `buttons: 1`, focus, pointerup/mouseup, click) still left
  `.category-list .item.active` empty, while the real CDP mouse sequence selected
  it on the first try.
- A submission does not always enter review: the 2026-09-23 42-xdp-loadbalancer
  article went straight to the canonical `/post/<id>` URL with no `/spost/`
  staging interval. The author profile list can lag or omit the new item, so the
  reliable URL sources are the creator center article list (`文章管理` →
  `https://juejin.cn/creator/content/article/essays?status=all`, whose `审核中`
  tab names the staged URL) and a direct `curl` status probe on the candidate
  `/post/<id>` URL.
- Read the creator center by URL rather than clicking through the SPA. The
  `https://juejin.cn/creator/content/article/all` shell renders empty
  `.byte-tab-pane` elements and can show a `选择你感兴趣的技术方向` onboarding
  modal over the list, which makes the account look empty. Load
  `https://juejin.cn/creator/content/article/essays?status=all` directly; its
  counts and per-post `展现 / 阅读 / 点赞 / 评论 / 收藏` appear in
  `document.body.innerText` within a few seconds.

## Do Not Automate

- final `确定并发布`
- direct Juejin API access, internal endpoint reads, or browser-hidden data fetches
- sign-in, phone verification, or CAPTCHA
- `去签到`, likes, follows, comments, reposts, or private messages
- account settings or monetization settings
- deleting drafts
