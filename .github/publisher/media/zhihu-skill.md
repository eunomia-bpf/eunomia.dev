# Zhihu Publishing Skill Brief

Last checked: 2026-07-18

Use this when preparing a eunomia.dev Markdown article for Zhihu. The canonical agent skill is `.claude/skills/zhihu-publisher/SKILL.md`, exposed locally through `.agents/skills/zhihu-publisher`.

Strict platform boundary: do not directly access Zhihu APIs or hidden/internal
endpoints under any circumstances. Use only normal browser interactions visible
to a regular logged-in user: navigation, scrolling, clicking, reading rendered
pages, editor workflows, and screenshots.

## Goal

Create a reviewed Zhihu draft from the canonical repository article, stop at the editor or publish settings, and wait for explicit approval before final publishing.

## Entry Points

- Article editor: <https://zhuanlan.zhihu.com/write>
- Creator center: <https://www.zhihu.com/creator>
- Profile observed in logged-in browser/sidebar: <https://www.zhihu.com/people/yun-wei-64-11>

## Observed Editor UI

The current editor shows:

- title textarea: `请输入标题（最多 100 个字）`
- body area: `请输入正文`
- toolbar items such as title, bold, italic, list, table of contents, quote, divider, code block, note, image, video, import, draft backup, more
- side/bottom publishing controls: cover, column inclusion, preview, publish
- status text: Markdown input mode and autosaved draft

Screenshot: [screenshots/zhihu-publish-page.png](screenshots/zhihu-publish-page.png)

## Safe Workflow

1. Read the source Markdown and extract the final title, summary, tags, and canonical eunomia.dev URL.
2. Create a platform copy:
   - remove front matter
   - keep title concise, usually Chinese if posting to Zhihu
   - keep the first paragraph as a strong hook
   - replace relative images with public URLs or prepare manual image upload
   - simplify unsupported Markdown extensions
3. Open <https://zhuanlan.zhihu.com/write>.
4. Fill title and body, or use the `导入` control if importing a converted document.
5. Verify formatting visually:
   - headings keep hierarchy
   - code blocks remain readable
   - tables do not collapse
   - images render
   - links point to canonical sources
6. Set cover and column inclusion if appropriate.
7. Stop at `预览` or the visible `发布` button. Do not press final publish without explicit user approval.
8. After a confirmed publish, record title, source path, Zhihu URL, date, tags, and any formatting fixes in `published.md`.

## Content Strategy

Zhihu works best when the post reads as an explanatory essay rather than a release note. For eunomia.dev technical posts:

- Start with the problem and a concrete scenario.
- Keep architecture claims grounded in examples.
- Add a short "适合谁读" paragraph near the top if the topic is niche.
- Keep links to GitHub, paper, and canonical eunomia.dev page, but avoid making the whole post feel like an outbound-link index.
- For long tutorials, publish the conceptual article on Zhihu and link to the complete code/tutorial.

## Formatting Notes

Zhihu can accept Markdown-style input, but imported Markdown often needs manual QA. Known risky areas are tables, formulas, local images, footnotes, Mermaid diagrams, and complex HTML. Community tools such as `md2zhihu` exist specifically to convert formulas, tables, and images for Zhihu-compatible import: <https://blog.openacid.com/toolkit/md2zhihu/>.

## Do Not Automate

- final `发布`
- direct Zhihu API access, internal endpoint reads, or browser-hidden data fetches
-投稿到专栏 unless the user explicitly asks for that exact column
- deleting drafts or changing account settings
- accepting security, privacy, or phone-verification prompts
- solving CAPTCHA without user confirmation
