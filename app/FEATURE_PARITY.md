# Feature Parity Checklist

This is the minimum parity target for a MkDocs-to-Next.js migration.

| Capability | Current source | Current app status | Must keep | Suggested implementation | Verification |
| --- | --- | --- | --- | --- | --- |
| Stable public routes | MkDocs nav + generated routes | Implemented | Yes | explicit route map + redirects manifest | `test/scripts/link-crawl.mjs` |
| English and Chinese paths | `mkdocs-static-i18n` | Implemented | Yes | path-based i18n in App Router | `test/scripts/http-audit.mjs` |
| Titles and descriptions | Markdown frontmatter + MkDocs metadata | Implemented | Yes | per-page metadata generator | `test/scripts/http-audit.mjs` |
| Canonical URLs | MkDocs output | Implemented | Yes | canonical builder from route metadata | `test/scripts/http-audit.mjs` |
| `hreflang` alternates | MkDocs output | Implemented | Yes | language alternate builder | `test/scripts/http-audit.mjs` |
| Open Graph tags | Material social plugin | Implemented | Yes | shared metadata builder + route-aware OG strategy | `test/scripts/http-audit.mjs` |
| `robots.txt` | generated today | Implemented | Yes | route handler | `test/scripts/http-audit.mjs` |
| `sitemap.xml` | generated today | Implemented | Yes | route handler | `test/scripts/http-audit.mjs` |
| RSS feed autodiscovery | none today | Implemented | Growth | locale-aware feed routes and `<link rel="alternate" type="application/rss+xml">` | `test/scripts/http-audit.mjs` |
| Blog index and post pages | MkDocs blog plugin | Implemented | Yes | content collection + dated slugs | `test/scripts/browser-smoke.mjs` |
| Docs search | MkDocs search | Implemented | Yes | server-backed content index with locale-aware ranking, keyboard support, and full results page | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Heading anchors and TOC | Markdown extensions | Implemented | Yes | rehype slug + TOC extraction | `test/scripts/browser-smoke.mjs` |
| Code blocks and highlighting | Markdown extensions | Implemented | Yes | `rehype-pretty-code` + language normalization | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Callouts/admonitions | Markdown extensions | Missing | Yes | custom remark plugin + components | page render review |
| Tabs | Markdown extensions | Missing | Yes | custom remark plugin + components | page render review |
| Edit links | `edit_uri` | Implemented | Yes | page metadata to GitHub source URL | `test/scripts/browser-smoke.mjs` |
| Last updated | git revision plugin | Implemented | Yes | git metadata during build | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Authors | git authors plugin | Implemented | Yes | git metadata during build | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Article continuation | theme affordance | Implemented | Growth | collection-aware continuation cards in the page footer | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Mobile nav and search | Material responsive header | Implemented | Yes | dedicated mobile menu with embedded search | `test/scripts/browser-smoke.mjs` |
| Analytics | MkDocs config | Implemented | Yes | script component gated by env | `test/scripts/http-audit.mjs` |
| Feedback CTA | MkDocs theme config | Implemented | Yes | reusable page footer CTA | `test/scripts/browser-smoke.mjs` |
| Share buttons | `hooks/socialmedia.py` | Implemented | Yes | page footer share component | `test/scripts/browser-smoke.mjs` |
| Legacy `/blogs/*` links | old content tree | Implemented | Yes | redirects or canonical aliases | `test/scripts/link-crawl.mjs` |

## Known Risk Areas

- `docs/blog` and `docs/blogs` currently overlap
- search is currently server-backed rather than a static index, so payload size is controlled but future cutover may still prefer a static index
- production verification now uses an isolated `distDir` so `next dev` and `next start` can run side-by-side without corrupting `.next`
- advanced Markdown extensions still need parity work
- full-text search result pages exist, but they are intentionally `noindex`
- feed support is an app enhancement, not a strict MkDocs parity requirement
- tutorial content is synced from a separate source repository
- local asset resolution must remain stable for deep tutorial paths
