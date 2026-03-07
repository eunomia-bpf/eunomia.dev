# Feature Parity Checklist

This is the minimum parity target for a MkDocs-to-Next.js migration.

| Capability | Current source | Current app status | Must keep | Suggested implementation | Verification |
| --- | --- | --- | --- | --- | --- |
| Stable public routes | MkDocs nav + generated routes | Implemented | Yes | explicit route map + redirects manifest | `test/scripts/link-crawl.mjs` |
| English and Chinese paths | `mkdocs-static-i18n` | Implemented | Yes | path-based i18n in App Router | `test/scripts/http-audit.mjs` |
| Titles and descriptions | Markdown frontmatter + MkDocs metadata | Implemented | Yes | per-page metadata generator | `test/scripts/http-audit.mjs` |
| Canonical URLs | MkDocs output | Implemented | Yes | canonical builder from route metadata | `test/scripts/http-audit.mjs` |
| `hreflang` alternates | MkDocs output | Implemented | Yes | language alternate builder | `test/scripts/http-audit.mjs` |
| Open Graph tags | Material social plugin | Partial | Yes | `generateMetadata` + OG images | `test/scripts/http-audit.mjs` |
| `robots.txt` | generated today | Implemented | Yes | route handler | `test/scripts/http-audit.mjs` |
| `sitemap.xml` | generated today | Implemented | Yes | route handler | `test/scripts/http-audit.mjs` |
| Blog index and post pages | MkDocs blog plugin | Implemented | Yes | content collection + dated slugs | `test/scripts/browser-smoke.mjs` |
| Docs search | MkDocs search | Implemented | Yes | server-backed content index with locale-aware ranking | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Heading anchors and TOC | Markdown extensions | Implemented | Yes | rehype slug + TOC extraction | `test/scripts/browser-smoke.mjs` |
| Code blocks and highlighting | Markdown extensions | Implemented | Yes | `rehype-pretty-code` + language normalization | `app/tests/content.test.ts`, `test/scripts/browser-smoke.mjs` |
| Callouts/admonitions | Markdown extensions | Missing | Yes | custom remark plugin + components | page render review |
| Tabs | Markdown extensions | Missing | Yes | custom remark plugin + components | page render review |
| Edit links | `edit_uri` | Implemented | Yes | page metadata to GitHub source URL | `test/scripts/browser-smoke.mjs` |
| Last updated | git revision plugin | Missing | Yes | git metadata during build | custom metadata check |
| Authors | git authors plugin | Missing | Yes | git metadata during build | custom metadata check |
| Analytics | MkDocs config | Implemented | Yes | script component gated by env | `test/scripts/http-audit.mjs` |
| Feedback CTA | MkDocs theme config | Missing | Yes | reusable feedback component | browser smoke |
| Share buttons | `hooks/socialmedia.py` | Missing | Yes | post footer share component | browser smoke |
| Legacy `/blogs/*` links | old content tree | Implemented | Yes | redirects or canonical aliases | `test/scripts/link-crawl.mjs` |

## Known Risk Areas

- `docs/blog` and `docs/blogs` currently overlap
- search is currently server-backed rather than a static index, so payload size is controlled but future cutover may still prefer a static index
- advanced Markdown extensions still need parity work
- tutorial content is synced from a separate source repository
- local asset resolution must remain stable for deep tutorial paths
