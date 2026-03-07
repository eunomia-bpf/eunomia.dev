# Feature Parity Checklist

This is the minimum parity target for a MkDocs-to-Next.js migration.

| Capability | Current source | Must keep | Suggested implementation | Verification |
| --- | --- | --- | --- | --- |
| Stable public routes | MkDocs nav + generated routes | Yes | explicit route map + redirects manifest | `test/scripts/link-crawl.mjs` |
| English and Chinese paths | `mkdocs-static-i18n` | Yes | path-based i18n in App Router | `test/scripts/http-audit.mjs` |
| Titles and descriptions | Markdown frontmatter + MkDocs metadata | Yes | per-page metadata generator | `test/scripts/http-audit.mjs` |
| Canonical URLs | MkDocs output | Yes | canonical builder from route metadata | `test/scripts/http-audit.mjs` |
| `hreflang` alternates | MkDocs output | Yes | language alternate builder | `test/scripts/http-audit.mjs` |
| Open Graph tags | Material social plugin | Yes | `generateMetadata` + OG images | `test/scripts/http-audit.mjs` |
| `robots.txt` | generated today | Yes | route handler | `test/scripts/http-audit.mjs` |
| `sitemap.xml` | generated today | Yes | route handler | `test/scripts/http-audit.mjs` |
| Blog index and post pages | MkDocs blog plugin | Yes | content collection + dated slugs | `test/scripts/browser-smoke.mjs` |
| Docs search | MkDocs search | Yes | `Pagefind` or equivalent | `test/scripts/browser-smoke.mjs` |
| Heading anchors and TOC | Markdown extensions | Yes | rehype slug + TOC extraction | page render review |
| Code blocks and highlighting | Markdown extensions | Yes | rehype/Prism or Shiki | page render review |
| Callouts/admonitions | Markdown extensions | Yes | custom remark plugin + components | page render review |
| Tabs | Markdown extensions | Yes | custom remark plugin + components | page render review |
| Edit links | `edit_uri` | Yes | page metadata to GitHub source URL | `test/scripts/browser-smoke.mjs` |
| Last updated | git revision plugin | Yes | git metadata during build | custom metadata check |
| Authors | git authors plugin | Yes | git metadata during build | custom metadata check |
| Analytics | MkDocs config | Yes | script component gated by env | `test/scripts/http-audit.mjs` |
| Feedback CTA | MkDocs theme config | Yes | reusable feedback component | browser smoke |
| Share buttons | `hooks/socialmedia.py` | Yes | post footer share component | browser smoke |
| Legacy `/blogs/*` links | old content tree | Yes | redirects or canonical aliases | `test/scripts/link-crawl.mjs` |

## Known Risk Areas

- `docs/blog` and `docs/blogs` currently overlap
- Chinese pages need stricter language-tag verification
- tutorial content is synced from a separate source repository
- local asset resolution must remain stable for deep tutorial paths
