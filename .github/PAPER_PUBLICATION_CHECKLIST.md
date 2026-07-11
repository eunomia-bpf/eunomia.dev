# Paper publication checklist

Use this checklist when a preprint is submitted or revised, a venue decision arrives, or a paper artifact moves to a new repository.

1. Add or update the paper in both `docs/others/papers/README.md` and `README.zh.md`.
2. Use the canonical arXiv, DOI, or official proceedings URL and the organization-owned artifact repository.
3. Add the English and Chinese technical blogs, or mark the blog explicitly as pending in the papers index.
4. Backfill the paper link, formal title, current version or venue, and evaluation results in existing project pages and blogs.
5. Remove superseded personal-repository links and stale paper IDs.
6. Run `cd app && npm run audit:papers -- --allow-warnings`, followed by the normal site validation.

The `Weekly Paper Publication Audit` workflow runs at 16:17 UTC every Monday on a GitHub-hosted Ubuntu runner. It compares the bilingual index with recent arXiv records by the tracked author and paper links exposed by recently active eunomia-bpf repositories, verifies local and external links, scans for known stale references, and keeps one GitHub issue updated with new candidates and outstanding blog work.

The workflow reports and tracks drift. It does not write or publish paper claims automatically; a maintainer reviews content changes before they enter the site.
