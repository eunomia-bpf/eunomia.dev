# Paper Publication Backfill Checklist

Use this checklist whenever a eunomia-bpf paper, technical report, or public artifact is published or receives a new canonical URL. Complete the update in one pull request so the paper record, project links, and developer-facing explanation cannot drift independently.

- Verify the canonical title, author list, venue or publication status, arXiv/DOI URL, and publication date against the paper source.
- Replace personal or stale artifact links with the canonical `eunomia-bpf` organization repository when one exists, while preserving historical URLs that still need redirects.
- Update both `docs/papers/README.md` and `docs/papers/README.zh.md` with matching paper, artifact, blog, and status entries, and add the paper's entry to `docs/papers/registry.yaml` (arXiv ID pinned to the current version, key numbers with measurement conditions).
- Add the paper's full text in both forms under `docs/papers/`: the PDF (`<slug>.pdf`) and a plain-text extraction (`<slug>.txt`). Only published or public-arXiv versions may enter this directory; papers under review must never appear here.
- Update the relevant project documentation and existing English/Chinese blog posts. Use the `tech-blog-writer` skill for new posts or substantial blog revisions, and keep both languages aligned in structure and claims.
- Search the repository for the old arXiv ID, title, and repository URL to catch stale references. Open each new external link and confirm it resolves to the intended paper or artifact.
- Run `cd app && npm run test:content`, `npm run typecheck`, and `npm run verify` before publishing the change.
- Submit the update through the mature OSS pull-request workflow required by `AGENTS.md`; complete independent review, Copilot comment checks, and CI monitoring before handoff.
