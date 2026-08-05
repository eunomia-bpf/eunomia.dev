# Human-only blockers

No human-only blockers.

## Known non-blocking limitations

- Google Analytics 4 and Search Console Drive exports are disabled in `site.md`
  until a public-safe export folder and filename contract are configured.
- Cloudflare collection is disabled until a read-only supported connector or
  GraphQL route is available.
- Disabled sources must be reported as unavailable, never as zero traffic.

The automation is authorized to perform all normal collection, repository,
branch, pull-request, CI-wait, self-review, squash-merge, deployment-wait,
verification, and closeout steps. Add a blocker only when an external system
truly requires a human-only action or the required account permission does not
exist. Include the exact blocked action, evidence, impact, and minimal human
action needed. Remove resolved items in the next pull request.
