# SEO operations blockers

Updated: 2026-09-28

## External blockers

### Cloudflare analytics is disabled

Cloudflare analytics is disabled in `site.md`. No Cloudflare trend is claimed until that
source is explicitly enabled with a safe read-only path. This does not block the daily
operation because the configured weekly exports, public/live-site collection, GitHub, and
public web evidence remain available.

## Data limitations that are not blockers

- The newest weekly family is still labelled `2026-09-14..09-20`, while its GSC date
  export stops at 2026-09-19.
- The newest GA4 weekly aggregate remains partial under the configured lag and lacks a date
  dimension.
- Complete current seven-day and 28-day comparable GSC claims remain unavailable until
  source-native contiguous history supports them.
- External crawler/search discovery can lag the exact Pages deployment and is supplementary
  rather than primary production evidence.

Missing coverage is unavailable, not zero, and these limitations do not justify skipping the
required bilingual Daily Report.
