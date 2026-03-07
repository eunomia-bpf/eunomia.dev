# Website Audit Scripts

These scripts audit a docs/blog site from the outside, with a bias toward the current `eunomia.dev` requirements:

- SEO infrastructure
- multilingual routing
- browser-level smoke behavior
- internal link and asset reachability

## Install

```bash
cd /home/yunwei37/workspace/eunomia.dev/test
npm install
npx playwright install chromium
```

## Run Against Production

```bash
npm run audit
```

## Run Against Local Development Server

```bash
BASE_URL=http://127.0.0.1:3000 npm run audit
```

## Available Commands

- `npm run audit:http` checks robots, sitemap, metadata, canonicals, hreflang, OG tags, and analytics hooks
- `npm run audit:browser` launches Chromium and exercises basic user flows
- `npm run audit:links` crawls internal links and page assets
- `npm run audit` runs all three checks in sequence

## Environment Variables

- `BASE_URL`: target site, default `https://eunomia.dev`
- `REQUEST_TIMEOUT_MS`: per-request timeout, default `15000`
- `MAX_PAGES`: maximum crawled HTML pages for link audit, default `25`
- `MAX_ASSETS`: maximum asset requests during crawl, default `80`
