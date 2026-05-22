# eunomia-bpf mainpage



This is the source of the official [eunomia-bpf website][official website].

For the tutorial, please edit https://github.com/eunomia-bpf/bpf-developer-tutorial

For the home page of each project, please edit the README of them.

[official website]: https://eunomia.dev

## Requirements

- Node.js 22+
- npm

## Local development

Clone this repo and enter, then:

```bash
cd app
npm ci
npm run dev
```

The website will now be accessible at http://localhost:3000.

For a production-compatible static export:

```bash
cd app
NEXT_PUBLIC_SITE_URL=https://eunomia.dev npm run build
```

The exported site is written to `app/out`. GitHub Actions verifies the static app, rebuilds it with production URLs, uploads `app/out` as a GitHub Pages artifact, and deploys it to https://eunomia.dev.
