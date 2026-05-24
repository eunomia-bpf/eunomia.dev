# eunomia-bpf mainpage



This is the source of the official [eunomia-bpf website][official website].

For the tutorial, please edit https://github.com/eunomia-bpf/bpf-developer-tutorial

For the home page of each project, please edit the README of them.

[official website]: https://eunomia.dev

## Tech stack

The website is a custom **Next.js + React + Tailwind CSS** frontend (in `app/`) that
statically exports to plain HTML/CSS/JS and deploys to GitHub Pages. Content is authored
as Markdown under `docs/**` and compiled into the site at build time; `mkdocs.yaml` is kept
as the navigation/IA configuration source. See `app/README.md` and `app/ARCHITECTURE.md`.

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
