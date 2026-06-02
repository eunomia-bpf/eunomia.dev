import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Emit static redirect stubs for legacy URLs that no longer match a current
 * route, so they 301-equivalent (meta-refresh + rel=canonical) into the live
 * canonical URL instead of returning 404. GitHub Pages has no server-side
 * redirect support, so the redirect must be a real HTML file in the export.
 *
 * Legacy classes:
 *
 *  1. `/en/<path>`  — the old i18n layout served English under `/en/`; English
 *     now lives at the site root. Mirror every emitted root page into `/en/`.
 *     (Port of the legacy MkDocs hook `hooks/en_redirects.py`, which only runs
 *     in the deprecated MkDocs build.)
 *  2. `/blog/posts/<stem>/` (+ `/zh/`, `/en/` variants) — the old blog used the
 *     source filename stem as the slug; posts now live at dated URLs
 *     `/blog/YYYY/MM/DD/<slug>/`. Map stem -> dated route via the manifest.
 *
 *  3. `/GPTtrace/agentsight/` (+ `/zh/`, `/en/` variants) — the old AgentSight
 *     docs page now lives under `/agentsight/`. This stub intentionally
 *     overwrites that exported legacy page, while the rest of `/GPTtrace/*`
 *     remains active.
 *  4. `/projects/agentsight/*` (+ `/zh/`, `/en/` variants) — an intermediate
 *     local docs path now redirects to the root-level AgentSight section.
 *
 * Most redirect generation only ADDS URLs; the GPTtrace AgentSight class
 * overwrites that exported legacy page by design.
 */

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const outDir = path.resolve(appDir, "out");
const manifestPath = path.join(appDir, ".generated", "content", "manifest.json");

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? "https://eunomia.dev").replace(/\/+$/, "");

function redirectHtml(target) {
  const absolute = `${siteUrl}${target}`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=${target}">
  <link rel="canonical" href="${absolute}">
  <title>Redirect</title>
</head>
<body>
  <p>Redirecting to <a href="${target}">${target}</a>...</p>
</body>
</html>
`;
}

/** Write a redirect stub at out/<routePath>index.html unless it already exists. */
function writeRedirect(routePath, target, options = {}) {
  const relative = routePath.replace(/^\/+/, "");
  const dest = path.join(outDir, relative, "index.html");
  if (fs.existsSync(dest) && !options.overwrite) {
    return false;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, redirectHtml(target));
  return true;
}

/** Collect every emitted page (index.html) outside the locale subtrees. */
function collectRootPages() {
  const pages = [];
  const walk = (absDir) => {
    for (const entry of fs.readdirSync(absDir, { withFileTypes: true })) {
      const abs = path.join(absDir, entry.name);
      const rel = path.relative(outDir, abs);
      const topSegment = rel.split(path.sep)[0];
      if (topSegment === "en" || topSegment === "zh") {
        continue;
      }
      if (entry.isDirectory()) {
        walk(abs);
      } else if (entry.isFile() && entry.name.endsWith(".html")) {
        pages.push(rel.split(path.sep).join("/"));
      }
    }
  };
  walk(outDir);
  return pages;
}

function generateEnMirror() {
  let count = 0;
  for (const relPath of collectRootPages()) {
    // Only mirror directory-style pages (`foo/index.html`) and the home page;
    // skip standalone files such as 404.html.
    if (relPath !== "index.html" && !relPath.endsWith("/index.html")) {
      continue;
    }
    // Root-relative URL this page serves: index.html -> directory URL.
    const target = `/${relPath}`.replace(/(^|\/)index\.html$/, "$1");
    if (writeRedirect(`/en/${target.replace(/^\//, "")}`, target)) {
      count += 1;
    }
  }
  return count;
}

function generateLegacyBlogRedirects() {
  if (!fs.existsSync(manifestPath)) {
    console.warn(`Manifest not found at ${manifestPath}; skipping legacy blog redirects`);
    return 0;
  }

  const { manifest } = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  let count = 0;

  for (const record of manifest) {
    const source = record.sourceByLocale?.en ?? record.sourceByLocale?.zh;
    if (!source || !/^blog\/posts\//.test(source)) {
      continue;
    }
    const stem = source
      .replace(/^blog\/posts\//, "")
      .replace(/\.(zh|en)\.md$/, "")
      .replace(/\.md$/, "");

    const enTarget = record.routeByLocale?.en;
    const zhTarget = record.routeByLocale?.zh;

    if (enTarget) {
      if (writeRedirect(`/blog/posts/${stem}/`, enTarget)) count += 1;
      if (writeRedirect(`/en/blog/posts/${stem}/`, enTarget)) count += 1;
    }
    if (zhTarget) {
      if (writeRedirect(`/zh/blog/posts/${stem}/`, zhTarget)) count += 1;
    }
  }
  return count;
}

function generateLegacyGpttraceRedirects() {
  const legacyPaths = ["/GPTtrace/agentsight/"];
  let count = 0;

  for (const legacyPath of legacyPaths) {
    if (writeRedirect(legacyPath, "/agentsight/", { overwrite: true })) count += 1;
    if (writeRedirect(`/en${legacyPath}`, "/agentsight/", { overwrite: true })) count += 1;
    if (writeRedirect(`/zh${legacyPath}`, "/zh/agentsight/", { overwrite: true })) count += 1;
  }

  return count;
}

function generateMovedAgentsightRedirects() {
  const movedPaths = [
    ["projects/agentsight", "agentsight"],
    ["projects/agentsight/quickstart", "agentsight/quickstart"],
    ["projects/agentsight/architecture", "agentsight/architecture"],
    ["projects/agentsight/visualization", "agentsight/visualization"],
    ["projects/agentsight/operational-notes", "agentsight/operational-notes"],
    ["projects/agentsight/troubleshooting", "agentsight/troubleshooting"]
  ];
  let count = 0;

  for (const [from, to] of movedPaths) {
    if (writeRedirect(`/${from}/`, `/${to}/`)) count += 1;
    if (writeRedirect(`/en/${from}/`, `/${to}/`)) count += 1;
    if (writeRedirect(`/zh/${from}/`, `/zh/${to}/`)) count += 1;
  }

  return count;
}

if (!fs.existsSync(outDir)) {
  console.error(`No static export found at ${outDir}`);
  process.exit(1);
}

// Order matters: legacy blog stubs write the dated targets directly for all
// locales first. GPTtrace stubs then overwrite old exported pages and install
// explicit /en/ and /zh/ redirects. Moved AgentSight stubs add compatibility
// for the intermediate /projects/agentsight/ path. Finally, the /en/ mirror
// fills in every other page without clobbering existing stubs, avoiding chains.
const blogCount = generateLegacyBlogRedirects();
const gpttraceCount = generateLegacyGpttraceRedirects();
const movedAgentsightCount = generateMovedAgentsightRedirects();
const enCount = generateEnMirror();

console.log(
  `Generated ${blogCount} legacy blog redirects, ${gpttraceCount} GPTtrace redirects, ${movedAgentsightCount} moved AgentSight redirects, and ${enCount} /en/ redirects in ${outDir}`
);
