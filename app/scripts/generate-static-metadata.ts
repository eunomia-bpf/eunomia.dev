import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { renderFeed } from "../lib/content/feed";
import { getContentManifest } from "../lib/content/manifest";
import { getActiveRolloutStage, stageAllowsRoute } from "../lib/rollout";
import { absoluteUrl, STATIC_OG_IMAGE_PATH } from "../lib/seo";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const defaultPublicDir = path.join(appDir, "public");
const defaultGeneratedIndexPath = path.join(appDir, ".generated", "content", "static-metadata.json");

type StaticMetadataPaths = {
  publicDir?: string;
  indexPath?: string;
};

type StaticMetadataFile = {
  relativePath: string;
  contents: string;
};

function writeFileAtomic(filePath: string, contents: string) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.tmp`;
  fs.writeFileSync(tempPath, contents, "utf8");
  fs.renameSync(tempPath, filePath);
}

function escapeXml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderStaticOgSvg() {
  const eyebrow = "eunomia.dev";
  const lines = ["Open-source eBPF tools,", "tutorials, and", "systems research"];

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="630" viewBox="0 0 1200 630" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="98" y1="72" x2="1086" y2="574" gradientUnits="userSpaceOnUse">
      <stop stop-color="#081A33"/>
      <stop offset="0.52" stop-color="#0F305D"/>
      <stop offset="1" stop-color="#18507F"/>
    </linearGradient>
    <radialGradient id="glowA" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(980 120) rotate(120) scale(380 420)">
      <stop stop-color="#F59E0B" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#F59E0B" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glowB" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(220 520) rotate(12) scale(360 260)">
      <stop stop-color="#38BDF8" stop-opacity="0.34"/>
      <stop offset="1" stop-color="#38BDF8" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1200" height="630" rx="40" fill="url(#bg)"/>
  <rect x="32" y="32" width="1136" height="566" rx="30" stroke="rgba(255,255,255,0.16)" fill="rgba(255,255,255,0.03)"/>
  <ellipse cx="980" cy="120" rx="320" ry="260" fill="url(#glowA)"/>
  <ellipse cx="220" cy="520" rx="280" ry="180" fill="url(#glowB)"/>
  <rect x="88" y="86" width="180" height="44" rx="22" fill="rgba(255,255,255,0.12)"/>
  <text x="118" y="114" fill="#E0F2FE" font-size="22" font-family="ui-sans-serif, system-ui, sans-serif" letter-spacing="3">${escapeXml(
    eyebrow.toUpperCase()
  )}</text>
  <text x="88" y="196" fill="#FFFFFF" font-size="64" font-weight="700" font-family="ui-serif, Georgia, serif">
    ${lines
      .map(
        (line, index) =>
          `<tspan x="88" y="${196 + index * 76}">${escapeXml(line)}</tspan>`
      )
      .join("")}
  </text>
  <text x="88" y="520" fill="#C7D2FE" font-size="28" font-family="ui-sans-serif, system-ui, sans-serif">eunomia.dev</text>
  <text x="88" y="560" fill="#E2E8F0" font-size="24" font-family="ui-sans-serif, system-ui, sans-serif">Static OG image shared by all pages for static export compatibility</text>
</svg>`;
}

export function buildSitemapXml() {
  const activeStage = getActiveRolloutStage();
  const entries: string[] = [];
  const seen = new Set<string>();

  for (const record of getContentManifest()) {
    if (!stageAllowsRoute(record.sitemapStage, activeStage)) {
      continue;
    }

    for (const routePath of Object.values(record.routeByLocale).filter(Boolean) as string[]) {
      const url = absoluteUrl(routePath);
      if (seen.has(url)) {
        continue;
      }

      seen.add(url);
      const alternates = [
        record.routeByLocale.en
          ? `    <xhtml:link rel="alternate" hreflang="en" href="${absoluteUrl(record.routeByLocale.en)}" />`
          : null,
        record.routeByLocale.zh
          ? `    <xhtml:link rel="alternate" hreflang="zh" href="${absoluteUrl(record.routeByLocale.zh)}" />`
          : null
      ]
        .filter(Boolean)
        .join("\n");

      entries.push(`  <url>
    <loc>${url}</loc>
${alternates}
  </url>`);
    }
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${entries.join("\n")}
</urlset>`;
}

export function buildRobotsTxt() {
  return `User-agent: *
Allow: /

Sitemap: ${absoluteUrl("/sitemap.xml")}
`;
}

export function writeStaticMetadata(paths: StaticMetadataPaths = {}) {
  const publicDir = paths.publicDir ?? defaultPublicDir;
  const indexPath = paths.indexPath ?? defaultGeneratedIndexPath;
  const files: StaticMetadataFile[] = [
    {
      relativePath: "feed.xml",
      contents: renderFeed("en")
    },
    {
      relativePath: path.join("zh", "feed.xml"),
      contents: renderFeed("zh")
    },
    {
      relativePath: "sitemap.xml",
      contents: buildSitemapXml()
    },
    {
      relativePath: "robots.txt",
      contents: buildRobotsTxt()
    },
    {
      relativePath: path.join("og", "default.svg"),
      contents: renderStaticOgSvg()
    }
  ];

  for (const file of files) {
    writeFileAtomic(path.join(publicDir, file.relativePath), file.contents);
  }

  const payload = {
    generatedAt: new Date().toISOString(),
    files: files.map((file) => ({
      relativePath: file.relativePath
    }))
  };
  writeFileAtomic(indexPath, `${JSON.stringify(payload)}\n`);

  return {
    publicDir,
    indexPath,
    files: files.map((file) => file.relativePath)
  };
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : null;
if (invokedPath === fileURLToPath(import.meta.url)) {
  const result = writeStaticMetadata();
  console.log(
    `Wrote ${result.files.length} static metadata files to ${result.publicDir} and index ${result.indexPath}`
  );
}
