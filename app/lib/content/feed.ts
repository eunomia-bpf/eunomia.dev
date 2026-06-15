import type { Locale } from "../site-data";
import { siteConfig } from "../site-data";
import { getBlogEntries } from "./collections";
import { getDocument } from "./documents";
import { getGitMetadata } from "./git";
import { buildBlogIndexPath, buildBlogPath } from "./route-paths";
import { absoluteUrl } from "../seo";
import { escapeXml } from "../utils";

export function renderFeed(locale: Locale): string {
  const entries = getBlogEntries()
    .filter((entry) => entry.date)
    .slice(0, 30);

  let latestDate: Date | null = null;

  const items = entries
    .map((entry) => {
      const sourceRelative = entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
      if (!sourceRelative) {
        return null;
      }

      const metadata = getDocument(sourceRelative);
      const git = getGitMetadata(sourceRelative);
      const path = buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale);
      const url = absoluteUrl(path);
      const publishedAt = new Date(`${entry.date}T00:00:00Z`);
      if (Number.isNaN(publishedAt.valueOf())) {
        return null;
      }

      if (!latestDate || publishedAt > latestDate) {
        latestDate = publishedAt;
      }

      const authorXml = git?.authors?.length
        ? git.authors.map((a) => `  <author>${escapeXml(a.email ?? "noreply@eunomia.dev")} (${escapeXml(a.name)})</author>`).join("\n")
        : `  <author>noreply@eunomia.dev (${escapeXml(siteConfig.name)})</author>`;

      const categoryXml = entry.slug.includes("bpftime")
        ? `  <category>bpftime</category>`
        : entry.slug.includes("agent") || entry.slug.includes("llm")
          ? `  <category>AI agent observability</category>`
          : `  <category>eBPF</category>`;

      return `<item>
  <title>${escapeXml(metadata.title)}</title>
  <link>${url}</link>
  <guid>${url}</guid>
  <pubDate>${publishedAt.toUTCString()}</pubDate>
  <description>${escapeXml(metadata.excerpt || metadata.description)}</description>
${authorXml}
${categoryXml}
</item>`;
    })
    .filter(Boolean)
    .join("\n");

  const title = locale === "zh" ? "eunomia 博客" : "eunomia blog";
  const description =
    locale === "zh"
      ? "eunomia.dev 上关于 eBPF、bpftime、AI tracing 和系统研究的最新文章。"
      : "Latest writing from eunomia.dev on eBPF, bpftime, AI tracing, and systems research.";
  const link = absoluteUrl(buildBlogIndexPath(locale));
  const feedUrl = locale === "zh" ? absoluteUrl("/zh/feed.xml") : absoluteUrl("/feed.xml");
  const lastBuildDate = latestDate ? (latestDate as Date).toUTCString() : new Date().toUTCString();

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
  <title>${escapeXml(title)}</title>
  <link>${link}</link>
  <description>${escapeXml(description)}</description>
  <language>${locale === "zh" ? "zh-CN" : "en-US"}</language>
  <lastBuildDate>${lastBuildDate}</lastBuildDate>
  <atom:link href="${feedUrl}" rel="self" type="application/rss+xml" />
  ${items}
</channel>
</rss>`;
}
