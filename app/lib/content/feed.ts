import type { Locale } from "../site-data";
import { getBlogEntries } from "./collections";
import { parseMarkdown } from "./markdown";
import { absoluteUrl } from "../seo";

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

export function renderFeed(locale: Locale): string {
  const items = getBlogEntries()
    .slice(0, 30)
    .map((entry) => {
      const sourceRelative = entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
      if (!sourceRelative) {
        return null;
      }

      const metadata = parseMarkdown(sourceRelative);
      const path =
        locale === "zh"
          ? `/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`
          : `/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`;
      const url = absoluteUrl(path);

      return `<item>
  <title>${escapeXml(metadata.title)}</title>
  <link>${url}</link>
  <guid>${url}</guid>
  <pubDate>${new Date(`${entry.year}-${entry.month}-${entry.day}T00:00:00Z`).toUTCString()}</pubDate>
  <description>${escapeXml(metadata.excerpt || metadata.description)}</description>
</item>`;
    })
    .filter(Boolean)
    .join("\n");

  const title = locale === "zh" ? "eunomia 博客" : "eunomia blog";
  const description =
    locale === "zh"
      ? "eunomia.dev 上关于 eBPF、bpftime、AI tracing 和系统研究的最新文章。"
      : "Latest writing from eunomia.dev on eBPF, bpftime, AI tracing, and systems research.";
  const link = absoluteUrl(locale === "zh" ? "/zh/blog/" : "/blog/");

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>${escapeXml(title)}</title>
  <link>${link}</link>
  <description>${escapeXml(description)}</description>
  <language>${locale === "zh" ? "zh-CN" : "en-US"}</language>
  ${items}
</channel>
</rss>`;
}
