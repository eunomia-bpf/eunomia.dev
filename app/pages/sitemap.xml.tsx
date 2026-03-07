import type { GetServerSideProps } from "next";

import { listSitemapRoutes } from "../lib/content";
import { absoluteUrl } from "../lib/seo";

export const getServerSideProps: GetServerSideProps = async ({ res }) => {
  const routes = listSitemapRoutes();

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${routes
  .map((path) => {
    const englishPath = path.startsWith("/zh/") ? path.replace(/^\/zh/, "") || "/" : path;
    const zhPath = path.startsWith("/zh/") ? path : path === "/" ? "/zh/" : `/zh${path}`;
    return `  <url>
    <loc>${absoluteUrl(path)}</loc>
    <xhtml:link rel="alternate" hreflang="en" href="${absoluteUrl(englishPath)}" />
    <xhtml:link rel="alternate" hreflang="zh" href="${absoluteUrl(zhPath)}" />
  </url>`;
  })
  .join("\n")}
</urlset>`;

  res.setHeader("Content-Type", "application/xml; charset=utf-8");
  res.write(xml);
  res.end();

  return {
    props: {}
  };
};

export default function Sitemap() {
  return null;
}
