import type { GetServerSideProps } from "next";

import { getContentManifest } from "../lib/content/manifest";
import { getActiveRolloutStage, stageAllowsRoute } from "../lib/rollout";
import { absoluteUrl } from "../lib/seo";

export const getServerSideProps: GetServerSideProps = async ({ res }) => {
  const activeStage = getActiveRolloutStage();
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${getContentManifest()
  .filter((record) => stageAllowsRoute(record.sitemapStage, activeStage))
  .flatMap((record) =>
    (Object.values(record.routeByLocale).filter(Boolean) as string[]).map((path) => {
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

      return `  <url>
    <loc>${absoluteUrl(path)}</loc>
${alternates}
  </url>`;
    })
  )
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
