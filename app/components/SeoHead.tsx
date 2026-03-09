import Head from "next/head";

import type { GitMetadata } from "../lib/content/types";
import { AlternateLink, absoluteUrl, ogImageUrl } from "../lib/seo";
import { siteConfig } from "../lib/site-data";

type SeoHeadProps = {
  title: string;
  description: string;
  path: string;
  alternates: AlternateLink[];
  article?: boolean;
  metadata?: GitMetadata | null;
  eyebrow?: string;
  robots?: string;
};

export function SeoHead({
  title,
  description,
  path,
  alternates,
  article = false,
  metadata,
  robots
}: SeoHeadProps) {
  const canonical = absoluteUrl(path);
  const fullTitle = `${title} | ${siteConfig.name}`;
  const ogImage = ogImageUrl();
  const feedHref = path.startsWith("/zh") ? absoluteUrl("/zh/feed.xml") : absoluteUrl("/feed.xml");

  return (
    <Head>
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      {robots ? <meta name="robots" content={robots} /> : null}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={ogImage} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:type" content={article ? "article" : "website"} />
      <meta property="og:url" content={canonical} />
      <meta property="og:image" content={ogImage} />
      {metadata?.createdAt ? <meta property="article:published_time" content={metadata.createdAt} /> : null}
      {metadata?.updatedAt ? <meta property="article:modified_time" content={metadata.updatedAt} /> : null}
      {metadata?.authors.map((author) => (
        <meta key={author.name} property="article:author" content={author.name} />
      ))}
      <link rel="canonical" href={canonical} />
      <link rel="alternate" type="application/rss+xml" title={`${siteConfig.name} feed`} href={feedHref} />
      {alternates.map((alternate) => (
        <link
          key={`${alternate.hrefLang}-${alternate.href}`}
          rel="alternate"
          hrefLang={alternate.hrefLang}
          href={alternate.href}
        />
      ))}
    </Head>
  );
}
