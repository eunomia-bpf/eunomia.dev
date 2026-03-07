import Head from "next/head";

import type { GitMetadata } from "../lib/content/types";
import { AlternateLink, absoluteUrl } from "../lib/seo";
import { siteConfig } from "../lib/site-data";

type SeoHeadProps = {
  title: string;
  description: string;
  path: string;
  alternates: AlternateLink[];
  article?: boolean;
  metadata?: GitMetadata | null;
};

export function SeoHead({ title, description, path, alternates, article = false, metadata }: SeoHeadProps) {
  const canonical = absoluteUrl(path);
  const fullTitle = `${title} | ${siteConfig.name}`;

  return (
    <Head>
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={siteConfig.ogImage} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:type" content={article ? "article" : "website"} />
      <meta property="og:url" content={canonical} />
      <meta property="og:image" content={siteConfig.ogImage} />
      {metadata?.createdAt ? <meta property="article:published_time" content={metadata.createdAt} /> : null}
      {metadata?.updatedAt ? <meta property="article:modified_time" content={metadata.updatedAt} /> : null}
      {metadata?.authors.map((author) => (
        <meta key={author.name} property="article:author" content={author.name} />
      ))}
      <link rel="canonical" href={canonical} />
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
