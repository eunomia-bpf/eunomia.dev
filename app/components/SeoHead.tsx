import Head from "next/head";

import { AlternateLink, absoluteUrl } from "../lib/seo";
import { siteConfig } from "../lib/site-data";

type SeoHeadProps = {
  title: string;
  description: string;
  path: string;
  alternates: AlternateLink[];
};

export function SeoHead({ title, description, path, alternates }: SeoHeadProps) {
  const canonical = absoluteUrl(path);
  const fullTitle = `${title} | ${siteConfig.name}`;

  return (
    <Head>
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:type" content="website" />
      <meta property="og:url" content={canonical} />
      <meta property="og:image" content={siteConfig.ogImage} />
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
