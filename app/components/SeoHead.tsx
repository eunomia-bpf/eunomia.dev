import Head from "next/head";

import type { GitMetadata } from "../lib/content/types";
import { toArticleDateTime } from "../lib/dates";
import { AlternateLink, absoluteUrl, ogImageUrl } from "../lib/seo";
import { siteConfig } from "../lib/site-data";

type SeoHeadProps = {
  title: string;
  description: string;
  path: string;
  alternates: AlternateLink[];
  article?: boolean;
  publishedAt?: string;
  metadata?: GitMetadata | null;
  robots?: string;
  isTutorial?: boolean;
  isCodeProject?: boolean;
  repoUrl?: string;
};

const OG_IMAGE_WIDTH = 1200;
const OG_IMAGE_HEIGHT = 630;

function buildOrganizationJsonLd(): object {
  return {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: siteConfig.name,
    url: siteConfig.siteUrl,
    logo: absoluteUrl("/og/default.svg"),
    description: siteConfig.description,
    sameAs: [
      "https://github.com/eunomia-bpf",
      siteConfig.repoUrl,
      "https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng",
      "https://dl.acm.org/doi/10.1145/3766882.3767169"
    ]
  };
}

function buildArticleJsonLd(
  title: string,
  description: string,
  canonical: string,
  ogImage: string,
  publishedAt?: string,
  modifiedAt?: string,
  authors?: { name: string }[]
): object {
  return {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: title,
    description,
    url: canonical,
    image: ogImage,
    ...(publishedAt ? { datePublished: publishedAt } : {}),
    ...(modifiedAt ? { dateModified: modifiedAt } : {}),
    ...(authors?.length
      ? { author: authors.map((a) => ({ "@type": "Person", name: a.name })) }
      : {}),
    publisher: {
      "@type": "Organization",
      name: siteConfig.name,
      url: siteConfig.siteUrl,
      logo: { "@type": "ImageObject", url: absoluteUrl("/og/default.svg") }
    }
  };
}

function buildWebSiteJsonLd(): object {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: siteConfig.name,
    url: siteConfig.siteUrl,
    description: siteConfig.description
  };
}

function buildBreadcrumbJsonLd(path: string): object {
  const segments = path.replace(/^\/zh/, "").split("/").filter(Boolean);
  const isZh = path.startsWith("/zh");
  const items = [
    { "@type": "ListItem", position: 1, name: "Home", item: absoluteUrl(isZh ? "/zh/" : "/") }
  ];

  let accumulated = isZh ? "/zh" : "";
  for (let i = 0; i < segments.length; i++) {
    accumulated += `/${segments[i]}`;
    items.push({
      "@type": "ListItem",
      position: i + 2,
      name: segments[i].replace(/-/g, " ").replace(/^\w/, (c) => c.toUpperCase()),
      item: absoluteUrl(`${accumulated}/`)
    });
  }

  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items
  };
}

function buildTechArticleJsonLd(
  title: string,
  description: string,
  canonical: string,
  ogImage: string,
  publishedAt?: string,
  modifiedAt?: string,
  authors?: { name: string }[]
): object {
  return {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: title,
    description,
    url: canonical,
    image: ogImage,
    proficiencyLevel: "Beginner",
    ...(publishedAt ? { datePublished: publishedAt } : {}),
    ...(modifiedAt ? { dateModified: modifiedAt } : {}),
    ...(authors?.length
      ? { author: authors.map((a) => ({ "@type": "Person", name: a.name })) }
      : {}),
    publisher: {
      "@type": "Organization",
      name: siteConfig.name,
      url: siteConfig.siteUrl,
      logo: { "@type": "ImageObject", url: absoluteUrl("/og/default.svg") }
    }
  };
}

function buildSoftwareSourceCodeJsonLd(name: string, description: string, repoUrl: string): object {
  return {
    "@context": "https://schema.org",
    "@type": "SoftwareSourceCode",
    name,
    description,
    codeRepository: repoUrl,
    programmingLanguage: ["C", "C++", "Rust"],
    runtimePlatform: "Linux",
    license: "https://opensource.org/licenses/MIT",
    publisher: {
      "@type": "Organization",
      name: siteConfig.name,
      url: siteConfig.siteUrl
    }
  };
}

export function SeoHead({
  title,
  description,
  path,
  alternates,
  article = false,
  publishedAt,
  metadata,
  robots,
  isTutorial = false,
  isCodeProject = false,
  repoUrl
}: SeoHeadProps) {
  const canonical = absoluteUrl(path);
  const fullTitle = `${title} | ${siteConfig.name}`;
  const ogImage = ogImageUrl();
  const feedHref = path.startsWith("/zh") ? absoluteUrl("/zh/feed.xml") : absoluteUrl("/feed.xml");
  const locale = path.startsWith("/zh") ? "zh_CN" : "en_US";
  const altLocale = locale === "en_US" ? "zh_CN" : "en_US";
  const articlePublishedAt = article ? toArticleDateTime(publishedAt ?? metadata?.createdAt) : undefined;
  const articleModifiedAt = article ? toArticleDateTime(metadata?.updatedAt) : undefined;
  const articleAuthors = article ? metadata?.authors ?? [] : [];
  const enPath = alternates.find((a) => a.hrefLang === "en")?.href;
  const xDefaultHref = enPath ?? absoluteUrl(path.startsWith("/zh") ? path.replace(/^\/zh/, "") || "/" : path);

  const jsonLdItems: object[] = [buildOrganizationJsonLd(), buildBreadcrumbJsonLd(path)];
  if (isTutorial) {
    jsonLdItems.push(
      buildTechArticleJsonLd(fullTitle, description, canonical, ogImage, articlePublishedAt, articleModifiedAt, articleAuthors)
    );
  } else if (article) {
    jsonLdItems.push(
      buildArticleJsonLd(fullTitle, description, canonical, ogImage, articlePublishedAt, articleModifiedAt, articleAuthors)
    );
  } else {
    jsonLdItems.push(buildWebSiteJsonLd());
  }
  if (isCodeProject && repoUrl) {
    jsonLdItems.push(buildSoftwareSourceCodeJsonLd(title, description, repoUrl));
  }

  return (
    <Head>
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      {robots ? <meta name="robots" content={robots} /> : null}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={ogImage} />
      <meta name="twitter:site" content="@eaborai" />
      <meta property="og:site_name" content={siteConfig.name} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:type" content={article ? "article" : "website"} />
      <meta property="og:url" content={canonical} />
      <meta property="og:image" content={ogImage} />
      <meta property="og:image:width" content={String(OG_IMAGE_WIDTH)} />
      <meta property="og:image:height" content={String(OG_IMAGE_HEIGHT)} />
      <meta property="og:image:type" content="image/svg+xml" />
      <meta property="og:locale" content={locale} />
      <meta property="og:locale:alternate" content={altLocale} />
      {articlePublishedAt ? <meta property="article:published_time" content={articlePublishedAt} /> : null}
      {articleModifiedAt ? <meta property="article:modified_time" content={articleModifiedAt} /> : null}
      {articleAuthors.map((author) => (
        <meta key={author.name} property="article:author" content={author.name} />
      ))}
      <link rel="canonical" href={canonical} />
      <link rel="alternate" hrefLang="x-default" href={xDefaultHref} />
      <link rel="alternate" type="application/rss+xml" title={`${siteConfig.name} feed`} href={feedHref} />
      {alternates.map((alternate) => (
        <link
          key={`${alternate.hrefLang}-${alternate.href}`}
          rel="alternate"
          hrefLang={alternate.hrefLang}
          href={alternate.href}
        />
      ))}
      {jsonLdItems.map((item, i) => (
        <script
          key={i}
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(item) }}
        />
      ))}
    </Head>
  );
}
