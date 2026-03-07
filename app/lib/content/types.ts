import type { Locale } from "../site-data";

export type MarkdownPage = {
  title: string;
  description: string;
  html: string;
  sourcePath: string;
  path: string;
  alternates: {
    en: string;
    zh: string;
  };
};

export type LandingCard = {
  title: string;
  description: string;
  href: string;
  badge?: string;
};

export type LandingPageData = {
  title: string;
  description: string;
  introHtml: string;
  path: string;
  alternates: {
    en: string;
    zh: string;
  };
  cards: LandingCard[];
};

export type ParsedMarkdown = {
  title: string;
  description: string;
  excerpt: string;
  body: string;
  date?: string;
};

export type BlogEntry = {
  key: string;
  slug: string;
  year: string;
  month: string;
  day: string;
  title: string;
  description: string;
  excerpt: string;
  sourceByLocale: Partial<Record<Locale, string>>;
};

export type LegacyBlogEntry = {
  key: string;
  title: string;
  description: string;
  excerpt: string;
  sourceByLocale: Partial<Record<Locale, string>>;
};

export type ContentManifestRecord = {
  kind: "home" | "tutorial" | "blog" | "legacy-blog" | "section";
  key: string;
  sourceByLocale: Partial<Record<Locale, string>>;
  routeByLocale: Partial<Record<Locale, string>>;
};
