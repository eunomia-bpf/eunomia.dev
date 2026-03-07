import type { Locale } from "../site-data";

export type GitAuthor = {
  name: string;
  email?: string;
};

export type GitMetadata = {
  updatedAt?: string;
  createdAt?: string;
  authors: GitAuthor[];
};

export type MarkdownPage = {
  title: string;
  description: string;
  html: string;
  headings: HeadingEntry[];
  sourcePath: string;
  path: string;
  metadata?: GitMetadata | null;
  continuation?: PageContinuation;
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
  sourcePath: string;
  metadata?: GitMetadata | null;
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

export type HeadingEntry = {
  id: string;
  text: string;
  depth: number;
};

export type RenderedMarkdown = {
  html: string;
  headings: HeadingEntry[];
};

export type PageLink = {
  title: string;
  description: string;
  href: string;
};

export type PageContinuation = {
  index?: PageLink;
  previous?: PageLink;
  next?: PageLink;
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

export type ContentManifestKind =
  | "home"
  | "tutorial-index"
  | "tutorial-page"
  | "blog-index"
  | "blog-page"
  | "legacy-blog-index"
  | "legacy-blog-page"
  | "section-page";

export type ContentManifestRecord = {
  kind: ContentManifestKind;
  key: string;
  sourceByLocale: Partial<Record<Locale, string>>;
  routeByLocale: Partial<Record<Locale, string>>;
  slug?: string[];
  section?: string;
};

export type SearchResult = {
  title: string;
  description: string;
  href: string;
  locale: Locale;
  kind: ContentManifestKind;
  section?: string;
};
