import type { Locale } from "../site-data";
import type { RolloutStage, RouteClass } from "../rollout";

export type LocaleAlternates = Partial<Record<Locale, string>>;

export type GitAuthor = {
  name: string;
  email?: string;
};

export type GitMetadata = {
  updatedAt?: string;
  createdAt?: string;
  authors: GitAuthor[];
};

export type LandingCard = {
  title: string;
  description: string;
  href: string;
  badge?: string;
};

export type ParsedMarkdown = {
  title: string;
  description: string;
  excerpt: string;
  body: string;
  date?: string;
};

export type DocumentRecord = ParsedMarkdown & {
  sourcePath: string;
  baseSourcePath: string;
  locale: Locale;
  section: string;
};

export type HeadingEntry = {
  id: string;
  text: string;
  depth: number;
};

export type SidebarItem = {
  title: string;
  href: string;
  depth?: number;
};

export type SidebarGroup = {
  title: string;
  items: SidebarItem[];
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
  date?: string;
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
  routeClass: RouteClass;
  sitemapStage: RolloutStage;
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

export type DocsPage = {
  layout: "directory" | "document";
  title: string;
  description: string;
  bodyHtml: string;
  sourcePath: string;
  path: string;
  metadata?: GitMetadata | null;
  alternates: LocaleAlternates;
  sidebar?: SidebarGroup[];
  cards?: LandingCard[];
  headings?: HeadingEntry[];
  continuation?: PageContinuation;
};
