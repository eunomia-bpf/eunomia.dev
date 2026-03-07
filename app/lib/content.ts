import fs from "node:fs";
import path from "node:path";

import matter from "gray-matter";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";
import { visit } from "unist-util-visit";

import type { Locale } from "./site-data";

const repoRoot = path.resolve(process.cwd(), "..");
const docsRoot = path.join(repoRoot, "docs");
const siteRoot = path.join(repoRoot, "site");

const excludedGenericSections = new Set(["assets", "img", "blog", "blogs", "tutorials"]);
const assetExtensions = new Set([
  ".avif",
  ".gif",
  ".jpeg",
  ".jpg",
  ".json",
  ".pdf",
  ".png",
  ".svg",
  ".txt",
  ".webm",
  ".webp",
  ".xml",
  ".yml",
  ".yaml"
]);

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

type ParsedMarkdown = {
  title: string;
  description: string;
  excerpt: string;
  body: string;
  date?: string;
};

type BlogEntry = {
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

type LegacyBlogEntry = {
  key: string;
  title: string;
  description: string;
  excerpt: string;
  sourceByLocale: Partial<Record<Locale, string>>;
};

const markdownCache = new Map<string, ParsedMarkdown>();
let docsFileSetCache: Set<string> | null = null;
let siteFileSetCache: Set<string> | null = null;
let topLevelSectionsCache: string[] | null = null;
let tutorialSourcesCache: string[] | null = null;
let blogEntriesCache: BlogEntry[] | null = null;
let legacyBlogEntriesCache: LegacyBlogEntry[] | null = null;
let genericSectionRouteCache: string[] | null = null;

function toPosix(value: string): string {
  return value.split(path.sep).join("/");
}

function walkFiles(root: string): string[] {
  const queue = [root];
  const files: string[] = [];

  while (queue.length) {
    const current = queue.pop();
    if (!current) {
      continue;
    }

    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        queue.push(fullPath);
        continue;
      }
      files.push(fullPath);
    }
  }

  return files;
}

function getDocsFileSet(): Set<string> {
  if (!docsFileSetCache) {
    docsFileSetCache = new Set(
      walkFiles(docsRoot).map((filePath) => toPosix(path.relative(docsRoot, filePath)))
    );
  }
  return docsFileSetCache;
}

function getSiteFileSet(): Set<string> {
  if (!siteFileSetCache) {
    siteFileSetCache = new Set(
      walkFiles(siteRoot).map((filePath) => toPosix(path.relative(siteRoot, filePath)))
    );
  }
  return siteFileSetCache;
}

function readFile(relativePath: string, root: string = docsRoot): string {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function isLocalizedMarkdown(relativePath: string): boolean {
  return relativePath.endsWith(".zh.md");
}

function baseMarkdownPath(relativePath: string): string {
  return relativePath.replace(/\.zh\.md$/, ".md");
}

function localizedVariant(relativePath: string, locale: Locale): string {
  if (locale === "en") {
    return baseMarkdownPath(relativePath);
  }
  const basePath = baseMarkdownPath(relativePath);
  return basePath.replace(/\.md$/, ".zh.md");
}

function resolveLocalizedSource(relativePath: string, locale: Locale): string | null {
  const docsFiles = getDocsFileSet();
  const candidates =
    locale === "zh"
      ? [localizedVariant(relativePath, "zh"), baseMarkdownPath(relativePath)]
      : [baseMarkdownPath(relativePath)];

  for (const candidate of candidates) {
    if (docsFiles.has(candidate)) {
      return candidate;
    }
  }

  return null;
}

function removeLeadingHeading(markdown: string): { titleFromHeading?: string; body: string } {
  const headingMatch = markdown.match(/^\s*#\s+(.+?)\s*$/m);
  if (!headingMatch) {
    return { body: markdown.trim() };
  }

  const titleFromHeading = stripInlineMarkdown(headingMatch[1]);
  const body = markdown.replace(/^\s*#\s+(.+?)\s*$(?:\n+)?/m, "").trim();
  return { titleFromHeading, body };
}

function stripInlineMarkdown(value: string): string {
  return value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_~>#]/g, "")
    .replace(/:[a-z0-9-]+:/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeMarkdown(markdown: string): string {
  return markdown
    .replace(/\r\n?/g, "\n")
    .replace(/<!--\s*more\s*-->/gi, "\n")
    .replace(/(!?\[[^\]]*]\([^)]+\))\{[^}\n]+\}/g, "$1")
    .replace(/(#+[^\n]+?)\s*\{#[^}\n]+\}/g, "$1")
    .replace(/:[a-z0-9-]+:/gi, "")
    .trim();
}

function collapseWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function makeExcerpt(markdown: string): string {
  const blocks = markdown
    .split(/\n{2,}/)
    .map((block) => collapseWhitespace(stripInlineMarkdown(block)))
    .filter(Boolean);

  const preferredBlock = blocks.find((block) => block.length > 40) ?? blocks[0] ?? "";
  return preferredBlock.slice(0, 220);
}

function parseDate(rawValue: unknown): string | undefined {
  if (!rawValue) {
    return undefined;
  }

  if (rawValue instanceof Date && !Number.isNaN(rawValue.valueOf())) {
    return rawValue.toISOString().slice(0, 10);
  }

  if (typeof rawValue === "string") {
    const parsed = new Date(rawValue);
    if (!Number.isNaN(parsed.valueOf())) {
      return parsed.toISOString().slice(0, 10);
    }
  }

  return undefined;
}

function parseMarkdown(relativePath: string): ParsedMarkdown {
  const cached = markdownCache.get(relativePath);
  if (cached) {
    return cached;
  }

  const source = readFile(relativePath);
  const parsed = matter(source);
  const rawContent = parsed.content.replace(/\r\n?/g, "\n");
  const normalized = normalizeMarkdown(parsed.content);
  const { titleFromHeading, body } = removeLeadingHeading(normalized);

  const title =
    (typeof parsed.data.title === "string" ? collapseWhitespace(parsed.data.title) : undefined) ??
    titleFromHeading ??
    path.posix.basename(relativePath, path.posix.extname(relativePath));

  const excerptSource = rawContent.includes("<!-- more -->")
    ? rawContent.split("<!-- more -->")[0]
    : body;
  const excerpt = makeExcerpt(excerptSource);
  const description =
    (typeof parsed.data.description === "string"
      ? collapseWhitespace(stripInlineMarkdown(parsed.data.description))
      : undefined) ?? excerpt;

  const value: ParsedMarkdown = {
    title,
    description,
    excerpt,
    body,
    date: parseDate(parsed.data.date)
  };

  markdownCache.set(relativePath, value);
  return value;
}

function formatGithubSourcePath(relativePath: string): string {
  return `https://github.com/eunomia-bpf/eunomia.dev/tree/main/docs/${relativePath}`;
}

function slugifyTitle(value: string): string {
  const slug = value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  return slug || value.trim().replace(/\s+/g, "-");
}

function sortNaturally(values: string[]): string[] {
  return [...values].sort((left, right) =>
    left.localeCompare(right, "en", {
      numeric: true,
      sensitivity: "base"
    })
  );
}

function getTutorialSources(): string[] {
  if (!tutorialSourcesCache) {
    tutorialSourcesCache = sortNaturally(
      [...getDocsFileSet()].filter(
        (relativePath) =>
          relativePath.startsWith("tutorials/") &&
          relativePath.endsWith("/README.md") &&
          !isLocalizedMarkdown(relativePath)
      )
    );
  }

  return tutorialSourcesCache;
}

function buildBlogEntries(relativePrefix: "blog/posts" | "blogs"): Array<BlogEntry | LegacyBlogEntry> {
  const entriesByKey = new Map<string, Partial<Record<Locale, string>>>();

  for (const relativePath of getDocsFileSet()) {
    if (!relativePath.startsWith(`${relativePrefix}/`) || !relativePath.endsWith(".md")) {
      continue;
    }
    if (relativePath.endsWith("/index.md") || relativePath.endsWith("/index.zh.md")) {
      continue;
    }

    const key = path.posix.basename(baseMarkdownPath(relativePath), ".md");
    const locale: Locale = isLocalizedMarkdown(relativePath) ? "zh" : "en";
    const current = entriesByKey.get(key) ?? {};
    current[locale] = relativePath;
    entriesByKey.set(key, current);
  }

  const items = [...entriesByKey.entries()].map(([key, sourceByLocale]) => {
    const preferredSource = sourceByLocale.en ?? sourceByLocale.zh;
    if (!preferredSource) {
      return null;
    }

    const metadata = parseMarkdown(preferredSource);
    if (relativePrefix === "blogs") {
      return {
        key,
        title: metadata.title,
        description: metadata.description,
        excerpt: metadata.excerpt,
        sourceByLocale
      } satisfies LegacyBlogEntry;
    }

    const [year, month, day] = (metadata.date ?? "1970-01-01").split("-");
    return {
      key,
      slug: slugifyTitle(metadata.title),
      year,
      month,
      day,
      title: metadata.title,
      description: metadata.description,
      excerpt: metadata.excerpt,
      sourceByLocale
    } satisfies BlogEntry;
  });

  return items.filter(Boolean) as Array<BlogEntry | LegacyBlogEntry>;
}

function getBlogEntries(): BlogEntry[] {
  if (!blogEntriesCache) {
    blogEntriesCache = buildBlogEntries("blog/posts") as BlogEntry[];
    blogEntriesCache.sort((left, right) =>
      `${right.year}-${right.month}-${right.day}`.localeCompare(`${left.year}-${left.month}-${left.day}`)
    );
  }

  return blogEntriesCache;
}

function getLegacyBlogEntries(): LegacyBlogEntry[] {
  if (!legacyBlogEntriesCache) {
    legacyBlogEntriesCache = buildBlogEntries("blogs") as LegacyBlogEntry[];
    legacyBlogEntriesCache.sort((left, right) =>
      left.key.localeCompare(right.key, "en", {
        numeric: true,
        sensitivity: "base"
      })
    );
  }

  return legacyBlogEntriesCache;
}

export function getTopLevelSections(): string[] {
  if (!topLevelSectionsCache) {
    topLevelSectionsCache = fs
      .readdirSync(docsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .filter((name) => !excludedGenericSections.has(name))
      .sort((left, right) =>
        left.localeCompare(right, "en", {
          numeric: true,
          sensitivity: "base"
        })
      );
  }

  return topLevelSectionsCache;
}

function isSupportedSection(section: string): boolean {
  return getTopLevelSections().includes(section);
}

function docPathToRoute(relativePath: string, locale: Locale): string | null {
  const normalized = baseMarkdownPath(relativePath);
  const prefix = locale === "zh" ? "/zh" : "";

  if (normalized === "index.md") {
    return locale === "zh" ? "/zh/" : "/";
  }

  if (normalized === "blog/index.md") {
    return `${prefix}/blog/`;
  }

  if (normalized === "blogs/index.md") {
    return `${prefix}/blogs/`;
  }

  if (normalized.startsWith("blog/posts/")) {
    const key = path.posix.basename(normalized, ".md");
    const blogEntry = getBlogEntries().find((entry) => entry.key === key);
    if (!blogEntry) {
      return null;
    }
    return `${prefix}/blog/${blogEntry.year}/${blogEntry.month}/${blogEntry.day}/${blogEntry.slug}/`;
  }

  if (normalized.startsWith("blogs/")) {
    const key = path.posix.basename(normalized, ".md");
    return `${prefix}/blogs/${key}/`;
  }

  const withoutExt = normalized.replace(/\.md$/, "");
  const segments = withoutExt.split("/");
  const head = segments.shift();

  if (!head) {
    return null;
  }

  if (head === "tutorials") {
    const trailing = segments.at(-1);
    const docSegments = trailing === "README" || trailing === "index" ? segments.slice(0, -1) : segments;
    return docSegments.length ? `${prefix}/tutorials/${docSegments.join("/")}/` : `${prefix}/tutorials/`;
  }

  if (!isSupportedSection(head)) {
    return null;
  }

  const trailing = segments.at(-1);
  const docSegments = trailing === "README" || trailing === "index" ? segments.slice(0, -1) : segments;
  return docSegments.length ? `${prefix}/${head}/${docSegments.join("/")}/` : `${prefix}/${head}/`;
}

function splitSuffix(value: string): { pathname: string; search: string; hash: string } {
  const hashIndex = value.indexOf("#");
  const searchIndex = value.indexOf("?");

  const pathEnd =
    searchIndex >= 0 && hashIndex >= 0
      ? Math.min(searchIndex, hashIndex)
      : searchIndex >= 0
        ? searchIndex
        : hashIndex >= 0
          ? hashIndex
          : value.length;

  return {
    pathname: value.slice(0, pathEnd),
    search:
      searchIndex >= 0
        ? value.slice(searchIndex, hashIndex >= 0 && hashIndex > searchIndex ? hashIndex : undefined)
        : "",
    hash: hashIndex >= 0 ? value.slice(hashIndex) : ""
  };
}

function looksLikeAsset(relativePath: string): boolean {
  return assetExtensions.has(path.posix.extname(relativePath).toLowerCase());
}

function toRawAssetPath(source: "docs" | "site", relativePath: string): string {
  return `/api/raw-assets/${source}/${relativePath}`;
}

function resolveDocLinkCandidate(relativePath: string): string | null {
  const docsFiles = getDocsFileSet();
  const candidates = new Set<string>();
  const normalized = path.posix.normalize(relativePath);

  if (normalized.endsWith(".md")) {
    candidates.add(baseMarkdownPath(normalized));
    candidates.add(localizedVariant(normalized, "zh"));
  } else {
    candidates.add(`${normalized}.md`);
    candidates.add(`${normalized}.zh.md`);
    candidates.add(path.posix.join(normalized, "README.md"));
    candidates.add(path.posix.join(normalized, "README.zh.md"));
    candidates.add(path.posix.join(normalized, "index.md"));
    candidates.add(path.posix.join(normalized, "index.zh.md"));
  }

  for (const candidate of candidates) {
    if (docsFiles.has(candidate)) {
      return candidate;
    }
  }

  return null;
}

function rewriteAbsolutePath(value: string): string {
  const { pathname, search, hash } = splitSuffix(value);
  const normalized = pathname.replace(/^\/+/, "");

  if (!normalized) {
    return `${pathname}${search}${hash}`;
  }

  if (getDocsFileSet().has(normalized)) {
    const route = docPathToRoute(normalized, normalized.endsWith(".zh.md") ? "zh" : "en");
    if (route) {
      return `${route}${search}${hash}`;
    }
  }

  if (getDocsFileSet().has(normalized) || looksLikeAsset(normalized)) {
    if (getDocsFileSet().has(normalized) && looksLikeAsset(normalized)) {
      return `${toRawAssetPath("docs", normalized)}${search}${hash}`;
    }
    if (getSiteFileSet().has(normalized)) {
      return `${toRawAssetPath("site", normalized)}${search}${hash}`;
    }
  }

  if (getSiteFileSet().has(normalized)) {
    return `${toRawAssetPath("site", normalized)}${search}${hash}`;
  }

  return `${pathname}${search}${hash}`;
}

function rewriteRelativePath(value: string, sourceRelativePath: string, locale: Locale): string {
  const { pathname, search, hash } = splitSuffix(value);
  const sourceDirectory = path.posix.dirname(baseMarkdownPath(sourceRelativePath));
  const resolved = path.posix.normalize(path.posix.join(sourceDirectory, pathname));
  const explicitLocale: Locale = pathname.endsWith(".zh.md") ? "zh" : locale;

  const docCandidate = resolveDocLinkCandidate(resolved);
  if (docCandidate) {
    const route = docPathToRoute(docCandidate, explicitLocale);
    if (route) {
      return `${route}${search}${hash}`;
    }
  }

  if (getDocsFileSet().has(resolved)) {
    return `${toRawAssetPath("docs", resolved)}${search}${hash}`;
  }

  if (getSiteFileSet().has(resolved)) {
    return `${toRawAssetPath("site", resolved)}${search}${hash}`;
  }

  return `${pathname}${search}${hash}`;
}

function rewriteUrl(value: unknown, sourceRelativePath: string, locale: Locale): string | null {
  if (typeof value !== "string" || !value) {
    return null;
  }

  if (
    value.startsWith("#") ||
    value.startsWith("mailto:") ||
    value.startsWith("tel:") ||
    value.startsWith("javascript:")
  ) {
    return value;
  }

  if (/^[a-z]+:/i.test(value)) {
    return value;
  }

  if (value.startsWith("/")) {
    return rewriteAbsolutePath(value);
  }

  return rewriteRelativePath(value, sourceRelativePath, locale);
}

function createRehypeRewriter(sourceRelativePath: string, locale: Locale) {
  return function rehypeRewriter() {
    return function rewriter(tree: unknown) {
      visit(tree, "element", (node: any) => {
        if (!node.properties) {
          return;
        }

        if (typeof node.properties.href === "string") {
          node.properties.href = rewriteUrl(node.properties.href, sourceRelativePath, locale);
        }

        if (typeof node.properties.src === "string") {
          node.properties.src = rewriteUrl(node.properties.src, sourceRelativePath, locale);
        }

        if (typeof node.properties.poster === "string") {
          node.properties.poster = rewriteUrl(node.properties.poster, sourceRelativePath, locale);
        }
      });
    };
  };
}

async function renderMarkdown(relativePath: string, locale: Locale): Promise<string> {
  const parsed = parseMarkdown(relativePath);
  const processed = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, {
      allowDangerousHtml: true
    })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(createRehypeRewriter(relativePath, locale))
    .use(rehypeStringify)
    .process(parsed.body);

  return String(processed);
}

function makeAlternates(pathname: string): { en: string; zh: string } {
  const englishPath = pathname.startsWith("/zh/") ? pathname.replace(/^\/zh/, "") || "/" : pathname;
  const zhPath = pathname.startsWith("/zh/") ? pathname : pathname === "/" ? "/zh/" : `/zh${pathname}`;

  return {
    en: englishPath,
    zh: zhPath
  };
}

async function loadMarkdownPage(relativePath: string, publicPath: string, locale: Locale): Promise<MarkdownPage> {
  const sourceRelative = resolveLocalizedSource(relativePath, locale) ?? baseMarkdownPath(relativePath);
  const parsed = parseMarkdown(sourceRelative);

  return {
    title: parsed.title,
    description: parsed.description,
    html: await renderMarkdown(sourceRelative, locale),
    sourcePath: formatGithubSourcePath(sourceRelative),
    path: publicPath,
    alternates: makeAlternates(publicPath)
  };
}

function tutorialSourceToSlugSegments(sourceRelative: string): string[] {
  return sourceRelative.replace(/^tutorials\//, "").replace(/\/README\.md$/, "").split("/");
}

function sectionSourceToSlugSegments(sourceRelative: string, section: string): string[] {
  const withoutExt = sourceRelative.replace(new RegExp(`^${section}/`), "").replace(/\.md$/, "");
  const pieces = withoutExt.split("/");
  const trailing = pieces.at(-1);
  return trailing === "README" || trailing === "index" ? pieces.slice(0, -1) : pieces;
}

function getGenericSectionRouteBases(): string[] {
  if (!genericSectionRouteCache) {
    genericSectionRouteCache = sortNaturally(
      [...getDocsFileSet()].filter((relativePath) => {
        if (isLocalizedMarkdown(relativePath) || !relativePath.endsWith(".md")) {
          return false;
        }

        const [topLevel] = relativePath.split("/");
        return Boolean(topLevel && isSupportedSection(topLevel));
      })
    );
  }

  return genericSectionRouteCache;
}

export function listSitemapRoutes(): string[] {
  const routes = new Set<string>([
    "/",
    "/zh/",
    "/tutorials/",
    "/zh/tutorials/",
    "/blog/",
    "/zh/blog/",
    "/blogs/",
    "/zh/blogs/"
  ]);

  for (const sourceRelative of getTutorialSources()) {
    routes.add(docPathToRoute(sourceRelative, "en") ?? "");
    routes.add(docPathToRoute(sourceRelative, "zh") ?? "");
  }

  for (const entry of getBlogEntries()) {
    routes.add(`/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`);
    routes.add(`/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`);
  }

  for (const entry of getLegacyBlogEntries()) {
    routes.add(`/blogs/${entry.key}/`);
    routes.add(`/zh/blogs/${entry.key}/`);
  }

  for (const sourceRelative of getGenericSectionRouteBases()) {
    routes.add(docPathToRoute(sourceRelative, "en") ?? "");
    routes.add(docPathToRoute(sourceRelative, "zh") ?? "");
  }

  return [...routes].filter(Boolean).sort();
}

export async function loadHomePage(locale: Locale): Promise<{
  title: string;
  description: string;
  intro: string;
  cards: LandingCard[];
  path: string;
  alternates: {
    en: string;
    zh: string;
  };
}> {
  const home = parseMarkdown("index.md");
  const tutorials = parseMarkdown(resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md");
  const bpftime = parseMarkdown(resolveLocalizedSource("bpftime/index.md", locale) ?? "bpftime/index.md");
  const eunomiaBpf = parseMarkdown(
    resolveLocalizedSource("eunomia-bpf/index.md", locale) ?? "eunomia-bpf/index.md"
  );

  const cards: LandingCard[] = [
    {
      title: tutorials.title,
      description: tutorials.description,
      href: locale === "zh" ? "/zh/tutorials/" : "/tutorials/",
      badge: `${getTutorialSources().length} walkthroughs`
    },
    {
      title: bpftime.title,
      description: bpftime.description,
      href: locale === "zh" ? "/zh/bpftime/" : "/bpftime/",
      badge: "Userspace eBPF"
    },
    {
      title: eunomiaBpf.title,
      description: eunomiaBpf.description,
      href: locale === "zh" ? "/zh/eunomia-bpf/" : "/eunomia-bpf/",
      badge: "Toolchain"
    }
  ];

  return {
    title: home.title,
    description: home.description,
    intro: home.excerpt || home.description,
    cards,
    path: locale === "zh" ? "/zh/" : "/",
    alternates: makeAlternates(locale === "zh" ? "/zh/" : "/")
  };
}

export async function loadTutorialIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("tutorials/index.md", locale) ?? "tutorials/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdown(sourceRelative, locale);

  const cards = getTutorialSources().map((source) => {
    const localizedSource = resolveLocalizedSource(source, locale) ?? source;
    const tutorial = parseMarkdown(localizedSource);
    const slugSegments = tutorialSourceToSlugSegments(source);
    const href = locale === "zh" ? `/zh/tutorials/${slugSegments.join("/")}/` : `/tutorials/${slugSegments.join("/")}/`;

    return {
      title: tutorial.title,
      description: tutorial.description,
      href,
      badge: slugSegments.join("/")
    };
  });

  return {
    title: parsed.title,
    description: parsed.description,
    introHtml,
    path: locale === "zh" ? "/zh/tutorials/" : "/tutorials/",
    alternates: makeAlternates(locale === "zh" ? "/zh/tutorials/" : "/tutorials/"),
    cards
  };
}

export async function loadTutorialPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  if (!slugSegments?.length) {
    return null;
  }

  const baseCandidate = `tutorials/${slugSegments.join("/")}`;
  const sourceRelative =
    resolveLocalizedSource(`${baseCandidate}.md`, locale) ??
    resolveLocalizedSource(`${baseCandidate}/README.md`, locale);

  if (!sourceRelative) {
    return null;
  }

  const publicPath = locale === "zh" ? `/zh/tutorials/${slugSegments.join("/")}/` : `/tutorials/${slugSegments.join("/")}/`;
  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function loadBlogIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("blog/index.md", locale) ?? "blog/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdown(sourceRelative, locale);

  const cards = getBlogEntries().map((entry) => ({
    title: entry.title,
    description: entry.description,
    href:
      locale === "zh"
        ? `/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`
        : `/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`,
    badge: `${entry.year}-${entry.month}-${entry.day}`
  }));

  return {
    title: parsed.title,
    description: parsed.description,
    introHtml,
    path: locale === "zh" ? "/zh/blog/" : "/blog/",
    alternates: makeAlternates(locale === "zh" ? "/zh/blog/" : "/blog/"),
    cards
  };
}

export async function loadBlogPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  if (!slugSegments || slugSegments.length !== 4) {
    return null;
  }

  const [year, month, day, slug] = slugSegments;
  const entry = getBlogEntries().find(
    (candidate) =>
      candidate.year === year &&
      candidate.month === month &&
      candidate.day === day &&
      candidate.slug === slug
  );

  if (!entry) {
    return null;
  }

  const sourceRelative =
    entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
  if (!sourceRelative) {
    return null;
  }

  const publicPath =
    locale === "zh"
      ? `/zh/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`
      : `/blog/${entry.year}/${entry.month}/${entry.day}/${entry.slug}/`;

  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function loadLegacyBlogIndex(locale: Locale): Promise<LandingPageData> {
  const sourceRelative = resolveLocalizedSource("blogs/index.md", locale) ?? "blogs/index.md";
  const parsed = parseMarkdown(sourceRelative);
  const introHtml = await renderMarkdown(sourceRelative, locale);

  const cards = getLegacyBlogEntries().map((entry) => ({
    title: entry.title,
    description: entry.description,
    href: locale === "zh" ? `/zh/blogs/${entry.key}/` : `/blogs/${entry.key}/`,
    badge: "Legacy"
  }));

  return {
    title: parsed.title,
    description: parsed.description,
    introHtml,
    path: locale === "zh" ? "/zh/blogs/" : "/blogs/",
    alternates: makeAlternates(locale === "zh" ? "/zh/blogs/" : "/blogs/"),
    cards
  };
}

export async function loadLegacyBlogPage(
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  if (!slugSegments || slugSegments.length !== 1) {
    return null;
  }

  const [slug] = slugSegments;
  const entry = getLegacyBlogEntries().find((candidate) => candidate.key === slug);
  if (!entry) {
    return null;
  }

  const sourceRelative =
    entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh;
  if (!sourceRelative) {
    return null;
  }

  const publicPath = locale === "zh" ? `/zh/blogs/${slug}/` : `/blogs/${slug}/`;
  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function loadSectionPage(
  section: string,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<MarkdownPage | null> {
  if (!isSupportedSection(section)) {
    return null;
  }

  const joined = slugSegments?.join("/");
  const sourceRelative = joined
    ? resolveLocalizedSource(`${section}/${joined}.md`, locale) ??
      resolveLocalizedSource(`${section}/${joined}/README.md`, locale)
    : resolveLocalizedSource(`${section}/index.md`, locale) ?? resolveLocalizedSource(`${section}/README.md`, locale);

  if (!sourceRelative) {
    return null;
  }

  const suffix = joined ? `/${joined}/` : "/";
  const publicPath = locale === "zh" ? `/zh/${section}${suffix}` : `/${section}${suffix}`;
  return loadMarkdownPage(sourceRelative, publicPath, locale);
}

export async function serveRawAsset(source: "docs" | "site", pathSegments: string[]): Promise<{
  filePath: string;
  contentType: string;
} | null> {
  const root = source === "docs" ? docsRoot : siteRoot;
  const relativePath = path.posix.normalize(pathSegments.join("/")).replace(/^(\.\.(\/|\\|$))+/, "");
  const absolutePath = path.join(root, relativePath);

  if (!absolutePath.startsWith(root) || !fs.existsSync(absolutePath) || fs.statSync(absolutePath).isDirectory()) {
    return null;
  }

  return {
    filePath: absolutePath,
    contentType: mimeTypeFor(absolutePath)
  };
}

function mimeTypeFor(filePath: string): string {
  const extension = path.extname(filePath).toLowerCase();
  switch (extension) {
    case ".css":
      return "text/css; charset=utf-8";
    case ".gif":
      return "image/gif";
    case ".html":
      return "text/html; charset=utf-8";
    case ".jpeg":
    case ".jpg":
      return "image/jpeg";
    case ".json":
      return "application/json; charset=utf-8";
    case ".pdf":
      return "application/pdf";
    case ".png":
      return "image/png";
    case ".svg":
      return "image/svg+xml";
    case ".txt":
    case ".md":
      return "text/plain; charset=utf-8";
    case ".webm":
      return "video/webm";
    case ".webp":
      return "image/webp";
    case ".xml":
      return "application/xml; charset=utf-8";
    case ".yml":
    case ".yaml":
      return "text/yaml; charset=utf-8";
    default:
      return "application/octet-stream";
  }
}

export function getLegacyBlogRoutes(): string[][] {
  return getLegacyBlogEntries().map((entry) => [entry.key]);
}

export function getBlogRoutes(): string[][] {
  return getBlogEntries().map((entry) => [entry.year, entry.month, entry.day, entry.slug]);
}

export function getTutorialRoutes(): string[][] {
  return getTutorialSources().map((source) => tutorialSourceToSlugSegments(source));
}

export function getGenericSectionRoutes(): Array<{ section: string; slug: string[] }> {
  return getGenericSectionRouteBases().map((sourceRelative) => {
    const [section] = sourceRelative.split("/");
    return {
      section,
      slug: sectionSourceToSlugSegments(sourceRelative, section)
    };
  });
}
