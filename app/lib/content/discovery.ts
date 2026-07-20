import path from "node:path";

import type { Locale } from "../site-data";
import { useContentCache } from "./cache";
import { listDocuments } from "./documents";
import { getDocsFileSet } from "./fs-index";
import { orderSourcesByMkdocsNav } from "./nav-order";
import {
  baseMarkdownPath,
  isLocalizedMarkdown,
  isSupportedSection,
  sectionSourceToSlugSegments,
  slugifyTitle,
  sortNaturally
} from "./source";
import type { BlogEntry, GenericSectionRouteEntry, LegacyBlogEntry } from "./types";

let tutorialReadmeSourcesCache: string[] | null = null;
let tutorialDocSourcesCache: string[] | null = null;
let blogEntriesCache: BlogEntry[] | null = null;
let legacyBlogEntriesCache: LegacyBlogEntry[] | null = null;
let genericSectionRouteCache: GenericSectionRouteEntry[] | null = null;

export function discoverTutorialReadmeSources(): string[] {
  if (!useContentCache || !tutorialReadmeSourcesCache) {
    tutorialReadmeSourcesCache = sortNaturally(
      [...getDocsFileSet()].filter(
        (relativePath) =>
          relativePath.startsWith("tutorials/") &&
          relativePath.endsWith("/README.md") &&
          !isLocalizedMarkdown(relativePath)
      )
    );
  }

  return tutorialReadmeSourcesCache;
}

export function discoverTutorialDocSources(): string[] {
  if (!useContentCache || !tutorialDocSourcesCache) {
    tutorialDocSourcesCache = sortNaturally(
      [...new Set(
        [...getDocsFileSet()]
          .filter((relativePath) => relativePath.startsWith("tutorials/") && relativePath.endsWith(".md"))
          .map((relativePath) => baseMarkdownPath(relativePath))
          .filter((relativePath) => relativePath !== "tutorials/index.md")
      )]
    );
  }

  return tutorialDocSourcesCache;
}

function buildBlogEntries(relativePrefix: "blog/posts" | "blogs"): Array<BlogEntry | LegacyBlogEntry> {
  const entriesByKey = new Map<string, Partial<Record<Locale, string>>>();
  const metadataBySource = new Map(
    listDocuments()
      .filter((document) => document.sourceRelative.startsWith(`${relativePrefix}/`))
      .map((document) => [document.sourceRelative, document] as const)
  );

  for (const relativePath of metadataBySource.keys()) {
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

    const metadata = metadataBySource.get(preferredSource);
    if (!metadata) {
      return null;
    }

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
      slug: metadata.slug ?? slugifyTitle(metadata.title),
      date: metadata.date,
      year,
      month,
      day,
      title: metadata.title,
      description: metadata.description,
      excerpt: metadata.excerpt,
      tags: metadata.tags,
      sourceByLocale
    } satisfies BlogEntry;
  });

  return items.filter(Boolean) as Array<BlogEntry | LegacyBlogEntry>;
}

export function discoverBlogEntries(): BlogEntry[] {
  if (!useContentCache || !blogEntriesCache) {
    blogEntriesCache = buildBlogEntries("blog/posts") as BlogEntry[];
    blogEntriesCache.sort((left, right) => (right.date ?? "").localeCompare(left.date ?? ""));
  }

  return blogEntriesCache;
}

export function discoverLegacyBlogEntries(): LegacyBlogEntry[] {
  if (!useContentCache || !legacyBlogEntriesCache) {
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

export function discoverGenericSectionRouteEntries(): GenericSectionRouteEntry[] {
  if (!useContentCache || !genericSectionRouteCache) {
    const orderedSources = orderSourcesByMkdocsNav(
      [...new Set(
        [...getDocsFileSet()]
          .filter((relativePath) => relativePath.endsWith(".md"))
          .map((relativePath) => baseMarkdownPath(relativePath))
          .filter((relativePath) => {
            const [topLevel] = relativePath.split("/");
            return Boolean(topLevel && isSupportedSection(topLevel));
          })
      )]
    );

    const routes = new Map<string, GenericSectionRouteEntry>();

    for (const sourceRelative of orderedSources) {
      const [section] = sourceRelative.split("/");
      if (!section) {
        continue;
      }

      const slug = sectionSourceToSlugSegments(sourceRelative, section);
      const key = `${section}:${slug.join("/")}`;
      const existing = routes.get(key);

      if (existing) {
        if (!existing.sourceAliases.includes(sourceRelative)) {
          existing.sourceAliases.push(sourceRelative);
        }
        continue;
      }

      routes.set(key, {
        section,
        slug,
        sourceAliases: [sourceRelative]
      });
    }

    genericSectionRouteCache = [...routes.values()];
  }

  return genericSectionRouteCache;
}
