import path from "node:path";

import type { Locale } from "../site-data";
import { getDocsFileSet } from "./fs-index";
import { parseMarkdown } from "./markdown";
import {
  baseMarkdownPath,
  isLocalizedMarkdown,
  isSupportedSection,
  resolveLocalizedSource,
  slugifyTitle,
  sortNaturally
} from "./source";
import type { BlogEntry, LegacyBlogEntry } from "./types";

let tutorialSourcesCache: string[] | null = null;
let blogEntriesCache: BlogEntry[] | null = null;
let legacyBlogEntriesCache: LegacyBlogEntry[] | null = null;
let genericSectionRouteCache: string[] | null = null;

export function getTutorialReadmeSources(): string[] {
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

export function getTutorialDocSources(): string[] {
  return sortNaturally(
    [...new Set(
      [...getDocsFileSet()]
        .filter(
          (relativePath) =>
            relativePath.startsWith("tutorials/") &&
            relativePath.endsWith(".md")
        )
        .map((relativePath) => baseMarkdownPath(relativePath))
        .filter((relativePath) => relativePath !== "tutorials/index.md")
    )]
  );
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

export function getBlogEntries(): BlogEntry[] {
  if (!blogEntriesCache) {
    blogEntriesCache = buildBlogEntries("blog/posts") as BlogEntry[];
    blogEntriesCache.sort((left, right) =>
      `${right.year}-${right.month}-${right.day}`.localeCompare(`${left.year}-${left.month}-${left.day}`)
    );
  }

  return blogEntriesCache;
}

export function getLegacyBlogEntries(): LegacyBlogEntry[] {
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

export function getGenericSectionRouteBases(): string[] {
  if (!genericSectionRouteCache) {
    genericSectionRouteCache = sortNaturally(
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
  }

  return genericSectionRouteCache;
}
