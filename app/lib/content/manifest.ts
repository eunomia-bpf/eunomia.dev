import type { Locale } from "../site-data";
import { useContentCache } from "./cache";
import {
  getBlogEntries,
  getGenericSectionRouteBases,
  getLegacyBlogEntries,
  getTutorialDocSources
} from "./collections";
import {
  baseMarkdownPath,
  resolveLocalizedSource,
  sectionSourceToSlugSegments,
  tutorialSourceToSlugSegments
} from "./source";
import type { ContentManifestRecord, LocaleAlternates } from "./types";

let contentManifestCache: ContentManifestRecord[] | null = null;
let manifestBySourceCache: Map<string, ContentManifestRecord> | null = null;
let manifestByRouteCache: Map<string, LocaleAlternates> | null = null;

function buildTutorialPath(slugSegments: string[], locale: Locale): string {
  const prefix = locale === "zh" ? "/zh" : "";
  return slugSegments.length ? `${prefix}/tutorials/${slugSegments.join("/")}/` : `${prefix}/tutorials/`;
}

function buildBlogPath(year: string, month: string, day: string, slug: string, locale: Locale): string {
  const prefix = locale === "zh" ? "/zh" : "";
  return `${prefix}/blog/${year}/${month}/${day}/${slug}/`;
}

function buildLegacyBlogPath(key: string, locale: Locale): string {
  const prefix = locale === "zh" ? "/zh" : "";
  return `${prefix}/blogs/${key}/`;
}

function buildSectionPath(section: string, slugSegments: string[], locale: Locale): string {
  const prefix = locale === "zh" ? "/zh" : "";
  return slugSegments.length ? `${prefix}/${section}/${slugSegments.join("/")}/` : `${prefix}/${section}/`;
}

export function getContentManifest(): ContentManifestRecord[] {
  if (useContentCache && contentManifestCache) {
    return contentManifestCache;
  }

  const manifest: ContentManifestRecord[] = [
    {
      kind: "home",
      key: "home",
      sourceByLocale: {
        en: "index.md",
        zh: "index.md"
      },
      routeByLocale: {
        en: "/",
        zh: "/zh/"
      }
    },
    {
      kind: "tutorial-index",
      key: "tutorials/index",
      sourceByLocale: {
        en: resolveLocalizedSource("tutorials/index.md", "en") ?? undefined,
        zh: resolveLocalizedSource("tutorials/index.md", "zh") ?? undefined
      },
      routeByLocale: {
        en: "/tutorials/",
        zh: "/zh/tutorials/"
      }
    },
    {
      kind: "blog-index",
      key: "blog/index",
      sourceByLocale: {
        en: resolveLocalizedSource("blog/index.md", "en") ?? undefined,
        zh: resolveLocalizedSource("blog/index.md", "zh") ?? undefined
      },
      routeByLocale: {
        en: "/blog/",
        zh: "/zh/blog/"
      }
    },
    {
      kind: "legacy-blog-index",
      key: "blogs/index",
      sourceByLocale: {
        en: resolveLocalizedSource("blogs/index.md", "en") ?? undefined,
        zh: resolveLocalizedSource("blogs/index.md", "zh") ?? undefined
      },
      routeByLocale: {
        en: "/blogs/",
        zh: "/zh/blogs/"
      }
    }
  ];

  for (const sourceRelative of getTutorialDocSources()) {
    const slug = tutorialSourceToSlugSegments(sourceRelative);
    manifest.push({
      kind: "tutorial-page",
      key: `tutorial:${slug.join("/")}`,
      slug,
      sourceByLocale: {
        en: resolveLocalizedSource(sourceRelative, "en") ?? undefined,
        zh: resolveLocalizedSource(sourceRelative, "zh") ?? undefined
      },
      routeByLocale: {
        en: resolveLocalizedSource(sourceRelative, "en") ? buildTutorialPath(slug, "en") : undefined,
        zh: resolveLocalizedSource(sourceRelative, "zh") ? buildTutorialPath(slug, "zh") : undefined
      }
    });
  }

  for (const entry of getBlogEntries()) {
    manifest.push({
      kind: "blog-page",
      key: `blog:${entry.year}-${entry.month}-${entry.day}:${entry.slug}`,
      slug: [entry.year, entry.month, entry.day, entry.slug],
      sourceByLocale: entry.sourceByLocale,
      routeByLocale: {
        en: buildBlogPath(entry.year, entry.month, entry.day, entry.slug, "en"),
        zh: buildBlogPath(entry.year, entry.month, entry.day, entry.slug, "zh")
      }
    });
  }

  for (const entry of getLegacyBlogEntries()) {
    manifest.push({
      kind: "legacy-blog-page",
      key: `legacy-blog:${entry.key}`,
      slug: [entry.key],
      sourceByLocale: entry.sourceByLocale,
      routeByLocale: {
        en: buildLegacyBlogPath(entry.key, "en"),
        zh: buildLegacyBlogPath(entry.key, "zh")
      }
    });
  }

  for (const sourceRelative of getGenericSectionRouteBases()) {
    const [section] = sourceRelative.split("/");
    const slug = sectionSourceToSlugSegments(sourceRelative, section);
    manifest.push({
      kind: "section-page",
      key: `section:${section}:${slug.join("/")}`,
      section,
      slug,
      sourceByLocale: {
        en: resolveLocalizedSource(sourceRelative, "en") ?? undefined,
        zh: resolveLocalizedSource(sourceRelative, "zh") ?? undefined
      },
      routeByLocale: {
        en: resolveLocalizedSource(sourceRelative, "en") ? buildSectionPath(section, slug, "en") : undefined,
        zh: resolveLocalizedSource(sourceRelative, "zh") ? buildSectionPath(section, slug, "zh") : undefined
      }
    });
  }

  if (useContentCache) {
    contentManifestCache = manifest;
  }

  return manifest;
}

function buildManifestIndexes() {
  if (useContentCache && manifestBySourceCache && manifestByRouteCache) {
    return {
      bySource: manifestBySourceCache,
      byRoute: manifestByRouteCache
    };
  }

  const bySource = new Map<string, ContentManifestRecord>();
  const byRoute = new Map<string, LocaleAlternates>();

  for (const record of getContentManifest()) {
    for (const source of Object.values(record.sourceByLocale)) {
      if (source) {
        bySource.set(baseMarkdownPath(source), record);
      }
    }

    const alternates: LocaleAlternates = {};
    if (record.routeByLocale.en) {
      alternates.en = record.routeByLocale.en;
      byRoute.set(record.routeByLocale.en, alternates);
    }
    if (record.routeByLocale.zh) {
      alternates.zh = record.routeByLocale.zh;
      byRoute.set(record.routeByLocale.zh, alternates);
    }
  }

  if (useContentCache) {
    manifestBySourceCache = bySource;
    manifestByRouteCache = byRoute;
  }

  return {
    bySource,
    byRoute
  };
}

function normalizeAlternates(
  alternates: LocaleAlternates | undefined,
  locale?: Locale,
  currentPath?: string
): LocaleAlternates {
  const normalized: LocaleAlternates = {};
  if (alternates?.en) {
    normalized.en = alternates.en;
  }
  if (alternates?.zh) {
    normalized.zh = alternates.zh;
  }
  if (locale && currentPath) {
    normalized[locale] ??= currentPath;
  }
  return normalized;
}

export function listSitemapRoutes(): string[] {
  const routes = new Set<string>();

  for (const record of getContentManifest()) {
    if (record.routeByLocale.en) {
      routes.add(record.routeByLocale.en);
    }
    if (record.routeByLocale.zh) {
      routes.add(record.routeByLocale.zh);
    }
  }

  return [...routes].sort();
}

export function resolveRouteFromDocSource(relativePath: string, locale: Locale): string | null {
  const normalized = baseMarkdownPath(relativePath);
  const { bySource } = buildManifestIndexes();
  const exactRecord = bySource.get(normalized);
  if (exactRecord) {
    return exactRecord.routeByLocale[locale] ?? exactRecord.routeByLocale.en ?? exactRecord.routeByLocale.zh ?? null;
  }

  for (const record of getContentManifest()) {
    const exactSource = record.sourceByLocale[locale];
    if (exactSource && baseMarkdownPath(exactSource) === normalized) {
      return record.routeByLocale[locale] ?? null;
    }

    const fallbackSources = Object.values(record.sourceByLocale).filter(Boolean) as string[];
    if (fallbackSources.some((candidate) => baseMarkdownPath(candidate) === normalized)) {
      return record.routeByLocale[locale] ?? record.routeByLocale.en ?? record.routeByLocale.zh ?? null;
    }
  }

  return null;
}

export function resolveAlternatesFromDocSource(
  relativePath: string,
  locale: Locale,
  currentPath?: string
): LocaleAlternates {
  const { bySource } = buildManifestIndexes();
  return normalizeAlternates(bySource.get(baseMarkdownPath(relativePath))?.routeByLocale, locale, currentPath);
}

export function resolveAlternatesFromRoute(path: string): LocaleAlternates {
  const { byRoute } = buildManifestIndexes();
  return normalizeAlternates(byRoute.get(path));
}
