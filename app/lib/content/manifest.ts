import type { Locale } from "../site-data";
import { getActiveRolloutStage, stageAllowsRoute, routeRolloutPolicies, type RolloutStage } from "../rollout";
import { useContentCache } from "./cache";
import {
  getBlogEntries,
  getGenericSectionRouteEntries,
  getLegacyBlogEntries,
  getTutorialDocSources
} from "./collections";
import { getDocsFileSet } from "./fs-index";
import {
  baseMarkdownPath,
  englishVariant,
  localizedVariant,
  resolveLocalizedSource,
  resolveSectionPageSource,
  supportedLocales,
  tutorialSourceToSlugSegments
} from "./source";
import { localizePath } from "../paths";
import type { ContentManifestRecord, LocaleAlternates } from "./types";

type CanonicalRouteDescriptor = {
  kind: ContentManifestRecord["kind"];
  key: string;
  routeClass: ContentManifestRecord["routeClass"];
  sitemapStage: RolloutStage;
  slug?: string[];
  section?: string;
  resolveSource: (locale: Locale) => string | null;
  buildPath: (locale: Locale) => string;
  getSourceAliases?: () => string[];
};

let contentManifestCache: ContentManifestRecord[] | null = null;
let canonicalDescriptorCache: CanonicalRouteDescriptor[] | null = null;
let manifestBySourceCache: Map<string, ContentManifestRecord> | null = null;
let manifestByRouteCache: Map<string, LocaleAlternates> | null = null;
let manifestRecordByRouteCache: Map<string, ContentManifestRecord> | null = null;

function buildTutorialPath(slugSegments: string[], locale: Locale): string {
  return localizePath(
    slugSegments.length ? `/tutorials/${slugSegments.join("/")}/` : "/tutorials/",
    locale
  );
}

function buildBlogPath(year: string, month: string, day: string, slug: string, locale: Locale): string {
  return localizePath(`/blog/${year}/${month}/${day}/${slug}/`, locale);
}

function buildLegacyBlogPath(key: string, locale: Locale): string {
  return localizePath(`/blogs/${key}/`, locale);
}

function buildSectionPath(section: string, slugSegments: string[], locale: Locale): string {
  return localizePath(
    slugSegments.length ? `/${section}/${slugSegments.join("/")}/` : `/${section}/`,
    locale
  );
}

function collectExistingMarkdownAliases(paths: string[]): string[] {
  const docsFiles = getDocsFileSet();
  const aliases = new Set<string>();

  for (const relativePath of paths) {
    const basePath = baseMarkdownPath(relativePath);
    for (const candidate of [basePath, englishVariant(basePath), localizedVariant(basePath, "zh")]) {
      if (docsFiles.has(candidate)) {
        aliases.add(baseMarkdownPath(candidate));
      }
    }
  }

  return [...aliases];
}

function buildCanonicalDescriptors(): CanonicalRouteDescriptor[] {
  const descriptors: CanonicalRouteDescriptor[] = [
    {
      kind: "home",
      key: "home",
      routeClass: routeRolloutPolicies.home.routeClass,
      sitemapStage: routeRolloutPolicies.home.sitemapStage,
      resolveSource: () => "index.md",
      buildPath: (locale) => localizePath("/", locale),
      getSourceAliases: () => ["index.md"]
    },
    {
      kind: "tutorial-index",
      key: "tutorials/index",
      routeClass: routeRolloutPolicies.tutorial.routeClass,
      sitemapStage: routeRolloutPolicies.tutorial.sitemapStage,
      resolveSource: (locale) => resolveLocalizedSource("tutorials/index.md", locale),
      buildPath: (locale) => localizePath("/tutorials/", locale),
      getSourceAliases: () => collectExistingMarkdownAliases(["tutorials/index.md"])
    },
    {
      kind: "blog-index",
      key: "blog/index",
      routeClass: routeRolloutPolicies["blog-index"].routeClass,
      sitemapStage: routeRolloutPolicies["blog-index"].sitemapStage,
      resolveSource: (locale) => resolveLocalizedSource("blog/index.md", locale),
      buildPath: (locale) => localizePath("/blog/", locale),
      getSourceAliases: () => collectExistingMarkdownAliases(["blog/index.md"])
    },
    {
      kind: "legacy-blog-index",
      key: "blogs/index",
      routeClass: routeRolloutPolicies["legacy-blog"].routeClass,
      sitemapStage: routeRolloutPolicies["legacy-blog"].sitemapStage,
      resolveSource: (locale) => resolveLocalizedSource("blogs/index.md", locale),
      buildPath: (locale) => localizePath("/blogs/", locale),
      getSourceAliases: () => collectExistingMarkdownAliases(["blogs/index.md"])
    }
  ];

  for (const sourceRelative of getTutorialDocSources()) {
    const slug = tutorialSourceToSlugSegments(sourceRelative);
    const tutorialBase = `tutorials/${slug.join("/")}`;
    descriptors.push({
      kind: "tutorial-page",
      key: `tutorial:${slug.join("/")}`,
      routeClass: routeRolloutPolicies.tutorial.routeClass,
      sitemapStage: routeRolloutPolicies.tutorial.sitemapStage,
      slug,
      resolveSource: (locale) => resolveLocalizedSource(sourceRelative, locale),
      buildPath: (locale) => buildTutorialPath(slug, locale),
      getSourceAliases: () =>
        collectExistingMarkdownAliases([
          `${tutorialBase}.md`,
          `${tutorialBase}/README.md`,
          `${tutorialBase}/index.md`
        ])
    });
  }

  for (const entry of getBlogEntries()) {
    descriptors.push({
      kind: "blog-page",
      key: `blog:${entry.year}-${entry.month}-${entry.day}:${entry.slug}`,
      routeClass: routeRolloutPolicies.blog.routeClass,
      sitemapStage: routeRolloutPolicies.blog.sitemapStage,
      slug: [entry.year, entry.month, entry.day, entry.slug],
      resolveSource: (locale) => entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null,
      buildPath: (locale) => buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale),
      getSourceAliases: () => collectExistingMarkdownAliases(Object.values(entry.sourceByLocale).filter(Boolean))
    });
  }

  for (const entry of getLegacyBlogEntries()) {
    descriptors.push({
      kind: "legacy-blog-page",
      key: `legacy-blog:${entry.key}`,
      routeClass: routeRolloutPolicies["legacy-blog"].routeClass,
      sitemapStage: routeRolloutPolicies["legacy-blog"].sitemapStage,
      slug: [entry.key],
      resolveSource: (locale) => entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null,
      buildPath: (locale) => buildLegacyBlogPath(entry.key, locale),
      getSourceAliases: () => collectExistingMarkdownAliases(Object.values(entry.sourceByLocale).filter(Boolean))
    });
  }

  for (const route of getGenericSectionRouteEntries()) {
    const { section, slug } = route;
    const joined = slug.join("/");
    descriptors.push({
      kind: "section-page",
      key: `section:${section}:${joined}`,
      routeClass: routeRolloutPolicies.section.routeClass,
      sitemapStage: routeRolloutPolicies.section.sitemapStage,
      section,
      slug,
      resolveSource: (locale) => resolveSectionPageSource(section, slug, locale),
      buildPath: (locale) => buildSectionPath(section, slug, locale),
      getSourceAliases: () =>
        collectExistingMarkdownAliases(
          joined
            ? [
                `${section}/${joined}.md`,
                `${section}/${joined}/README.md`,
                `${section}/${joined}/index.md`
              ]
            : [`${section}/index.md`, `${section}/README.md`]
        )
    });
  }

  return descriptors;
}

function getCanonicalDescriptors(): CanonicalRouteDescriptor[] {
  if (useContentCache && canonicalDescriptorCache) {
    return canonicalDescriptorCache;
  }

  const descriptors = buildCanonicalDescriptors();
  if (useContentCache) {
    canonicalDescriptorCache = descriptors;
  }

  return descriptors;
}

function expandDescriptor(descriptor: CanonicalRouteDescriptor): ContentManifestRecord {
  const sourceByLocale: ContentManifestRecord["sourceByLocale"] = {};
  const routeByLocale: ContentManifestRecord["routeByLocale"] = {};

  for (const locale of supportedLocales) {
    const source = descriptor.resolveSource(locale);
    if (!source) {
      continue;
    }

    sourceByLocale[locale] = source;
    routeByLocale[locale] = descriptor.buildPath(locale);
  }

  return {
    kind: descriptor.kind,
    key: descriptor.key,
    routeClass: descriptor.routeClass,
    sitemapStage: descriptor.sitemapStage,
    slug: descriptor.slug,
    section: descriptor.section,
    sourceByLocale,
    routeByLocale
  };
}

export function getContentManifest(): ContentManifestRecord[] {
  if (useContentCache && contentManifestCache) {
    return contentManifestCache;
  }

  const manifest = getCanonicalDescriptors().map(expandDescriptor);

  if (useContentCache) {
    contentManifestCache = manifest;
  }

  return manifest;
}

function setUniqueSourceAlias(
  bySource: Map<string, ContentManifestRecord>,
  key: string,
  record: ContentManifestRecord
) {
  const existing = bySource.get(key);
  if (existing && existing.key !== record.key) {
    throw new Error(`source alias collision: ${key} -> ${existing.key}, ${record.key}`);
  }
  bySource.set(key, record);
}

function setUniqueRouteAlternates(
  byRoute: Map<string, LocaleAlternates>,
  routeOwners: Map<string, string>,
  key: string,
  ownerKey: string,
  alternates: LocaleAlternates
) {
  const existingOwner = routeOwners.get(key);
  if (existingOwner && existingOwner !== ownerKey) {
    throw new Error(`route collision: ${key} -> ${existingOwner}, ${ownerKey}`);
  }
  routeOwners.set(key, ownerKey);
  byRoute.set(key, alternates);
}

function buildManifestIndexes() {
  if (useContentCache && manifestBySourceCache && manifestByRouteCache && manifestRecordByRouteCache) {
    return {
      bySource: manifestBySourceCache,
      byRoute: manifestByRouteCache,
      recordByRoute: manifestRecordByRouteCache
    };
  }

  const bySource = new Map<string, ContentManifestRecord>();
  const byRoute = new Map<string, LocaleAlternates>();
  const recordByRoute = new Map<string, ContentManifestRecord>();
  const routeOwners = new Map<string, string>();
  const descriptors = getCanonicalDescriptors();
  const manifest = getContentManifest();

  for (const [index, record] of manifest.entries()) {
    const aliases = descriptors[index]?.getSourceAliases?.() ?? [];
    for (const source of aliases.length ? aliases : Object.values(record.sourceByLocale).filter(Boolean)) {
      setUniqueSourceAlias(bySource, baseMarkdownPath(source), record);
    }

    const alternates: LocaleAlternates = {};
    if (record.routeByLocale.en) {
      alternates.en = record.routeByLocale.en;
    }
    if (record.routeByLocale.zh) {
      alternates.zh = record.routeByLocale.zh;
    }

    if (alternates.en) {
      setUniqueRouteAlternates(byRoute, routeOwners, alternates.en, record.key, alternates);
      recordByRoute.set(alternates.en, record);
    }
    if (alternates.zh) {
      setUniqueRouteAlternates(byRoute, routeOwners, alternates.zh, record.key, alternates);
      recordByRoute.set(alternates.zh, record);
    }
  }

  if (useContentCache) {
    manifestBySourceCache = bySource;
    manifestByRouteCache = byRoute;
    manifestRecordByRouteCache = recordByRoute;
  }

  return {
    bySource,
    byRoute,
    recordByRoute
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

export function listSitemapRoutes(stage: RolloutStage = getActiveRolloutStage()): string[] {
  const routes = new Set<string>();

  for (const record of getContentManifest()) {
    if (!stageAllowsRoute(record.sitemapStage, stage)) {
      continue;
    }
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
  const record = bySource.get(normalized);
  if (!record) {
    return null;
  }

  return record.routeByLocale[locale] ?? record.routeByLocale.en ?? record.routeByLocale.zh ?? null;
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

export function resolveManifestRecordFromRoute(path: string): ContentManifestRecord | null {
  const { recordByRoute } = buildManifestIndexes();
  return recordByRoute.get(path) ?? null;
}
