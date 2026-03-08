import type { Locale } from "../site-data";
import { getActiveRolloutStage, stageAllowsRoute, routeRolloutPolicies, type RolloutStage } from "../rollout";
import { useContentCache } from "./cache";
import { getGenericSectionRouteEntries } from "./collections";
import { getDocsFileSet } from "./fs-index";
import { buildHomePath, buildSectionPath } from "./route-paths";
import { getCollectionFamilies } from "./registry";
import {
  baseMarkdownPath,
  englishVariant,
  localizedVariant,
  resolveLocalizedSource,
  resolveSectionPageSource,
  supportedLocales
} from "./source";
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
      buildPath: (locale) => buildHomePath(locale),
      getSourceAliases: () => ["index.md"]
    }
  ];

  for (const family of getCollectionFamilies()) {
    descriptors.push({
      kind: family.indexKind,
      key: family.indexKey,
      routeClass: family.indexRouteClass,
      sitemapStage: family.indexSitemapStage,
      resolveSource: (locale) => resolveLocalizedSource(family.indexSource, locale),
      buildPath: family.buildIndexPath,
      getSourceAliases: () => collectExistingMarkdownAliases([family.indexSource])
    });

    for (const page of family.getPages()) {
      descriptors.push({
        kind: page.kind,
        key: page.key,
        routeClass: family.pageRouteClass,
        sitemapStage: family.pageSitemapStage,
        slug: page.slug,
        resolveSource: page.resolveSource,
        buildPath: page.buildPath,
        getSourceAliases: () => collectExistingMarkdownAliases(page.getSourceAliases())
      });
    }
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
