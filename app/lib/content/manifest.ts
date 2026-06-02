import fs from "node:fs";
import path from "node:path";

import type { Locale } from "../site-data";
import { getActiveRolloutStage, stageAllowsRoute, routeRolloutPolicies, type RolloutStage } from "../rollout";
import { useContentCache } from "./cache";
import { getGenericSectionRouteEntries } from "./collections";
import { generatedContentDir } from "./roots";
import { buildHomePath, buildSectionPath } from "./route-paths";
import { getCollectionFamilies } from "./registry";
import {
  baseMarkdownPath,
  resolveLocalizedSource,
  resolveSectionPageSource,
  supportedLocales
} from "./source";
import type { ContentManifestRecord, LocaleAlternates } from "./types";

type SerializedContentManifestRecord = ContentManifestRecord & {
  sourceAliases: string[];
};

const generatedContentManifestPath = path.join(generatedContentDir, "manifest.json");

let contentManifestCache: SerializedContentManifestRecord[] | null = null;
let manifestBySourceCache: Map<string, SerializedContentManifestRecord> | null = null;
let manifestByRouteCache: Map<string, LocaleAlternates> | null = null;
let manifestRecordByRouteCache: Map<string, SerializedContentManifestRecord> | null = null;

const routeAliasesByCanonicalPath: Record<string, string[]> = {
  "/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/": [
    "/blog/2026/02/13/reverse-engineering-claude-code-s-ssl-traffic-with-ebpf/"
  ],
  "/zh/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/": [
    "/zh/blog/2026/02/13/reverse-engineering-claude-code-s-ssl-traffic-with-ebpf/"
  ]
};

const sitemapExcludedRoutes = new Set(["/GPTtrace/agentsight/", "/zh/GPTtrace/agentsight/"]);

export function isSitemapExcludedRoute(routePath: string): boolean {
  return sitemapExcludedRoutes.has(routePath);
}

function allowManifestFallback(): boolean {
  return process.env.NODE_ENV === "development";
}

function getRouteAliases(canonicalPath: string | undefined): string[] {
  return canonicalPath ? routeAliasesByCanonicalPath[canonicalPath] ?? [] : [];
}

function resolvePageSource(
  sourceByLocale: Partial<Record<Locale, string>>,
  locale: Locale
): string | null {
  return sourceByLocale[locale] ?? sourceByLocale.en ?? sourceByLocale.zh ?? null;
}

function buildManifestFromSource(): SerializedContentManifestRecord[] {
  const manifest: SerializedContentManifestRecord[] = [
    {
      kind: "home",
      key: "home",
      routeClass: routeRolloutPolicies.home.routeClass,
      sitemapStage: routeRolloutPolicies.home.sitemapStage,
      sourceByLocale: {
        en: resolveLocalizedSource("index.md", "en") ?? "index.md",
        zh: resolveLocalizedSource("index.md", "zh") ?? "index.md"
      },
      routeByLocale: {
        en: buildHomePath("en"),
        zh: buildHomePath("zh")
      },
      sourceAliases: ["index.md"]
    }
  ];

  for (const family of getCollectionFamilies()) {
    const indexSourceByLocale: Partial<Record<Locale, string>> = {};
    const indexRouteByLocale: Partial<Record<Locale, string>> = {};

    for (const locale of supportedLocales) {
      const source = resolveLocalizedSource(family.indexSource, locale);
      if (!source) {
        continue;
      }

      indexSourceByLocale[locale] = source;
      indexRouteByLocale[locale] = family.buildIndexPath(locale);
    }

    manifest.push({
      kind: family.indexKind,
      key: family.indexKey,
      routeClass: family.indexRouteClass,
      sitemapStage: family.indexSitemapStage,
      sourceByLocale: indexSourceByLocale,
      routeByLocale: indexRouteByLocale,
      sourceAliases: [baseMarkdownPath(family.indexSource)]
    });

    for (const page of family.getPageDescriptorsFromSource()) {
      const sourceByLocale: Partial<Record<Locale, string>> = {};
      const routeByLocale: Partial<Record<Locale, string>> = {};

      for (const locale of supportedLocales) {
        const source = resolvePageSource(page.sourceByLocale, locale);
        if (!source) {
          continue;
        }

        sourceByLocale[locale] = source;
        routeByLocale[locale] = page.buildPath(locale);
      }

      manifest.push({
        kind: page.kind,
        key: page.key,
        routeClass: family.pageRouteClass,
        sitemapStage: family.pageSitemapStage,
        slug: page.slug,
        sourceByLocale,
        routeByLocale,
        sourceAliases: [...new Set(page.sourceAliases.map(baseMarkdownPath))]
      });
    }
  }

  for (const route of getGenericSectionRouteEntries()) {
    const routeSourceByLocale: Partial<Record<Locale, string>> = {};
    const routeByLocale: Partial<Record<Locale, string>> = {};

    for (const locale of supportedLocales) {
      const source = resolveSectionPageSource(route.section, route.slug, locale);
      if (!source) {
        continue;
      }

      routeSourceByLocale[locale] = source;
      routeByLocale[locale] = buildSectionPath(route.section, route.slug, locale);
    }

    manifest.push({
      kind: "section-page",
      key: `section:${route.section}:${route.slug.join("/")}`,
      routeClass: routeRolloutPolicies.section.routeClass,
      sitemapStage: routeRolloutPolicies.section.sitemapStage,
      slug: route.slug,
      section: route.section,
      sourceByLocale: routeSourceByLocale,
      routeByLocale,
      sourceAliases: [...new Set(route.sourceAliases.map(baseMarkdownPath))]
    });
  }

  return manifest;
}

function readPrebuiltContentManifest(
  filePath: string = generatedContentManifestPath
): SerializedContentManifestRecord[] | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as {
      generatedAt?: string;
      manifest?: SerializedContentManifestRecord[];
    };

    if (!Array.isArray(payload.manifest)) {
      return null;
    }

    return payload.manifest;
  } catch (error) {
    if (!allowManifestFallback() && filePath === generatedContentManifestPath) {
      throw new Error(`Failed to read prebuilt content manifest: ${String(error)}`);
    }

    console.warn("Failed to read prebuilt content manifest. Falling back to source build.", error);
    return null;
  }
}

function getSerializedContentManifest(
  options: {
    allowFallback?: boolean;
    outputPath?: string;
  } = {}
): SerializedContentManifestRecord[] {
  const outputPath = options.outputPath ?? generatedContentManifestPath;
  const fallbackAllowed = options.allowFallback ?? allowManifestFallback();

  if (useContentCache && outputPath === generatedContentManifestPath && contentManifestCache) {
    return contentManifestCache;
  }

  const prebuilt = readPrebuiltContentManifest(outputPath);
  if (prebuilt) {
    if (useContentCache && outputPath === generatedContentManifestPath) {
      contentManifestCache = prebuilt;
    }
    return prebuilt;
  }

  if (!fallbackAllowed) {
    throw new Error(`Missing prebuilt content manifest at ${outputPath}. Run generate:content-index first.`);
  }

  const rebuilt = buildManifestFromSource();
  if (useContentCache && outputPath === generatedContentManifestPath) {
    contentManifestCache = rebuilt;
  }
  return rebuilt;
}

export function writeContentManifest(outputPath: string = generatedContentManifestPath) {
  const manifest = buildManifestFromSource();
  const payload = {
    generatedAt: new Date().toISOString(),
    manifest
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const tempPath = `${outputPath}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(payload)}\n`, "utf8");
  fs.renameSync(tempPath, outputPath);

  if (useContentCache && outputPath === generatedContentManifestPath) {
    contentManifestCache = manifest;
  }

  return {
    count: manifest.length,
    filePath: outputPath
  };
}

export function getContentManifest(): ContentManifestRecord[] {
  return getSerializedContentManifest();
}

function setUniqueSourceAlias(
  bySource: Map<string, SerializedContentManifestRecord>,
  key: string,
  record: SerializedContentManifestRecord
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

  const bySource = new Map<string, SerializedContentManifestRecord>();
  const byRoute = new Map<string, LocaleAlternates>();
  const recordByRoute = new Map<string, SerializedContentManifestRecord>();
  const routeOwners = new Map<string, string>();
  const manifest = getSerializedContentManifest();

  for (const record of manifest) {
    for (const source of record.sourceAliases.length ? record.sourceAliases : Object.values(record.sourceByLocale)) {
      if (!source) {
        continue;
      }
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
      for (const alias of getRouteAliases(alternates.en)) {
        setUniqueRouteAlternates(byRoute, routeOwners, alias, record.key, alternates);
        recordByRoute.set(alias, record);
      }
    }
    if (alternates.zh) {
      setUniqueRouteAlternates(byRoute, routeOwners, alternates.zh, record.key, alternates);
      recordByRoute.set(alternates.zh, record);
      for (const alias of getRouteAliases(alternates.zh)) {
        setUniqueRouteAlternates(byRoute, routeOwners, alias, record.key, alternates);
        recordByRoute.set(alias, record);
      }
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

  for (const record of getSerializedContentManifest()) {
    if (!stageAllowsRoute(record.sitemapStage, stage)) {
      continue;
    }
    if (record.routeByLocale.en && !isSitemapExcludedRoute(record.routeByLocale.en)) {
      routes.add(record.routeByLocale.en);
    }
    if (record.routeByLocale.zh && !isSitemapExcludedRoute(record.routeByLocale.zh)) {
      routes.add(record.routeByLocale.zh);
    }
  }

  return [...routes].sort();
}

export function listRenderableRoutesForLocale(locale: Locale): string[] {
  const routes = new Set<string>();

  for (const record of getSerializedContentManifest()) {
    const canonicalRoute = record.routeByLocale[locale];
    if (!canonicalRoute) {
      continue;
    }

    routes.add(canonicalRoute);
    for (const alias of getRouteAliases(canonicalRoute)) {
      routes.add(alias);
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
