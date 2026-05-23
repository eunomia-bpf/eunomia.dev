import fs from "node:fs";
import path from "node:path";

import { useContentCache } from "./content/cache";
import { getDocsFileSet, getTopLevelSections } from "./content/fs-index";
import {
  readMkdocsSiteSections,
  readMkdocsTopLevelNavSections,
  type MkdocsSiteSectionConfig,
  type MkdocsTopLevelNavSection
} from "./content/mkdocs-config";
import { getCollectionFamilies, type PublishedSiteSectionFlags } from "./content/registry";
import { generatedContentDir } from "./content/roots";
import {
  buildHomePath,
  buildSectionPath
} from "./content/route-paths";
import type { Locale } from "./site-data";

export type SiteSectionKey = string;

export type SerializedPublishedSiteSection = PublishedSiteSectionFlags;

export type SerializedSiteSectionDefinition = {
  key: SiteSectionKey;
  labels: Record<Locale, string>;
  indexSource: string;
  hrefByLocale: Record<Locale, string>;
  discovered: boolean;
  published: SerializedPublishedSiteSection;
  order: number;
};

type SiteSectionOverride = MkdocsSiteSectionConfig;

type SectionSeed = {
  key: SiteSectionKey;
  topLevelDir: string;
  indexSource: string;
  href: (locale: Locale) => string;
  defaults: SerializedPublishedSiteSection;
  defaultOrder: number;
};

type SerializedSiteSectionIndex = {
  generatedAt: string;
  sections: SerializedSiteSectionDefinition[];
};

const generatedSiteSectionsPath = path.join(generatedContentDir, "site-sections.json");

let sectionDefinitionsCache: SerializedSiteSectionDefinition[] | null = null;

function assertNonEmptyLabel(key: SiteSectionKey, locale: Locale, value: string | undefined) {
  if (!value?.trim()) {
    throw new Error(`Missing site IA label for section "${key}" (${locale})`);
  }
}

function humanizeKey(key: string): string {
  return key
    .split(/[-_/]/)
    .filter(Boolean)
    .map((piece) => {
      if (piece.toUpperCase() === piece) {
        return piece;
      }

      return `${piece.slice(0, 1).toUpperCase()}${piece.slice(1)}`;
    })
    .join(" ");
}

function resolveSectionIndexSource(section: string): string | null {
  const docsFiles = getDocsFileSet();
  const indexBase = `${section}/index.md`;
  const readmeBase = `${section}/README.md`;
  const candidates = [
    [indexBase, `${section}/index.en.md`, `${section}/index.zh.md`],
    [readmeBase, `${section}/README.en.md`, `${section}/README.zh.md`]
  ];

  for (const [base, ...variants] of candidates) {
    if ([base, ...variants].some((candidate) => docsFiles.has(candidate))) {
      return base;
    }
  }

  return null;
}

function defaultLabels(key: SiteSectionKey): Record<Locale, string> {
  return {
    en: humanizeKey(key),
    zh: key
  };
}

function buildCollectionSectionSeeds(): SectionSeed[] {
  return getCollectionFamilies().map((family, index) => ({
    key: family.siteSection.key,
    topLevelDir: family.siteSection.topLevelDir,
    indexSource: family.indexSource,
    href: family.buildIndexPath,
    defaults: family.siteSection.defaults,
    defaultOrder: 10 + index * 10
  }));
}

function validateSectionOverrides(
  seeds: SectionSeed[],
  sectionOverrides: Map<SiteSectionKey, SiteSectionOverride>
) {
  const discoveredKeys = new Set(seeds.map((seed) => seed.key));
  const unknownKeys = [...sectionOverrides.keys()].filter((key) => !discoveredKeys.has(key));
  if (unknownKeys.length) {
    throw new Error(`Unknown site IA overrides: ${unknownKeys.join(", ")}`);
  }
}

function buildDiscoveredSectionSeeds(sectionOverrides: Map<SiteSectionKey, SiteSectionOverride>): SectionSeed[] {
  const seeds = [...buildCollectionSectionSeeds()];
  const reservedTopLevelDirs = new Set(seeds.map((seed) => seed.topLevelDir));

  for (const section of getTopLevelSections()) {
    if (reservedTopLevelDirs.has(section)) {
      continue;
    }

    const indexSource = resolveSectionIndexSource(section);
    if (!indexSource) {
      continue;
    }

    seeds.push({
      key: section,
      topLevelDir: section,
      indexSource,
      href: (locale) => buildSectionPath(section, [], locale),
      defaults: {
        nav: false,
        homeTrack: false,
        homeExplore: false,
        footerExplore: false,
        footerProject: false,
      },
      defaultOrder: 1000
    });
  }

  validateSectionOverrides(seeds, sectionOverrides);
  return seeds;
}

function mergeLabels(key: SiteSectionKey, override?: SiteSectionOverride): Record<Locale, string> {
  return {
    ...defaultLabels(key),
    ...(override?.labels ?? {})
  };
}

function mergePublished(
  seed: SectionSeed,
  override?: SiteSectionOverride,
  mkdocsNavSection?: MkdocsTopLevelNavSection
): SerializedPublishedSiteSection {
  const published = {
    ...seed.defaults,
    ...(override?.published ?? {})
  };

  if (mkdocsNavSection && override?.published?.nav === undefined) {
    published.nav = true;
  }

  return published;
}

function buildSiteSectionDefinitions(): SerializedSiteSectionDefinition[] {
  const mkdocsNavSections = new Map(readMkdocsTopLevelNavSections().map((section) => [section.key, section]));
  const sectionOverrides = readMkdocsSiteSections();
  const definitions = buildDiscoveredSectionSeeds(sectionOverrides).map((seed, index) => {
    const override = sectionOverrides.get(seed.key);
    const mkdocsNavSection = mkdocsNavSections.get(seed.key);
    const labels = mergeLabels(seed.key, override);
    assertNonEmptyLabel(seed.key, "en", labels.en);
    assertNonEmptyLabel(seed.key, "zh", labels.zh);

    return {
      key: seed.key,
      labels,
      indexSource: seed.indexSource,
      hrefByLocale: {
        en: seed.href("en"),
        zh: seed.href("zh")
      },
      discovered: true as const,
      published: mergePublished(seed, override, mkdocsNavSection),
      order: override?.order ?? mkdocsNavSection?.order ?? seed.defaultOrder + index
    };
  });

  const seenKeys = new Set<SiteSectionKey>();
  const seenOrders = new Map<number, SiteSectionKey>();
  for (const definition of definitions) {
    if (seenKeys.has(definition.key)) {
      throw new Error(`Duplicate site IA section key: ${definition.key}`);
    }
    seenKeys.add(definition.key);

    const existingOrder = seenOrders.get(definition.order);
    if (existingOrder && existingOrder !== definition.key) {
      throw new Error(
        `Duplicate site IA order ${definition.order} for sections "${existingOrder}" and "${definition.key}"`
      );
    }
    seenOrders.set(definition.order, definition.key);
  }

  definitions.sort((left, right) => {
    if (left.order !== right.order) {
      return left.order - right.order;
    }

    return left.key.localeCompare(right.key, "en", {
      numeric: true,
      sensitivity: "base"
    });
  });

  return definitions;
}

export function getGeneratedSiteSections(): SerializedSiteSectionDefinition[] {
  if (useContentCache && sectionDefinitionsCache) {
    return sectionDefinitionsCache;
  }

  const definitions = buildSiteSectionDefinitions();
  if (useContentCache) {
    sectionDefinitionsCache = definitions;
  }

  return definitions;
}

export function writeSiteSections(outputPath: string = generatedSiteSectionsPath) {
  const sections = buildSiteSectionDefinitions();
  const payload: SerializedSiteSectionIndex = {
    generatedAt: new Date().toISOString(),
    sections
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const tempPath = `${outputPath}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.renameSync(tempPath, outputPath);

  if (useContentCache) {
    sectionDefinitionsCache = sections;
  }

  return {
    count: sections.length,
    filePath: outputPath
  };
}

export function getHomePath(locale: Locale): string {
  return buildHomePath(locale);
}
