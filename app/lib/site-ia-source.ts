import fs from "node:fs";
import path from "node:path";

import { useContentCache } from "./content/cache";
import { getDocsFileSet, getTopLevelSections } from "./content/fs-index";
import { generatedContentDir } from "./content/roots";
import {
  buildBlogIndexPath,
  buildHomePath,
  buildLegacyBlogIndexPath,
  buildSectionPath,
  buildTutorialIndexPath
} from "./content/route-paths";
import type { Locale } from "./site-data";

export type SiteSectionKey = string;

export type SerializedSiteSectionDefinition = {
  key: SiteSectionKey;
  labels: Record<Locale, string>;
  indexSource: string;
  hrefByLocale: Record<Locale, string>;
  nav: boolean;
  homeTrack: boolean;
  homeExplore: boolean;
  footerExplore: boolean;
  footerProject: boolean;
  order: number;
};

type SiteSectionOverride = {
  labels?: Partial<Record<Locale, string>>;
  nav?: boolean;
  homeTrack?: boolean;
  homeExplore?: boolean;
  footerExplore?: boolean;
  footerProject?: boolean;
  order?: number;
};

type SectionSeed = {
  key: SiteSectionKey;
  indexSource: string;
  href: (locale: Locale) => string;
  defaults: Pick<
    SerializedSiteSectionDefinition,
    "nav" | "homeTrack" | "homeExplore" | "footerExplore" | "footerProject" | "order"
  >;
};

type SerializedSiteSectionIndex = {
  generatedAt: string;
  sections: SerializedSiteSectionDefinition[];
};

const generatedSiteSectionsPath = path.join(generatedContentDir, "site-sections.json");
const sectionOverrides = new Map<SiteSectionKey, SiteSectionOverride>([
  ["tutorials", { labels: { en: "Tutorials", zh: "教程" }, order: 10 }],
  ["blog", { labels: { en: "Blog", zh: "博客" }, order: 20 }],
  [
    "legacy-blog",
    {
      labels: { en: "Legacy blog", zh: "旧博客" },
      nav: false,
      homeExplore: true,
      footerExplore: true,
      order: 70
    }
  ],
  [
    "bpftime",
    {
      labels: { en: "bpftime", zh: "bpftime" },
      homeTrack: true,
      homeExplore: false,
      footerProject: false,
      order: 30
    }
  ],
  [
    "GPTtrace",
    {
      labels: { en: "eBPF×AI/LLMs", zh: "eBPF×AI/LLMs" },
      homeTrack: false,
      homeExplore: true,
      footerProject: true,
      order: 40
    }
  ],
  [
    "eunomia-bpf",
    {
      labels: { en: "eunomia-bpf", zh: "eunomia-bpf" },
      homeTrack: true,
      homeExplore: false,
      footerProject: false,
      order: 50
    }
  ],
  [
    "others",
    {
      labels: { en: "Ecosystem", zh: "生态" },
      homeTrack: false,
      homeExplore: true,
      footerProject: false,
      order: 60
    }
  ],
  [
    "wasm-bpf",
    {
      labels: { en: "wasm-bpf", zh: "wasm-bpf" },
      nav: false,
      footerProject: true,
      order: 80
    }
  ]
]);

let sectionDefinitionsCache: SerializedSiteSectionDefinition[] | null = null;

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

function buildSectionSeeds(): SectionSeed[] {
  const seeds: SectionSeed[] = [
    {
      key: "tutorials",
      indexSource: "tutorials/index.md",
      href: buildTutorialIndexPath,
      defaults: {
        nav: true,
        homeTrack: true,
        homeExplore: false,
        footerExplore: true,
        footerProject: false,
        order: 10
      }
    },
    {
      key: "blog",
      indexSource: "blog/index.md",
      href: buildBlogIndexPath,
      defaults: {
        nav: true,
        homeTrack: false,
        homeExplore: false,
        footerExplore: true,
        footerProject: false,
        order: 20
      }
    },
    {
      key: "legacy-blog",
      indexSource: "blogs/index.md",
      href: buildLegacyBlogIndexPath,
      defaults: {
        nav: false,
        homeTrack: false,
        homeExplore: true,
        footerExplore: true,
        footerProject: false,
        order: 70
      }
    }
  ];

  for (const section of getTopLevelSections()) {
    const indexSource = resolveSectionIndexSource(section);
    if (!indexSource) {
      continue;
    }

    seeds.push({
      key: section,
      indexSource,
      href: (locale) => buildSectionPath(section, [], locale),
      defaults: {
        nav: true,
        homeTrack: false,
        homeExplore: true,
        footerExplore: true,
        footerProject: false,
        order: 1000
      }
    });
  }

  return seeds;
}

function mergeLabels(key: SiteSectionKey, override?: SiteSectionOverride): Record<Locale, string> {
  return {
    ...defaultLabels(key),
    ...(override?.labels ?? {})
  };
}

function buildSiteSectionDefinitions(): SerializedSiteSectionDefinition[] {
  const definitions = buildSectionSeeds().map((seed, index) => {
    const override = sectionOverrides.get(seed.key);

    return {
      key: seed.key,
      labels: mergeLabels(seed.key, override),
      indexSource: seed.indexSource,
      hrefByLocale: {
        en: seed.href("en"),
        zh: seed.href("zh")
      },
      nav: override?.nav ?? seed.defaults.nav,
      homeTrack: override?.homeTrack ?? seed.defaults.homeTrack,
      homeExplore: override?.homeExplore ?? seed.defaults.homeExplore,
      footerExplore: override?.footerExplore ?? seed.defaults.footerExplore,
      footerProject: override?.footerProject ?? seed.defaults.footerProject,
      order: override?.order ?? seed.defaults.order + index
    };
  });

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
