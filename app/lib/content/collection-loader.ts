import type { Locale } from "../site-data";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import {
  buildBlogSidebar,
  buildLegacyBlogSidebar,
  buildTutorialSidebar
} from "./sidebar";
import { getBlogEntries, getLegacyBlogEntries, getTutorialReadmeSources } from "./collections";
import { getCollectionFamilyById, type CollectionFamilyId } from "./registry";
import {
  buildBlogPath,
  buildLegacyBlogPath,
  buildTutorialPath
} from "./route-paths";
import { tutorialSourceToSlugSegments } from "./source";
import type { DocsPage, LandingCard } from "./types";
import { loadDirectoryPage, loadDocumentPage, requireDocument, withContinuation } from "./page-loader-utils";
import type { SidebarGroup } from "./types";

type CollectionRuntimeConfig = {
  buildSidebar: (locale: Locale) => SidebarGroup[];
  buildIndexCards: (locale: Locale) => LandingCard[];
  resolvePageSource: (slugSegments: string[] | undefined, locale: Locale) => string | null;
  buildPagePath: (slugSegments: string[], locale: Locale) => string;
};

const collectionRuntimes: Record<CollectionFamilyId, CollectionRuntimeConfig> = {
  tutorial: {
    buildSidebar: buildTutorialSidebar,
    buildIndexCards: (locale) =>
      getTutorialReadmeSources().map((source) => {
        const tutorial = requireDocument(source, locale);
        const slugSegments = tutorialSourceToSlugSegments(source);

        return {
          title: tutorial.title,
          description: tutorial.description,
          href: buildTutorialPath(slugSegments, locale),
          badge: slugSegments.join("/")
        };
      }),
    resolvePageSource: (slugSegments, locale) => {
      if (!slugSegments?.length) {
        return null;
      }

      const baseCandidate = `tutorials/${slugSegments.join("/")}`;
      return (
        requireDocumentSource(`${baseCandidate}.md`, locale) ??
        requireDocumentSource(`${baseCandidate}/README.md`, locale)
      );
    },
    buildPagePath: (slugSegments, locale) => buildTutorialPath(slugSegments, locale)
  },
  blog: {
    buildSidebar: buildBlogSidebar,
    buildIndexCards: (locale) =>
      getBlogEntries().map((entry) => ({
        title: entry.title,
        description: entry.description,
        href: buildBlogPath(entry.year, entry.month, entry.day, entry.slug, locale),
        badge: `${entry.year}-${entry.month}-${entry.day}`
      })),
    resolvePageSource: (slugSegments, locale) => {
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

      return entry ? entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null : null;
    },
    buildPagePath: (slugSegments, locale) => {
      const [year, month, day, slug] = slugSegments;
      return buildBlogPath(year ?? "", month ?? "", day ?? "", slug ?? "", locale);
    }
  },
  "legacy-blog": {
    buildSidebar: buildLegacyBlogSidebar,
    buildIndexCards: (locale) =>
      getLegacyBlogEntries().map((entry) => ({
        title: entry.title,
        description: entry.description,
        href: buildLegacyBlogPath(entry.key, locale),
        badge: "Legacy"
      })),
    resolvePageSource: (slugSegments, locale) => {
      if (!slugSegments || slugSegments.length !== 1) {
        return null;
      }

      const [slug] = slugSegments;
      const entry = getLegacyBlogEntries().find((candidate) => candidate.key === slug);
      return entry ? entry.sourceByLocale[locale] ?? entry.sourceByLocale.en ?? entry.sourceByLocale.zh ?? null : null;
    },
    buildPagePath: (slugSegments, locale) => buildLegacyBlogPath(slugSegments[0] ?? "", locale)
  }
};

function requireDocumentSource(relativePath: string, locale: Locale): string | null {
  try {
    return requireDocument(relativePath, locale).sourceRelative;
  } catch {
    return null;
  }
}

function getCollectionRuntime(id: CollectionFamilyId): CollectionRuntimeConfig {
  return collectionRuntimes[id];
}

export async function loadCollectionIndex(
  familyId: CollectionFamilyId,
  locale: Locale
): Promise<DocsPage> {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  const runtime = getCollectionRuntime(familyId);
  const sourceRelative = requireDocument(family.indexSource, locale).sourceRelative;

  return loadDirectoryPage({
    sourceRelative,
    publicPath: family.buildIndexPath(locale),
    locale,
    cards: runtime.buildIndexCards(locale),
    sidebar: runtime.buildSidebar(locale)
  });
}

export async function loadCollectionPage(
  familyId: CollectionFamilyId,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  const runtime = getCollectionRuntime(familyId);
  const sourceRelative = runtime.resolvePageSource(slugSegments, locale);
  if (!sourceRelative || !slugSegments?.length) {
    return null;
  }

  const publicPath = runtime.buildPagePath(slugSegments, locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const continuation = buildCollectionContinuation({
    kind: family.pageKind,
    locale,
    currentPath: publicPath,
    index: buildIndexLink(requireDocument(family.indexSource, locale).sourceRelative, family.buildIndexPath(locale))
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: runtime.buildSidebar(locale)
  };
}

export function loadTutorialIndex(locale: Locale) {
  return loadCollectionIndex("tutorial", locale);
}

export function loadTutorialPage(slugSegments: string[] | undefined, locale: Locale) {
  return loadCollectionPage("tutorial", slugSegments, locale);
}

export function loadBlogIndex(locale: Locale) {
  return loadCollectionIndex("blog", locale);
}

export function loadBlogPage(slugSegments: string[] | undefined, locale: Locale) {
  return loadCollectionPage("blog", slugSegments, locale);
}

export function loadLegacyBlogIndex(locale: Locale) {
  return loadCollectionIndex("legacy-blog", locale);
}

export function loadLegacyBlogPage(slugSegments: string[] | undefined, locale: Locale) {
  return loadCollectionPage("legacy-blog", slugSegments, locale);
}
