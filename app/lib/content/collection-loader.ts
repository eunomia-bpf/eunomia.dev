import type { Locale } from "../site-data";
import { buildCollectionContinuation, buildIndexLink } from "./navigation";
import {
  getCollectionFamilyById,
  getCollectionPageDescriptors,
  resolveCollectionPageSource,
  type CollectionFamilyDefinition,
  type CollectionFamilyId
} from "./registry";
import { buildCollectionSidebar } from "./sidebar";
import type { BlogEntry, DocsPage, LandingCard } from "./types";
import { getDocument } from "./documents";
import { loadDirectoryPage, loadDocumentPage, requireDocument, withContinuation } from "./page-loader-utils";
import { getBlogEntriesForLocale } from "./collections";

function buildCollectionIndexCards(familyId: CollectionFamilyId, locale: Locale): LandingCard[] {
  return getCollectionPageDescriptors(familyId)
    .map((descriptor) => {
      const source =
        descriptor.sourceByLocale[locale] ?? descriptor.sourceByLocale.en ?? descriptor.sourceByLocale.zh ?? null;
      if (!source) {
        return null;
      }

      const document = getDocument(source);
      return {
        title: document.title,
        description: document.description,
        href: descriptor.buildPath(locale)
      } satisfies LandingCard;
    })
    .filter((card): card is LandingCard => Boolean(card));
}

async function loadCollectionIndexPage(
  family: CollectionFamilyDefinition,
  locale: Locale
): Promise<DocsPage | null> {
  const document = requireDocument(family.indexSource, locale);
  if (!document) {
    return null;
  }

  const page = await loadDirectoryPage({
    sourceRelative: document.sourceRelative,
    publicPath: family.buildIndexPath(locale),
    locale,
    cards: buildCollectionIndexCards(family.id, locale),
    sidebar: buildCollectionSidebar(family.id, locale)
  });

  // For the blog collection, attach the full sorted entry list so the
  // React blog listing component can render it without parsing markdown.
  if (family.id === "blog") {
    const blogEntries: BlogEntry[] = getBlogEntriesForLocale(locale);
    return { ...page, blogEntries };
  }

  return page;
}

async function loadCollectionArticlePage(
  family: CollectionFamilyDefinition,
  slugSegments: string[] | undefined,
  locale: Locale
): Promise<DocsPage | null> {
  if (!slugSegments?.length) {
    return null;
  }

  const sourceRelative = resolveCollectionPageSource(family.id, slugSegments, locale);
  if (!sourceRelative) {
    return null;
  }

  const publicPath = family.buildPagePath(slugSegments, locale);
  const page = await loadDocumentPage(sourceRelative, publicPath, locale);
  const continuation = buildCollectionContinuation({
    kind: family.pageKind,
    locale,
    currentPath: publicPath,
    index: buildIndexLink(requireDocument(family.indexSource, locale).sourceRelative, family.buildIndexPath(locale))
  });

  return {
    ...withContinuation(page, continuation),
    sidebar: buildCollectionSidebar(family.id, locale)
  };
}

export async function loadCollectionIndex(
  familyId: CollectionFamilyId,
  locale: Locale
): Promise<DocsPage> {
  const family = getCollectionFamilyById(familyId);
  if (!family) {
    throw new Error(`Unknown collection family: ${familyId}`);
  }

  const page = await loadCollectionIndexPage(family, locale);
  if (!page) {
    throw new Error(`Missing collection index page for ${familyId} (${locale})`);
  }

  return page;
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

  if (!slugSegments?.length) {
    return null;
  }

  return loadCollectionArticlePage(family, slugSegments, locale);
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
